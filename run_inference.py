import os
import pickle
import pandas as pd
import time
from tqdm import tqdm

from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.retrieval import VectorStore
from src.contradiction import contradiction_score
from src.aggregation import aggregate_scores
from src.gemini_llm import gemini_call_batch


DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_FILE = "submission.csv"


def load_novel(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vector_store(novel_text):
    chunks = chunk_text(novel_text)
    embeddings = embed_texts(chunks)
    store = VectorStore(embedding_dim=len(embeddings[0]))
    store.add(embeddings, chunks)
    return store


def split_backstory(backstory):
    return [s.strip() for s in backstory.split(".") if len(s.strip()) > 5]


def create_contradiction_prompt(claim, evidence):
    """Create evidence-grounded prompt per problem statement guidelines"""
    return f"""You are a logical consistency checker.

Claim:
{claim}

Novel Evidence:
{evidence}

Instructions:
- Use ONLY the provided evidence.
- Do NOT infer missing facts.
- If the evidence is insufficient, answer NEUTRAL.

Respond with exactly ONE word:
CONTRADICT, SUPPORT, or NEUTRAL"""


def main():
    print("="*60)
    print("🚀 KDSH'26 Track A - Optimized Inference")
    print("="*60)
    
    overall_start = time.time()
    
    # Load test data and model
    print("\n[1/5] 📊 Loading test data and model...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"      ✓ Loaded {len(test_df)} test examples")
    
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    print("      ✓ Loaded trained classifier")
    
    # Load novels
    print("\n[2/5] 📚 Loading and processing novels...")
    novel_start = time.time()
    novels = {
        "castaways": build_vector_store(
            load_novel(os.path.join(DATA_DIR, "Books", "In search of the castaways.txt"))
        ),
        "monte_cristo": build_vector_store(
            load_novel(os.path.join(DATA_DIR, "Books", "The Count of Monte Cristo.txt"))
        )
    }
    novel_time = time.time() - novel_start
    print(f"      ✓ Completed in {novel_time/60:.1f} minutes")
    
    # Prepare all prompts
    print("\n[3/5] 🔍 Retrieving evidence and preparing prompts...")
    all_prompts = []
    prompt_metadata = []
    
    with tqdm(total=len(test_df), desc="      Processing examples",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for idx, row in test_df.iterrows():
            backstory = row["content"]
            book_name = row["book_name"].lower()
            book = "castaways" if "castaways" in book_name else "monte_cristo"
            
            claims = split_backstory(backstory)
            store = novels[book]
            
            for claim in claims:
                claim_emb = embed_texts([claim])[0]
                evidence_chunks = store.search(claim_emb, k=5)
                
                for evidence in evidence_chunks:
                    prompt = create_contradiction_prompt(claim, evidence)
                    all_prompts.append(prompt)
                    prompt_metadata.append({'example_idx': idx})
            pbar.update(1)
    
    print(f"      ✓ Prepared {len(all_prompts)} prompts")
    
    # Batch process
    print("\n[4/5] 🤖 Processing LLM calls...")
    estimated_minutes = (len(all_prompts) * 5) / (3 * 60)
    print(f"      Estimated time: ~{estimated_minutes:.1f} minutes")
    
    llm_start = time.time()
    all_responses = gemini_call_batch(all_prompts, max_workers=3)
    llm_time = time.time() - llm_start
    
    print(f"\n      ✓ Completed in {llm_time/60:.1f} minutes")
    
    # Aggregate and predict
    print("\n[5/5] 📈 Aggregating and making predictions...")
    example_scores = {}
    
    with tqdm(total=len(all_responses), desc="      Aggregating",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for meta, response in zip(prompt_metadata, all_responses):
            idx = meta['example_idx']
            # Parse response
            mapping = {"CONTRADICT": 1.0, "NEUTRAL": 0.3, "SUPPORT": 0.0}
            score = mapping.get(response.strip().upper(), 0.3)
            
            if idx not in example_scores:
                example_scores[idx] = []
            example_scores[idx].append(score)
            pbar.update(1)
    
    predictions = []
    for idx in sorted(example_scores.keys()):
        aggregated = aggregate_scores(example_scores[idx])
        feature_vector = [[
            aggregated["max_score"],
            aggregated["mean_score"],
            aggregated["contradiction_count"]
        ]]
        label = int(clf.predict(feature_vector)[0])
        predictions.append(label)
    
    print(f"      ✓ Generated {len(predictions)} predictions")
    
    # Save results
    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": predictions
    })
    
    submission.to_csv(OUTPUT_FILE, index=False)
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("🎉 Inference completed successfully!")
    print("="*60)
    print(f"\n⏱️  Time Breakdown:")
    print(f"    Novel processing:    {novel_time/60:>6.1f} min")
    print(f"    LLM processing:      {llm_time/60:>6.1f} min")
    print(f"    " + "-"*35)
    print(f"    Total time:          {total_time/60:>6.1f} min")
    print(f"\n📄 Results saved to: {OUTPUT_FILE}")
    print(f"💰 API calls made: {len(all_prompts)}")
    print("="*60)


if __name__ == "__main__":
    main()
