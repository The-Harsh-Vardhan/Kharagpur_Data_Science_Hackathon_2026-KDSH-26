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
from src.train_classifier import train_classifier
from src.gemini_llm import gemini_call_batch


DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


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
    print("🚀 KDSH'26 Track A - Optimized Training")
    print("="*60)
    
    overall_start = time.time()
    
    # Load training data
    print("\n[1/6] 📊 Loading training data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print(f"      ✓ Loaded {len(train_df)} training examples")
    
    # Load novels and build vector stores
    print("\n[2/6] 📚 Loading and processing novels...")
    print("      This will take 2-5 minutes depending on hardware")
    
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
    
    # Prepare all prompts for batch processing
    print("\n[3/6] 🔍 Retrieving evidence and preparing prompts...")
    all_prompts = []
    prompt_metadata = []  # Track which example/claim each prompt belongs to
    
    start_prep = time.time()
    
    with tqdm(total=len(train_df), desc="      Processing examples", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for idx, row in train_df.iterrows():
            backstory = row["content"]
            book_name = row["book_name"].lower()
            book = "castaways" if "castaways" in book_name else "monte_cristo"
            
            claims = split_backstory(backstory)
            store = novels[book]
            
            for claim in claims:
                # Retrieve evidence chunks for this claim
                claim_emb = embed_texts([claim])[0]
                evidence_chunks = store.search(claim_emb, k=5)
                
                for evidence in evidence_chunks:
                    prompt = create_contradiction_prompt(claim, evidence)
                    all_prompts.append(prompt)
                    prompt_metadata.append({
                        'example_idx': idx,
                        'claim': claim,
                        'evidence': evidence
                    })
            pbar.update(1)
    
    prep_time = time.time() - start_prep
    print(f"      ✓ Prepared {len(all_prompts)} prompts in {prep_time:.1f}s")
    print(f"      ℹ️  Average: {len(all_prompts)/len(train_df):.1f} prompts per example")
    
    # Batch process all prompts with rate limiting
    print("\n[4/6] 🤖 Processing LLM calls (this is the longest step)...")
    estimated_minutes = (len(all_prompts) * 5) / (3 * 60)  # 3 workers, ~5s per call
    print(f"      Estimated time: ~{estimated_minutes:.1f} minutes")
    print(f"      Total API calls: {len(all_prompts)}")
    print(f"      Free tier safe: ✓ (rate limited to 12 RPM)")
    
    start_llm = time.time()
    
    # Use max_workers=3 for free tier to avoid rate limits
    all_responses = gemini_call_batch(all_prompts, max_workers=3)
    
    llm_time = time.time() - start_llm
    print(f"\n      ✓ Completed {len(all_prompts)} API calls in {llm_time/60:.1f} minutes")
    print(f"      ℹ️  Average: {llm_time/len(all_prompts):.2f}s per call")
    
    # Aggregate results by example
    print("\n[5/6] 📈 Aggregating evidence scores...")
    features = []
    labels = []
    
    example_scores = {}  # Group scores by example index
    
    with tqdm(total=len(all_responses), desc="      Aggregating",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for meta, response in zip(prompt_metadata, all_responses):
            idx = meta['example_idx']
            score = contradiction_score(lambda p: response, meta['claim'], meta['evidence'])
            
            if idx not in example_scores:
                example_scores[idx] = []
            example_scores[idx].append(score)
            pbar.update(1)
    
    # Create feature vectors
    for idx in sorted(example_scores.keys()):
        aggregated = aggregate_scores(example_scores[idx])
        features.append([
            aggregated["max_score"],
            aggregated["mean_score"],
            aggregated["contradiction_count"]
        ])
        label = 1 if train_df.iloc[idx]["label"] == "consistent" else 0
        labels.append(label)
    
    print(f"      ✓ Created {len(features)} feature vectors")
    
    # Train classifier
    print("\n[6/6] 🤖 Training classifier...")
    clf = train_classifier(features, labels)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    
    total_time = time.time() - overall_start
    
    print(f"\n      ✓ Model saved to {model_path}")
    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print("="*60)
    print(f"\n⏱️  Time Breakdown:")
    print(f"    Novel processing:    {novel_time/60:>6.1f} min")
    print(f"    Evidence retrieval:  {prep_time/60:>6.1f} min")
    print(f"    LLM processing:      {llm_time/60:>6.1f} min")
    print(f"    " + "-"*35)
    print(f"    Total time:          {total_time/60:>6.1f} min")
    print(f"\n💰 API Usage: {len(all_prompts)} calls")
    print("="*60)


if __name__ == "__main__":
    main()
