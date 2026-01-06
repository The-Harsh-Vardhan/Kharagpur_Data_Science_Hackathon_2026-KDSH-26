import os
import pickle
import pandas as pd
from tqdm import tqdm

from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.retrieval import VectorStore
from src.contradiction import contradiction_score
from src.aggregation import aggregate_scores
from src.train_classifier import train_classifier

from src.gemini_llm import gemini_call   # comment this if using another LLM


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


def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # Load novels once
    novels = {
        "castaways": build_vector_store(
            load_novel(os.path.join(DATA_DIR, "Books", "In search of the castaways.txt"))
        ),
        "monte_cristo": build_vector_store(
            load_novel(os.path.join(DATA_DIR, "Books", "The Count of Monte Cristo.txt"))
        )
    }

    features = []
    labels = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        backstory = row["content"]
        label = 1 if row["label"] == "consistent" else 0
        book_name = row["book_name"].lower()
        book = "castaways" if "castaways" in book_name else "monte_cristo"

        claims = split_backstory(backstory)
        store = novels[book]

        all_scores = []

        for claim in claims:
            claim_emb = embed_texts([claim])[0]
            evidence_chunks = store.search(claim_emb, k=5)

            for evidence in evidence_chunks:
                score = contradiction_score(
                    llm_call=gemini_call,
                    claim=claim,
                    evidence=evidence
                )
                all_scores.append(score)

        aggregated = aggregate_scores(all_scores)
        features.append([
            aggregated["max_score"],
            aggregated["mean_score"],
            aggregated["contradiction_count"]
        ])
        labels.append(label)

    clf = train_classifier(features, labels)

    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    main()
