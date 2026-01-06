from sentence_transformers import SentenceTransformer

# Strong, stable embedding model
MODEL_NAME = "all-mpnet-base-v2"

_model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    """
    Converts a list of texts into dense embeddings.
    """
    return _model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )