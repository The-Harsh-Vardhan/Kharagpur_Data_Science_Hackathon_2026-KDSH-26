from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, using CPU only")

# Strong, stable embedding model
MODEL_NAME = "all-mpnet-base-v2"

# Determine device
if TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = "cpu"
    print("💻 Using CPU for embeddings")

_model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def embed_texts(texts, batch_size=32):
    """
    Converts texts into dense embeddings with GPU acceleration if available.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for encoding
    
    Returns:
        numpy array of normalized embeddings
    """
    print(f"\n🔢 Embedding {len(texts)} chunks on {DEVICE.upper()}...")
    
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE
    )
    
    if DEVICE == "cuda" and TORCH_AVAILABLE:
        print(f"⚡ GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return embeddings