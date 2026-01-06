import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def chunk_text(text, max_tokens=800, overlap=100):
    """
    Splits long text into overlapping chunks based on sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_len = 0

    for sent in sentences:
        tokens = sent.split()
        sent_len = len(tokens)

        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            if overlap > 0:
                current_chunk = current_chunk[-overlap:]
                current_len = len(" ".join(current_chunk).split())
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks