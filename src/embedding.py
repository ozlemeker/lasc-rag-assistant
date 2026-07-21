"""
embedding.py
BGE modeli ile doküman ve sorgu embedding'lerini üretir.

Sorumluluklar:
- Model yönetimi: tek bir model örneği, lazy loading ile yüklenir.
- Belge embedding'i: batch halinde, normalize edilmiş embedding üretir.
- Sorgu embedding'i: query prefix'ini otomatik ekler, normalize edilmiş embedding döndürür.
- Dönüş tipi: her zaman list[float] / list[list[float]] -- çağıran taraf
  (retrieval.py, vectordb.py) modelin NumPy döndürdüğünü, prefix veya
  normalization kullandığını bilmek zorunda kalmaz.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Modeli ilk çağrıda yükler, sonraki çağrılarda aynı instance'ı döndürür."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(query: str) -> list[float]:
    """Kullanıcı sorgusunu BGE'nin beklediği query prefix'i ile embed eder."""
    model = get_model()
    text = QUERY_PREFIX + query
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_documents(documents: list[str]) -> list[list[float]]:
    """Chunk metinlerini batch halinde, prefix'siz embed eder."""
    if not documents:
        return []

    model = get_model()
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


if __name__ == "__main__":
    query_vector = embed_query("What is the maximum rocket mass?")
    doc_vectors = embed_documents([
        "Maximum rocket mass is 25 kg.",
        "The competition takes place in Nevada.",
    ])

    print(f"Query vector boyutu: {len(query_vector)}")
    print(f"Doküman sayısı: {len(doc_vectors)}, her biri {len(doc_vectors[0])} boyutlu")