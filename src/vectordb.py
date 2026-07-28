"""
vectordb.py
Chunk'ları, embedding'lerini ve metadata'larını ChromaDB'ye kaydeder ve
gerektiğinde geri alır. Embedding uretmez -- bu embedding.py'nin isi.
"""
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult

from src.chunking import Chunk

COLLECTION_NAME = "lasc_documents"
CHROMA_PATH = "./chroma_db"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Persistent ChromaDB client'ini ilk cagrida olusturur, sonra ayni instance'i dondurur."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection() -> Collection:
    """Koleksiyon varsa acar, yoksa olusturur. Embedding fonksiyonu atanmaz --
    embedding'ler her zaman disaridan (embedding.py) saglanir."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def index_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    force: bool = False,
) -> None:
    """
    Chunk'lari embedding'leriyle birlikte koleksiyona ekler.

    Args:
        chunks: chunking.py'den gelen chunk listesi.
        embeddings: embed_documents() ile uretilmis, chunks ile ayni
            sirada ve ayni uzunlukta embedding listesi.
        force: True ise, koleksiyon dolu olsa bile once silinip yeniden
            indekslenir. False (varsayilan) ise koleksiyon zaten doluysa
            islem atlanir.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) ve embeddings ({len(embeddings)}) "
            "uzunluklari eslesmiyor."
        )

    collection = get_collection()

    if collection.count() > 0:
        if not force:
            print("Collection already contains data, skipping indexing.")
            return
        _get_client().delete_collection(COLLECTION_NAME)
        collection = get_collection()

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def search(query_embedding: list[float], top_k: int = 5) -> QueryResult:
    """Sorgu embedding'ine en yakin top_k chunk'i dondurur."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


def reset_database() -> None:
    """Koleksiyonu tamamen siler. Sonraki get_collection() cagrisi bos bir koleksiyon olusturur."""
    try:
        _get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass


if __name__ == "__main__":
    from src.loader import load_documents
    from src.chunking import create_chunks
    from src.embedding import embed_documents, embed_query

    documents = load_documents("./docs")
    chunks = create_chunks(documents)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_documents(texts)

    #reset_database()
    index_chunks(chunks, embeddings)

    collection = get_collection()
    print(f"Collection count: {collection.count()}")

    test_embedding = embed_query("What is the maximum rocket mass?")
    results = search(test_embedding, top_k=3)
    
    for doc, meta, dist in zip(  #denemelr daha düzenli output versin diye zip kullandım
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # for döngüsünün gövdesi. Yani her zip() turunda (yani her bir retrieval sonucunda — top_k=3 dediğin için 3 kez)
        #  bu 4 print() satırı çalışır: mesafe, kaynak (dosya+sayfa), ve chunk metninin ilk 300 karakteri basılır. 
        # 3 sonuç için toplam 3 kere tekrarlanan bir blok olacak, "="*60 ile aralarında görsel ayraç oluşacak.
        print("=" * 60)
        print(f"Distance : {dist:.4f}")
        print(f"Source   : {meta['file_name']} (Page {meta['page']})")
        print(doc[:300])