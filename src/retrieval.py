"""
retrieval.py
Kullanici sorusunu alir, en alakali chunk'lari bulur ve LLM'in kullanacagi
context string'ini uretir. Bu modul Phi-3.5 / LLM ile hic ilgilenmez --
sadece embed_query() ve vectordb.search() akisini birlestirir.
"""
from chromadb.api.types import QueryResult

from src.embedding import embed_query
from src.vectordb import search

SEPARATOR = "-" * 50
DEFAULT_TOP_K = 5


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Soruyu embed eder, en alakali top_k chunk'i bulur ve bunlari
    tek bir context string'ine donusturur.
    """
    query_embedding = embed_query(question)
    results = search(query_embedding, top_k)
    return _build_context(results)


def _build_context(results: QueryResult) -> str:
    """Chroma query sonucunu, kaynak/sayfa bilgisiyle birlikte okunabilir
    tek bir context string'ine cevirir."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return "No relevant context found."

    blocks = []
    for doc, meta in zip(documents, metadatas):
        block = (
            f"Source: {meta['file_name']}\n"
            f"Page: {meta['page']}\n\n"
            f"{doc}"
        )
        blocks.append(block)

    return f"\n\n{SEPARATOR}\n\n".join(blocks)


if __name__ == "__main__":
    context = retrieve("What is the maximum rocket mass?")
    print(context)