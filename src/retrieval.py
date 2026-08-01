"""
retrieval.py
Kullanici sorusunu alir, en alakali chunk'lari bulur ve LLM'in kullanacagi
context string'ini uretir. Bu modul Phi-3.5 / LLM ile hic ilgilenmez --
sadece embed_query() ve vectordb.search() akisini birlestirir.
"""
from typing import TypedDict
from chromadb.api.types import QueryResult

from src.embedding import embed_query
from src.vectordb import search

SEPARATOR = "-" * 50
DEFAULT_TOP_K = 5

class Source(TypedDict):
    file_name: str
    page: int
    chunk_indez: int
    distance: float
    text: str

class RetrievalResult(TypedDict):
    sources: list[Source]
    context: str


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Soruyu embed eder, en alakali top_k chunk'i bulur ve bunlari
    tek bir context string'ine donusturur.
    """
    query_embedding = embed_query(question)
    results = search(query_embedding, top_k)
    context, sources = _build_context_and_sources(results)
    return {"context": context, "sources": sources}


def _build_context_and_sources(results: QueryResult) -> tuple[str, list[Source]]:
    """Chroma query sonucunu, hem okunabilir context string'ine hem de
    kaynak metadata listesine cevirir."""
    documents = results["documents"][0]

    if not documents:
        return "No relevant context found."

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
 
    blocks: list[str] = []
    sources: list[Source] = []
 
    for doc, meta, dist in zip(documents, metadatas, distances):
        sources.append({
            "file_name": meta["file_name"],
            "page": meta["page"],
            "chunk_index": meta.get("chunk_index", -1),
            "distance": dist,
            "text": doc,
        })
 
        blocks.append(
            f"Source: {meta['file_name']}\n"
            f"Page: {meta['page']}\n\n"
            f"{doc}"
        )
 
    context = f"\n\n{SEPARATOR}\n\n".join(blocks)
    return context, sources
 
 
if __name__ == "__main__":
    result = retrieve("What is the maximum rocket mass?")
 
    print(result["context"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src['file_name']} | Page {src['page']} | Chunk {src['chunk_index']}")