"""
chunking.py
loader.py'dan gelen sayfa bazlı dokümanları chunk'lara ayırır.
Her chunk, kaynağı olan sayfanın metadata'sını (file_name, page) korur.
"""
from typing import TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class Document(TypedDict):
    file_name: str
    page: int
    text: str


class ChunkMetadata(TypedDict):
    file_name: str
    page: int
    chunk_index: int


class Chunk(TypedDict):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


def create_chunks(documents: list[Document]) -> list[Chunk]:
    """
    Sayfa bazlı dokümanları chunk'lara böler.

    Args:
        documents: loader.py'dan gelen liste.

    Returns:
        [
            {
                "chunk_id": "chunk_000000",
                "text": "...",
                "metadata": {"file_name": "...", "page": 3, "chunk_index": 0}
            },
            ...
        ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Chunk] = []
    chunk_counter = 0

    for document in documents:
        page_chunks = _split_page(document, splitter)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:06d}",
                "text": chunk_text,
                "metadata": {
                    "file_name": document["file_name"],
                    "page": document["page"],
                    "chunk_index": chunk_index,
                },
            })
            chunk_counter += 1

    return chunks


def _split_page(document: Document, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Tek bir sayfanın text'ini chunk'lara böler, boş chunk'ları eler."""
    raw_chunks = splitter.split_text(document["text"])
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]


if __name__ == "__main__":
    from loader import load_documents

    docs = load_documents("./docs")
    chunks = create_chunks(docs)

    print(f"{len(docs)} sayfa -> {len(chunks)} chunk")
    for c in chunks[:3]:
        meta = c["metadata"]
        print(c["chunk_id"], "-", meta["file_name"], "- sayfa", meta["page"],
              "- chunk_index", meta["chunk_index"], "-", c["text"][:80])