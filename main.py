from src.loader import load_documents
from src.chunking import create_chunks
from src.embedding import embed_documents
from src.vectordb import get_collection, index_chunks


def main(force: bool = False) -> None:
    print("=" * 50)
    print("Preparing vector database...")
    print("=" * 50)

    collection = get_collection()

    if collection.count() > 0 and not force:
        print(f"\nCollection already contains data ({collection.count()} chunks), skipping indexing.")
    else:
        print("\nLoading documents...")
        documents = load_documents("./docs")

        print("Creating chunks...")
        chunks = create_chunks(documents)

        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_documents(texts)

        print("Indexing ChromaDB...")
        index_chunks(chunks, embeddings, force=force)

        print()
        print(f"Pages      : {len(documents)}")
        print(f"Chunks     : {len(chunks)}")
        print(f"Embeddings : {len(embeddings)}")

    print("\nVector database is ready.")
    print("You can now run:")
    print("  python -m src.chat")
    print("or")
    print("  streamlit run ui.py")


if __name__ == "__main__":
    main()