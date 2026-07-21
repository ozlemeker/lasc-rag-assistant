from src.loader import load_documents
from src.chunking import create_chunks
from src.embedding import embed_documents

# 1. PDF'leri yükle
documents = load_documents("./docs")

# 2. Chunk'lara ayır
chunks = create_chunks(documents)

# 3. Chunk metinlerini al
texts = [chunk["text"] for chunk in chunks]

# 4. Embedding üret
embeddings = embed_documents(texts)

print(f"Pages      : {len(documents)}")
print(f"Chunks     : {len(chunks)}")
print(f"Embeddings : {len(embeddings)}")

if embeddings:
    print(f"Dimension  : {len(embeddings[0])}")

print("\nİlk chunk:")
print(chunks[0])

print("\nİlk embedding'in ilk 10 değeri:")
print(embeddings[0][:10])