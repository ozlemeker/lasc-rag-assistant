#pdf okuma
"""
loader.py

PDF dosyalarını okuyup sayfa bazlı metin + metadata çıkaran modül.
Chunking aşamasında file_name ve page bilgisini kaybetmemek için
her sayfa ayrı bir dict olarak döndürülür.
"""

import os
import fitz  # PyMuPDF


def load_documents(folder_path: str) -> list[dict]:
    """
    folder_path içindeki tüm .pdf dosyalarını sayfa sayfa okur.

    Returns:
        [
            {"file_name": "ornek.pdf", "page": 1, "text": "..."},
            {"file_name": "ornek.pdf", "page": 2, "text": "..."},
            ...
        ]
    """
    documents = []

    pdf_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")

    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        documents.extend(_load_single_pdf(file_path, file_name))

    return documents


def _load_single_pdf(file_path: str, file_name: str) -> list[dict]:
    """Tek bir PDF'i açar, her sayfayı ayrı bir dict olarak çıkarır."""
    pages = []

    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text().strip()

            # Boş sayfaları (örn. sadece görsel içeren) atla
            if not text:
                continue

            pages.append({
                "file_name": file_name,
                "page": page_index + 1,  # 1-indexed, kullanıcıya gösterim için daha mantıklı
                "text": text,
                "source": file_name,
            })

    return pages


if __name__ == "__main__":
    # Basit test / örnek kullanım
    folder = "./docs"
    docs = load_documents(folder)
    print(f"{len(docs)} sayfa yüklendi.")
    for d in docs[:3]:
        print(d["file_name"], "- sayfa", d["page"], "-", d["text"][:80])