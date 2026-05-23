from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

# Current directory
BASE_DIR = Path(__file__).resolve().parent

# Books folder path
books_path = BASE_DIR / "books"

# Directory Loader
loader = DirectoryLoader(
    path=str(books_path),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Lazy Load Documents
docs = loader.lazy_load()

# Iterate Documents
for i, document in enumerate(docs, start=1):
    print(f"\n===== DOCUMENT {i} =====")
    print(document.metadata)