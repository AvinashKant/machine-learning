from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "../data/dl-curriculum.pdf"

loader = PyPDFLoader(
    str(file_path)
)

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_documents(docs)

print(result[1].page_content)