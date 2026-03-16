from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings import HCLAzureEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("data/policy.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"Loaded {len(chunks)} chunks from policy.pdf")

embeddings = HCLAzureEmbeddings()

print("Testing single embed...")
test = embeddings.embed_query("test")
print(f"Single embed OK: vector size = {len(test)}")

print("Building vectorstore...")
vector_db = FAISS.from_documents(chunks, embeddings)

vector_db.save_local("vectorstore")

print("Documents embedded and stored successfully")