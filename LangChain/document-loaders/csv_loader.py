from pathlib import Path
from langchain_community.document_loaders import CSVLoader

# Get current file directory
BASE_DIR = Path(__file__).resolve().parent

# CSV file path
csv_path = BASE_DIR / "Social_Network_Ads.csv"

# Load CSV
loader = CSVLoader(
    file_path=str(csv_path),
    encoding="utf-8"
)

# Load documents
docs = loader.load()

# Print total rows
print("Total Documents:", len(docs))

# Print second row
print("\n===== DOCUMENT 2 =====\n")
print(docs[1])