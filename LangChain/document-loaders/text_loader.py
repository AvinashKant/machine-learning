import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "../data/cricket.txt"

# Load environment variables
load_dotenv()

# Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Chat Wrapper
model = ChatHuggingFace(llm=llm)

# Prompt Template
prompt = PromptTemplate(
    template="""
Write a summary for the following text:

{text}
""",
    input_variables=["text"]
)

# Output Parser
parser = StrOutputParser()

# File Loader
loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

# Load Documents
docs = loader.load()

# Debug Information
print("Document Type:", type(docs))
print("Number of Documents:", len(docs))

print("\n===== CONTENT =====\n")
print(docs[0].page_content)

print("\n===== METADATA =====\n")
print(docs[0].metadata)

# Create Chain
chain = prompt | model | parser

# Invoke Chain
result = chain.invoke({
    "text": docs[0].page_content
})

print("\n===== SUMMARY =====\n")
print(result)