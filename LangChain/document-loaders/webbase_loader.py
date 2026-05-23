import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# Chat wrapper
model = ChatHuggingFace(llm=llm)

# Prompt Template
prompt = PromptTemplate(
    template="""
Answer the following question based on the given text.

Question:
{question}

Text:
{text}
""",
    input_variables=["question", "text"]
)

# Output Parser
parser = StrOutputParser()

# URL
url = "https://modelcontextprotocol.io/docs/getting-started/intro"

# Load Webpage
loader = WebBaseLoader(url)

docs = loader.load()

# Optional: Reduce text size
text_content = docs[0].page_content[:4000]

# Create Chain
chain = prompt | model | parser

# Invoke Chain
result = chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": text_content
})

print(result)