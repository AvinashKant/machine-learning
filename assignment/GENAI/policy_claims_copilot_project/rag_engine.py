from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from embeddings import HCLAzureEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY", "")

embeddings = HCLAzureEmbeddings()

db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

llm = AzureChatOpenAI(
    azure_endpoint="https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/gpt-4.1/chat/completions?api-version=2024-12-01-preview",
    azure_deployment="gpt-4.1",
    openai_api_version="2024-12-01-preview",
    api_key=openai_api_key,
    default_headers={"api-key": openai_api_key},
    temperature=0
)

def ask_question(query):
    docs = db.similarity_search(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Use the policy information below to answer.

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.content