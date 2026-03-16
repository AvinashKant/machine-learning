"""
Custom embeddings class that uses raw HTTP requests to bypass LangChain's
tokenization batching, which sends arrays of token IDs instead of strings.
The HCL endpoint expects plain string 'input', not token arrays.
"""
import requests
import os
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

class HCLAzureEmbeddings(Embeddings):
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.endpoint = (
            "https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/"
            "deployments/ada/embeddings?api-version=2023-05-15"
        )
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
    
    def _embed(self, text: str) -> list[float]:
        """Embed a single string."""
        body = {"input": text}
        response = requests.post(self.endpoint, headers=self.headers, json=body, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed(text) for text in texts]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        return self._embed(text)
