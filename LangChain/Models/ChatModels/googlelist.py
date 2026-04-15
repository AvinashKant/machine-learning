import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

for m in genai.list_models():
    print(m.name)