from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=2)
result = model.invoke("What is the capital of India")

print(result.content)