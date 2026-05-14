import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Prompt 1 - Generate Joke
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Chat wrapper
model = ChatHuggingFace(llm=llm)

# Output parser
parser = StrOutputParser()

# Prompt 2 - Explain Joke
prompt2 = PromptTemplate(
    template="Explain the following joke:\n\n{text}",
    input_variables=["text"]
)

# Create Chain
chain = (
    prompt1
    | model
    | parser
    | prompt2
    | model
    | parser
)

# Run chain
result = chain.invoke({"topic": "AI"})

print(result)