import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# Prompt for Tweet
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

# Prompt for LinkedIn
prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

# Hugging Face Model
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

# Individual chains
tweet_chain = prompt1 | model | parser
linkedin_chain = prompt2 | model | parser

# Parallel chain
parallel_chain = RunnableParallel(
    tweet=tweet_chain,
    linkedin=linkedin_chain
)

# Execute
result = parallel_chain.invoke({"topic": "AI"})

# Output
print("\n===== TWEET =====\n")
print(result["tweet"])

print("\n===== LINKEDIN POST =====\n")
print(result["linkedin"])