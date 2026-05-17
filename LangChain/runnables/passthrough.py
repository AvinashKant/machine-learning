import os
from dotenv import load_dotenv


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

# Load environment variables
load_dotenv()

# Prompt 1 - Joke Generator
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Chat wrapper
model = ChatHuggingFace(llm=llm)

# Output Parser
parser = StrOutputParser()

# Prompt 2 - Joke Explanation
prompt2 = PromptTemplate(
    template="Explain the following joke:\n\n{text}",
    input_variables=["text"]
)

# Joke Generation Chain
joke_gen_chain = prompt1 | model | parser

# Parallel Chain
parallel_chain = RunnableParallel(
    joke=RunnablePassthrough(),
    explanation=(
        {"text": RunnablePassthrough()}
        | prompt2
        | model
        | parser
    )
)

# Final Chain
final_chain = joke_gen_chain | parallel_chain

# Execute
result = final_chain.invoke({"topic": "cricket"})

# Output
print("\n===== JOKE =====\n")
print(result["joke"])

print("\n===== EXPLANATION =====\n")
print(result["explanation"])