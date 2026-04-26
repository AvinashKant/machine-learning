
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from utils.prompt_loader import load_prompt_by_style
from langchain_core.output_parsers import StrOutputParser

# ---------------- LOAD ENV ---------------- #
load_dotenv()

# ---------------- LLM SETUP ---------------- #
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)


st.set_page_config(page_title="Research Tool", page_icon="📄")

st.header('📄 Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


template = load_prompt_by_style(style_input)

chain = template | model | StrOutputParser()

if st.button('Summarize'):
    with st.spinner("Generating summary..."):

        try:

            template = load_prompt_by_style(style_input)
            chain = template | model

            result = chain.invoke({
                "paper_input": paper_input,
                "length_input": length_input
            })

            st.success("Summary generated ✅")
            st.write(result.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")