import os
from langchain_core.prompts import PromptTemplate

# Get project root (Routing_Orchestration)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_MAP = {
    "Beginner-Friendly": "beginner.txt",
    "Technical": "technical.txt",
    "Code-Oriented": "code.txt",
    "Mathematical": "mathematical.txt"
}

def load_prompt_by_style(style):
    file_name = PROMPT_MAP.get(style)

    if not file_name:
        raise ValueError(f"Invalid style: {style}")

    file_path = os.path.join(BASE_DIR, "prompts", file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    return PromptTemplate.from_template(template_str)