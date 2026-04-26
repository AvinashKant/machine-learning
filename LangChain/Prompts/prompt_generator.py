from langchain_core.prompts import PromptTemplate
import json

# Define template
template = PromptTemplate.from_template(
    """
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
)

# Save template (modern LangChain way)
with open("template.json", "w") as f:
    json.dump(template.model_dump(), f, indent=2)