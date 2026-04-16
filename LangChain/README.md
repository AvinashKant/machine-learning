# Create Virtual Environment
- python -m venv LangChainVenv
- source venv/bin/activate   # Linux / Mac
- LangChainVenv\Scripts\activate      # Windows

# Generate the requirements.txt
- pip freeze > requirements.txt

# Install all packages listed in a requirements.txt file
pip install -r requirements.txt


- streamlit run Prompts\Routing_Orchestration\ui.py