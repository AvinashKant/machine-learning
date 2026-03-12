# Create Virtual Environment
- python -m venv venv
- source venv/bin/activate   # Linux / Mac
- venv\Scripts\activate      # Windows

# Generate the requirements.txt
- pip freeze > requirements.txt

# Install all packages listed in a requirements.txt file
pip install -r requirements.txt


# Run JupyterLab
jupyter lab
