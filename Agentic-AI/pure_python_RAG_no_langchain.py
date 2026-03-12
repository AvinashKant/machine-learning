import requests
import os
 

API_KEY = os.getenv("HAPI_KEY") 
 
# --- Endpoint config ---
API_URL = (
    "https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/"
    "deployments/gpt-4.1/chat/completions?api-version=2024-12-01-preview"
)
#
SYSTEM_PROMPT = """
You are a helpful AI agent.
- Answer clearly and concisely.
- Ask for clarification only when truly needed.
"""
 
def run_agent():
    print("Simple AI agent. Type 'quit' to exit.\n")
 
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Agent: Goodbye! 👋")
            break
 
        # Build request body (OpenAI-style)
        body = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0.7,
        }
 
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
 
        try:
            response = requests.post(API_URL, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            data = response.json()
            agent_reply = data["choices"][0]["message"]["content"].strip()
            print(f"Agent: {agent_reply}\n")
        except Exception as e:
            print(f"Error: {e}")
 
if __name__ == "__main__":
    run_agent()