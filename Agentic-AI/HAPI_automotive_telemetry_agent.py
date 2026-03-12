import requests
from typing import TypedDict, List, Dict
import json
import numpy as np
import os

# ─────────────────────────────────────────────
#  CONFIG  –  fill in your credentials
# ─────────────────────────────────────────────
API_KEY = os.getenv("HAPI_KEY") 
API_URL   = "https://api.openai.com/v1/chat/completions"
EMBED_URL = "https://api.openai.com/v1/embeddings"

CHAT_MODEL  = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-large"

HEADERS = {
    "api-key": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
class VehicleState(TypedDict):
    vehicle_id:     str
    telemetry:      Dict[str, float]
    anomaly:        str
    retrieved_docs: List[str]
    diagnosis:      str
    decision:       str


# ─────────────────────────────────────────────
#  TINY VECTOR STORE  (cosine similarity over FAISS-free embeddings)
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "High engine temperature may indicate coolant leak or radiator failure.",
    "Low battery voltage often suggests battery degradation.",
    "Brake pressure loss is usually caused by hydraulic fluid leakage.",
    "Brake system faults are safety critical and require immobilization.",
]

def get_embedding(text: str) -> List[float]:
    """Fetch a single embedding vector via the REST API."""
    body = {"model": EMBED_MODEL, "input": text}
    resp = requests.post(EMBED_URL, headers=HEADERS, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

def build_vector_store() -> List[Dict]:
    """Embed every document in the knowledge base once at startup."""
    print("Building knowledge-base embeddings …")
    store = []
    for doc in KNOWLEDGE_BASE:
        store.append({"content": doc, "embedding": get_embedding(doc)})
    print("Done.\n")
    return store

def retrieve(query: str, store: List[Dict], k: int = 2) -> List[str]:
    """Return the k most-similar documents to the query."""
    q_emb = get_embedding(query)
    scored = [(cosine_similarity(q_emb, item["embedding"]), item["content"])
              for item in store]
    scored.sort(reverse=True)
    return [content for _, content in scored[:k]]


# ─────────────────────────────────────────────
#  LLM HELPER
# ─────────────────────────────────────────────
def chat(system: str, user: str, temperature: float = 0.0) -> str:
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
    }
    resp = requests.post(API_URL, headers=HEADERS, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────
#  LANGGRAPH NODES
# ─────────────────────────────────────────────
def telemetry_node(state: VehicleState) -> VehicleState:
    """Entry point – pass telemetry through unchanged."""
    return state

def anomaly_node(state: VehicleState) -> VehicleState:
    """Rule-based anomaly detection on raw sensor values."""
    t = state["telemetry"]
    if t["engine_temp"] > 110:
        state["anomaly"] = "High engine temperature"
    elif t["battery_voltage"] < 11.5:
        state["anomaly"] = "Low battery voltage"
    elif t["brake_pressure"] < 20:
        state["anomaly"] = "Brake pressure loss"
    else:
        state["anomaly"] = "Normal"
    return state

def retrieval_node(state: VehicleState, store: List[Dict]) -> VehicleState:
    """RAG retrieval: find relevant knowledge for the detected anomaly."""
    state["retrieved_docs"] = retrieve(state["anomaly"], store)
    return state

def diagnosis_node(state: VehicleState) -> VehicleState:
    """Ask the LLM for the most likely root cause."""
    context = "\n".join(state["retrieved_docs"])
    system  = "You are an automotive diagnostics expert."
    user    = f"""
Telemetry:
{json.dumps(state['telemetry'], indent=2)}

Detected anomaly:
{state['anomaly']}

Relevant knowledge:
{context}

Identify the most likely root cause.
""".strip()
    state["diagnosis"] = chat(system, user)
    return state

def decision_node(state: VehicleState) -> VehicleState:
    """Ask the LLM for the safest single action to take."""
    system = "You are a fleet safety AI."
    user   = f"""
Diagnosis:
{state['diagnosis']}

Decide the safest action.
Constraints:
- Human safety first
- Vehicle protection second
- Cost last

Respond with ONE action.
""".strip()
    state["decision"] = chat(system, user, temperature=0.3)
    return state

def report_node(state: VehicleState) -> VehicleState:
    """Print the final incident report."""
    print("\n===== VEHICLE INCIDENT REPORT =====")
    print("Vehicle ID :", state["vehicle_id"])
    print("Telemetry  :", state["telemetry"])
    print("Anomaly    :", state["anomaly"])
    print("\nDiagnosis  :\n", state["diagnosis"])
    print("\nDecision   :\n", state["decision"])
    print("===================================\n")
    return state


# ─────────────────────────────────────────────
#  LANGGRAPH PIPELINE  (manual sequential graph)
# ─────────────────────────────────────────────
def run_pipeline(initial_state: VehicleState, store: List[Dict]) -> VehicleState:
    """
    Execute the six-node graph in order:
        telemetry → anomaly → retrieve → diagnose → decide → report → END
    """
    state = initial_state
    state = telemetry_node(state)
    state = anomaly_node(state)
    state = retrieval_node(state, store)
    state = diagnosis_node(state)
    state = decision_node(state)
    state = report_node(state)
    return state


# ─────────────────────────────────────────────
#  INTERACTIVE AGENT LOOP
# ─────────────────────────────────────────────
AGENT_SYSTEM = """
You are an automotive telemetry AI agent.

When the user provides vehicle sensor data (engine_temp, battery_voltage,
brake_pressure) you run a full diagnostic pipeline and report the findings.

If the user just wants to chat, answer clearly and concisely.
Ask for clarification only when truly needed.
"""

def parse_telemetry_from_input(user_input: str) -> Dict[str, float] | None:
    """
    Simple heuristic: if the user supplies three numbers we treat them as
    engine_temp, battery_voltage, brake_pressure respectively.
    Otherwise return None (plain chat).
    """
    tokens = user_input.replace(",", " ").split()
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            pass
    if len(nums) == 3:
        return {
            "engine_temp":      nums[0],
            "battery_voltage":  nums[1],
            "brake_pressure":   nums[2],
        }
    return None

def run_agent():
    print("Automotive Telemetry Agent")
    print("  • Type 'quit' or 'exit' to leave.")
    print("  • To run a diagnostic, enter three numbers:  <engine_temp>  <battery_voltage>  <brake_pressure>")
    print("  • Otherwise just chat.\n")

    # Build the knowledge-base vector store once at startup
    store = build_vector_store()

    vehicle_counter = 1

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Agent: Goodbye! 👋")
            break

        telemetry = parse_telemetry_from_input(user_input)

        if telemetry:
            # ── Telemetry diagnostic path ──────────────────────────────────
            vehicle_id = f"CAR-{vehicle_counter:04d}"
            vehicle_counter += 1
            print(f"\nAgent: Running diagnostic for {vehicle_id} …")
            initial_state: VehicleState = {
                "vehicle_id":     vehicle_id,
                "telemetry":      telemetry,
                "anomaly":        "",
                "retrieved_docs": [],
                "diagnosis":      "",
                "decision":       "",
            }
            try:
                run_pipeline(initial_state, store)
            except Exception as e:
                print(f"Error during pipeline: {e}\n")
        else:
            # ── Plain chat path ────────────────────────────────────────────
            try:
                reply = chat(AGENT_SYSTEM, user_input, temperature=0.7)
                print(f"Agent: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    run_agent()
