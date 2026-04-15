from huggingface_hub import HfApi

api = HfApi()

def is_chat_model(model):
    keywords = ["chat", "instruct", "assistant", "dialog"]
    
    # Check model ID
    if any(k in model.modelId.lower() for k in keywords):
        return True
    
    # Check tags safely (if available)
    if hasattr(model, "tags") and model.tags:
        if any(k in tag.lower() for tag in model.tags for k in keywords):
            return True

    return False


def main():
    print("\n🔹 CHAT MODELS (Filtered)")

    models = api.list_models(
        filter="text-generation",
        sort="downloads",
        limit=50
    )

    chat_models = [m.modelId for m in models if is_chat_model(m)]

    for m in chat_models[:10]:
        print("-", m)


main()