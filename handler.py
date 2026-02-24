import os
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# -----------------------------
# Load Model Safely
# -----------------------------
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

except Exception as e:
    print(f"Model loading failed: {e}")
    raise e


# -----------------------------
# Handler Function
# -----------------------------
def handler(event):
    try:
        prompt = event.get("input", {}).get("prompt", "").strip()

        if not prompt:
            return {"error": "Prompt is empty."}

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt echo
        clean_response = full_output.replace(formatted_prompt, "").strip()

        return {"output": clean_response}

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Start RunPod
# -----------------------------
runpod.serverless.start({"handler": handler})