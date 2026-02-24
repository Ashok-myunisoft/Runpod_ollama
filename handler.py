import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

def handler(event):
    try:
        prompt = event.get("input", {}).get("prompt", "").strip()

        if not prompt:
            return {"error": "Empty prompt"}

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=False,
                
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean = response.replace(formatted, "").strip()

        return {"output": clean}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
