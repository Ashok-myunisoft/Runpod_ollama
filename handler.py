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
    torch_dtype=torch.float32  # float32 for stability
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

        # Proper Qwen2 chat formatting
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        # Slice only newly generated tokens
        generated_tokens = outputs[0][inputs.shape[-1]:]

        clean = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return {"output": clean}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})