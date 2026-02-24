FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install git (needed for HF sometimes)
RUN apt-get update && apt-get install -y git

# Download model during build
ENV HF_HOME=/app/hf_cache

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
model='Qwen/Qwen2-1.5B-Instruct'; \
AutoTokenizer.from_pretrained(model); \
AutoModelForCausalLM.from_pretrained(model)"

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]