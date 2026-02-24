import runpod
import requests
import time
import subprocess
import os

# Start Ollama server when container starts
subprocess.Popen(["ollama", "serve"])

# Wait a bit for server to start
time.sleep(5)

# Pull Qwen model (only first time)
subprocess.run(["ollama", "pull", "qwen:7b"])

def handler(event):
    prompt = event["input"]["prompt"]

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    return {"output": response.json()["response"]}

runpod.serverless.start({"handler": handler})

