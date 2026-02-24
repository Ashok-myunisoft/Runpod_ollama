FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y curl python3 python3-pip

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]