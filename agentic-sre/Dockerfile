FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y sqlite3 curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    langgraph \
    langgraph-checkpoint-sqlite \
    langchain-ollama \
    langchain-community \
    mcp \
    ollama \
    pyTelegramBotAPI

# Create data directory for the SQLite state DB
RUN mkdir -p /app/data

COPY agent.py .

CMD ["python", "agent.py"]
