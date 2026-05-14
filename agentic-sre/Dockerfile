FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    langgraph \
    langgraph-checkpoint-sqlite \
    langchain-ollama \
    langchain-community \
    mcp \
    ollama \
    pyTelegramBotAPI

# Directory for the SQLite state DB
RUN mkdir -p /app/data

COPY agent.py .

CMD ["python", "agent.py"]
