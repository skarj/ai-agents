# ai-agents
[![Agentic SRE](https://github.com/skarj/ai-agents/actions/workflows/actions.yml/badge.svg)](https://github.com/skarj/ai-agents/actions/workflows/actions.yml)

kubectl create secret generic sre-agent-secrets \
  --from-literal=token="TELEGRAM_TOKEN" \
  --from-literal=chat-id="TELEGRAM_ID" \
  -n ai-agents
