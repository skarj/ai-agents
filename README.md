# ai-agents

kubectl create secret generic sre-agent-secrets \
  --from-literal=token="YOUR_TELEGRAM_TOKEN" \
  --from-literal=chat-id="YOUR_TELEGRAM_ID" \
  -n ai-agents
