# ai-agents
[![Agentic SRE](https://github.com/skarj/ai-agents/actions/workflows/actions.yml/badge.svg)](https://github.com/skarj/ai-agents/actions/workflows/actions.yml)

```
kubectl create secret generic sre-agent-secrets \
  --from-literal=token="TELEGRAM_TOKEN" \
  --from-literal=chat-id="TELEGRAM_ID" \
  -n ai-agents
```

```
kubectl create secret generic github-write-creds \
  --namespace=argocd \
  --from-literal=creds=https://YOUR_GITHUB_USERNAME:YOUR_GITHUB_PAT@github.com/skarj/ai-agents-simple-k8s-agent.git
```
