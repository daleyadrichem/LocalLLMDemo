# Docker Quickstart

## Start everything

```bash
docker compose up --build
````

This starts:

* Ollama backend
* Automatic model pull
* FastAPI service

API runs at:

[http://localhost:8000](http://localhost:8000)

---

## Test the API

Health check:

```bash
curl http://localhost:8000/health
```

Generate text:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain neural networks"}'
```