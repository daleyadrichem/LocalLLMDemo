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

````

---

# 📁 `docs/guides/local-dev.md`

```markdown
# Local Development

## 1. Install Ollama

```bash
ollama pull llama3.2:3b
ollama serve
````

---

## 2. Install project

```bash
pip install -e .
```

Or with dev extras:

```bash
pip install -e ".[dev]"
```

---

## 3. Run API

```bash
uvicorn llm_local.api:app --reload
```

---

## Environment Variables

```bash
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3.2:3b
```