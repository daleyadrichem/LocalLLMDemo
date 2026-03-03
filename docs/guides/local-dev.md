# Local Development

## 1. Install Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

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