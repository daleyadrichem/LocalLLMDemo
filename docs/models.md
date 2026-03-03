# Model Management

## List Models

```bash
curl http://localhost:8000/models
````

---

## Pull Model

```bash
curl -X POST http://localhost:8000/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:3b"}'
```

---

## Delete Model

```bash
curl -X DELETE http://localhost:8000/models/llama3.2:3b
```

---

## Show Model Info

```bash
curl http://localhost:8000/models/llama3.2:3b
```