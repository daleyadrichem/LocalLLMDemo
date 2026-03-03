# Persistent Sessions

Sessions are stored in-memory.

⚠ Not durable. Intended for dev/single-instance use.

---

## Create Session

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are helpful."}'
````

---

## Send Message

```bash
curl -X POST http://localhost:8000/sessions/<id>/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## Streaming Session Message

```bash
curl -N -X POST \
  http://localhost:8000/sessions/<id>/messages/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'
```

---

## Delete Session

```bash
curl -X DELETE http://localhost:8000/sessions/<id>
```