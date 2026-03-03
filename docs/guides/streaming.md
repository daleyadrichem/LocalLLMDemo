# Streaming (Server-Sent Events)

The API supports streaming for:

- `/generate/stream`
- `/chat/stream`
- `/sessions/{id}/messages/stream`

Content type:

````

text/event-stream

````

---

## Example

```bash
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem"}'
```

---

## Events

The stream emits:

* `meta`
* `delta`
* `done`
* `error`

Example event:

```
event: delta
data: {"content": "Hello"}
```
