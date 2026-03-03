# HTTP API Reference

Interactive docs are available at:

- `/docs` (Swagger UI)
- `/redoc`

---

## Core Endpoints

### System
- GET /health

### Models
- GET /models
- POST /models/pull
- DELETE /models/{name}
- GET /models/{name}

### Generation
- POST /generate
- POST /generate/stream
- POST /chat
- POST /chat/stream

### Sessions
- POST /sessions
- GET /sessions/{id}
- GET /sessions/{id}/history
- DELETE /sessions/{id}
- POST /sessions/{id}/messages
- POST /sessions/{id}/messages/stream