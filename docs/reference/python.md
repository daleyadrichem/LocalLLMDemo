# Python API Reference

## LocalLLM

Main client class for interacting with a local backend.

### Basic Usage

```python
from llm_local import LocalLLM

llm = LocalLLM()

text = llm.generate("Explain transformers.")
print(text)
```

---

### Streaming

```python
for chunk in llm.generate_stream("Write a poem"):
    print(chunk, end="")
```

---

### Chat

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain RL."}
]

reply = llm.chat(messages)
print(reply)
```

---

### Model Lifecycle

```python
llm.list_models()
llm.pull_model("llama3.2:3b")
llm.delete_model("llama3.2:3b")
llm.show_model("llama3.2:3b")
```