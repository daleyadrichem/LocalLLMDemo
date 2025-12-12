Getting started
===============

Requirements
------------

- Python 3.10+
- Ollama installed and available on your PATH

Ollama setup using uv tasks
---------------------------

This project uses uv tasks to pull a model and start the local server.

Pull the model::

  ollama pull llama3.2:3b

Start the Ollama server::

  ollama serve

The server is expected to be available at::

  http://localhost:11434

Install the package
-------------------

From the repository root::

  uv pip install -e .

Quick sanity check
------------------

In Python::

  from llm_local import LocalLLM
  llm = LocalLLM()
  print(llm.is_backend_available())
