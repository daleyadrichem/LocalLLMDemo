Usage
=====

LocalLLM
--------

The ``LocalLLM`` class is intentionally focused on one responsibility:
communicating with a local LLM backend and returning raw outputs.

Example::

  from llm_local import LocalLLM, LocalLLMConfig

  llm = LocalLLM(LocalLLMConfig(model="llama3.2:3b"))
  print(llm.generate("Explain what a neural network is."))

Persistent chat sessions
------------------------

Example::

  llm.start_chat(system_prompt="You are a helpful assistant.")
  reply = llm.send_chat_message("Hi! Who are you?")
  print(reply)

CodeService
-----------

``CodeService`` builds code-development workflows on top of ``LocalLLM``.
It takes a file's code content as a string and optional repository context.

Example::

  from llm_local import LocalLLM
  from llm_local.code_service import CodeService

  service = CodeService(code=code_text, context=repo_context, llm=LocalLLM())
  print(service.describe_class("MyClass"))

Workspace analysis
------------------

You can index all classes in a workspace and store their interface + summaries
in a JSON file. This can later be used to power documentation or tooling.

Example::

  from pathlib import Path
  from llm_local import LocalLLM
  from llm_local.workspace_index import ClassMetadataStore
  from llm_local.workspace_analyzer import WorkspaceAnalyzer

  store = ClassMetadataStore(json_path=Path("class_index.json"))
  analyzer = WorkspaceAnalyzer(root_dir=Path("."), llm=LocalLLM(), metadata_store=store)
  analyzer.analyze()
