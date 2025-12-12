from __future__ import annotations

from pathlib import Path

from llm_local import LocalLLM, LocalLLMConfig
from llm_local.workspace_index import ClassMetadataStore
from llm_local.workspace_analyzer import WorkspaceAnalyzer


def main() -> None:
    root_dir = Path(".").resolve()  # your repo root

    llm = LocalLLM(LocalLLMConfig(model="llama3.2:3b"))

    metadata_store = ClassMetadataStore(
        json_path=root_dir / "class_index.json",
    )

    repo_context = (
        "This is a Python project. The goal of the system is to provide local "
        "LLM utilities and development tools (summaries, tests, refactors, etc.)."
    )

    analyzer = WorkspaceAnalyzer(
        root_dir=root_dir,
        llm=llm,
        metadata_store=metadata_store,
        repo_context=repo_context,
    )

    analyzer.analyze()


if __name__ == "__main__":
    main()
