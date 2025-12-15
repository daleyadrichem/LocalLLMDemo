from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llm_local import LocalLLM, LocalLLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a workspace using its class_index.json metadata."
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Path to the workspace root directory (default: current directory).",
    )

    parser.add_argument(
        "--index",
        default="class_index.json",
        help="Metadata index filename inside the workspace (default: class_index.json).",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Local LLM model to use (default: llama3.2:3b).",
    )

    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of most relevant classes to include as context (default: 12).",
    )

    return parser.parse_args()


def load_index(index_path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load the metadata index (class_index.json) into memory.
    """
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Index file not found: {index_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in index file: {index_path}\n{exc}") from exc


def tokenize(text: str) -> List[str]:
    """
    Very lightweight tokenization for relevance scoring (no external deps).
    """
    return [t.strip(".,:;!?()[]{}\"'").lower() for t in text.split() if t.strip()]


def score_entry(question_tokens: List[str], interface: str, summary: str) -> int:
    """
    Simple relevance scoring: count how many question tokens appear in
    interface+summary.
    """
    haystack = (interface + "\n" + summary).lower()
    return sum(1 for t in question_tokens if t and t in haystack)


def select_relevant_classes(
    index: Dict[str, Dict[str, Dict[str, str]]],
    question: str,
    top_k: int,
) -> List[Tuple[str, str, str, str]]:
    """
    Select top_k relevant (file_path, class_name, interface, summary) records.
    """
    q_tokens = tokenize(question)
    scored: List[Tuple[int, str, str, str, str]] = []

    for file_path, classes in index.items():
        for class_name, data in classes.items():
            interface = data.get("interface", "")
            summary = data.get("summary", "")
            s = score_entry(q_tokens, interface, summary)
            if s > 0:
                scored.append((s, file_path, class_name, interface, summary))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[: max(top_k, 1)]

    return [(fp, cn, iface, summ) for _, fp, cn, iface, summ in selected]


def build_llm_context(
    relevant: List[Tuple[str, str, str, str]],
    workspace_root: Path,
) -> str:
    """
    Build a compact context block from relevant class metadata.
    """
    if not relevant:
        return (
            "No relevant classes were matched from the metadata index. "
            "Answer with best-effort guidance and suggest how to improve the index "
            "(e.g., re-run analysis or broaden the question)."
        )

    blocks: List[str] = []
    for file_path, class_name, interface, summary in relevant:
        blocks.append(
            "\n".join(
                [
                    f"File: {file_path}",
                    f"Class: {class_name}",
                    "Interface:",
                    interface.strip(),
                    "Summary:",
                    summary.strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def answer_question(
    llm: LocalLLM,
    question: str,
    index: Dict[str, Dict[str, Dict[str, str]]],
    workspace_root: Path,
    top_k: int,
) -> str:
    """
    Use metadata + local LLM to answer the user's question.
    """
    relevant = select_relevant_classes(index=index, question=question, top_k=top_k)
    context_block = build_llm_context(relevant=relevant, workspace_root=workspace_root)

    prompt = f"""
You are an expert software engineering assistant.

You are answering questions about a codebase using a metadata index that contains:
- a class interface (signatures/docstrings)
- a short summary for each class

Important rules:
- Prefer pointing to the most relevant class(es) and file paths.
- If multiple candidates exist, list them and explain why.
- If the metadata is insufficient to answer confidently, say what is missing and
  suggest what to inspect next (e.g., which file/class to open).
- Do NOT invent classes or file paths that are not present in the metadata.

Metadata context:
--------------------
{context_block}
--------------------

Question:
{question}

Answer:
""".strip()

    return llm.generate(prompt=prompt, temperature=0.2)


def interactive_loop(
    llm: LocalLLM,
    index: Dict[str, Dict[str, Dict[str, str]]],
    workspace_root: Path,
    top_k: int,
) -> None:
    """
    Simple interactive Q&A loop.
    """
    print("💬 Workspace Q&A mode")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        print("🔎 Searching metadata and generating an answer...")
        reply = answer_question(
            llm=llm,
            question=question,
            index=index,
            workspace_root=workspace_root,
            top_k=top_k,
        )
        print("\n" + reply)


def main() -> None:
    args = parse_args()

    workspace_root = Path(args.workspace).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise SystemExit(f"Workspace path does not exist or is not a directory: {workspace_root}")

    index_path = workspace_root / args.index
    print(f"📁 Workspace root: {workspace_root}")
    print(f"📦 Loading metadata index: {index_path}")

    index = load_index(index_path)

    llm = LocalLLM(LocalLLMConfig(model=args.model))
    if not llm.is_backend_available():
        raise SystemExit(
            "LLM backend is not reachable. Make sure Ollama is running "
            "(e.g., `uv run serve`) and try again."
        )

    if args.question:
        print(f"❓ Question: {args.question}")
        print("🔎 Searching metadata and generating an answer...")
        reply = answer_question(
            llm=llm,
            question=args.question,
            index=index,
            workspace_root=workspace_root,
            top_k=args.top_k,
        )
        print("\n" + reply)
        return

    interactive_loop(
        llm=llm,
        index=index,
        workspace_root=workspace_root,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()