from __future__ import annotations

import argparse
from pathlib import Path

from llm_local import LocalLLM, LocalLLMConfig
from llm_local.workspace_index import ClassMetadataStore
from llm_local.workspace_analyzer import WorkspaceAnalyzer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Python workspace and generate a JSON index containing "
            "class interfaces and summaries."
        )
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help=(
            "Path to the workspace root directory to analyze "
            "(default: current directory)."
        ),
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Name of the local LLM model to use (default: llama3.2:3b).",
    )

    parser.add_argument(
        "--output",
        default="class_index.json",
        help=(
            "Name of the output JSON file (stored inside the workspace root). "
            "Default: class_index.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workspace_root = Path(args.workspace).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise SystemExit(
            f"Workspace path does not exist or is not a directory: {workspace_root}"
        )

    output_path = workspace_root / args.output

    print("🔍 Starting workspace analysis")
    print(f"📁 Workspace root: {workspace_root}")
    print(f"🧠 LLM model: {args.model}")
    print(f"💾 Output file: {output_path}")
    print("-" * 60)

    llm = LocalLLM(
        LocalLLMConfig(model=args.model)
    )

    metadata_store = ClassMetadataStore(json_path=output_path)

    repo_context = (
        "This is a Python workspace being analyzed to extract class-level "
        "interfaces and summaries for documentation and tooling purposes."
    )

    analyzer = WorkspaceAnalyzer(
        root_dir=workspace_root,
        llm=llm,
        metadata_store=metadata_store,
        repo_context=repo_context,
    )

    # --- Run analysis ---
    analyzer.analyze()

    print("-" * 60)
    print("✅ Workspace analysis completed successfully")
    print(f"📄 Metadata written to: {output_path}")


if __name__ == "__main__":
    main()
