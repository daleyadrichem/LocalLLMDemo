from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from llm_local import LocalLLM, LocalLLMConfig
from llm_local.code_service import CodeService
from llm_local.diff_utils import apply_diff_with_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask for code suggestions/changes, review a generated diff, "
            "and optionally apply it via git."
        )
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Path to the workspace root (default: current directory).",
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the target Python file (relative to workspace or absolute).",
    )

    parser.add_argument(
        "--class",
        dest="class_name",
        required=True,
        help="Class name to target inside the file.",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Local LLM model to use (default: llama3.2:3b).",
    )

    parser.add_argument(
        "--mode",
        choices=["refactor", "add", "docstrings"],
        required=True,
        help=(
            "What kind of change to propose: "
            "'refactor' (refactor class), "
            "'add' (add functionality), "
            "'docstrings' (add/improve docstrings)."
        ),
    )

    parser.add_argument(
        "--description",
        "-d",
        default=None,
        help=(
            "Description of the change. If omitted, you'll be prompted "
            "interactively (recommended for demos)."
        ),
    )

    parser.add_argument(
        "--context",
        default="",
        help="Extra repo context to guide the LLM (optional).",
    )

    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="Skip confirmation prompt and apply the diff automatically (use carefully).",
    )

    return parser.parse_args()


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _prompt_if_missing(value: Optional[str], prompt: str) -> str:
    if value and value.strip():
        return value.strip()
    return input(prompt).strip()


def _confirm(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def main() -> None:
    args = parse_args()

    workspace_root = Path(args.workspace).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise SystemExit(f"Workspace root does not exist or is not a directory: {workspace_root}")

    file_path = Path(args.file).expanduser()
    if not file_path.is_absolute():
        file_path = (workspace_root / file_path).resolve()

    if not file_path.exists() or not file_path.is_file():
        raise SystemExit(f"Target file does not exist: {file_path}")

    # Use a relative path in the diff headers so `git apply` can locate the file.
    try:
        relative_file_path = str(file_path.relative_to(workspace_root))
    except ValueError:
        # If the file is outside the workspace root, patching may not behave as expected.
        # Keep the file name but warn.
        relative_file_path = str(file_path)
        print("⚠️ Warning: file is not inside the workspace root; patch apply may fail.")

    print("🧠 Code suggestion & apply tool")
    print(f"📁 Workspace: {workspace_root}")
    print(f"📄 Target file: {relative_file_path}")
    print(f"🧩 Target class: {args.class_name}")
    print(f"🤖 Model: {args.model}")
    print(f"🛠️ Mode: {args.mode}")
    print("-" * 70)

    description = _prompt_if_missing(
        args.description,
        "Describe what you want (e.g. 'refactor for readability', 'add method X', ...):\n> ",
    )

    llm = LocalLLM(LocalLLMConfig(model=args.model))
    if not llm.is_backend_available():
        raise SystemExit(
            "LLM backend is not reachable. Make sure Ollama is running "
            "(e.g. `uv run serve`) and try again."
        )

    original_code = _read_file(file_path)

    service = CodeService(
        code=original_code,
        context=args.context,
        llm=llm,
        language="python",
        file_path=relative_file_path,
    )

    print("🔎 Generating proposed changes (diff)...")

    if args.mode == "refactor":
        diff_text = service.refactor_class(
            class_name=args.class_name,
            description=description,
            return_diff=True,
        )
    elif args.mode == "add":
        diff_text = service.add_functionality_to_class(
            class_name=args.class_name,
            description=description,
            return_diff=True,
        )
    else:  # docstrings
        diff_text = service.generate_docstrings(
            class_name=args.class_name,
            return_diff=True,
        )

    print("\n" + "=" * 70)
    print("PROPOSED DIFF")
    print("=" * 70)
    print(diff_text.rstrip())
    print("=" * 70 + "\n")

    if args.auto_apply:
        do_apply = True
        print("⚠️ auto-apply enabled: will apply diff without confirmation.")
    else:
        do_apply = _confirm("Apply this diff to your workspace? [y/n]: ")

    if not do_apply:
        print("❌ Not applying diff. Exiting.")
        return

    # Safety check before applying
    print("✅ Checking if diff can be applied cleanly (git apply --check)...")
    try:
        # Reuse the same function but with a dry-run check is better;
        # our diff_utils uses `git apply -` by default. We'll do a check here
        # by invoking a tiny helper pattern: run check first.
        import subprocess

        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=diff_text.encode("utf-8"),
            cwd=workspace_root,
            capture_output=True,
            check=False,
        )

        if check.returncode != 0:
            stdout = check.stdout.decode("utf-8", errors="replace")
            stderr = check.stderr.decode("utf-8", errors="replace")
            print("❌ Patch check failed. The diff may not apply cleanly.")
            print("---- git apply --check stdout ----")
            print(stdout.strip() or "(empty)")
            print("---- git apply --check stderr ----")
            print(stderr.strip() or "(empty)")
            return

    except FileNotFoundError:
        raise SystemExit("git is not available on PATH. Install git to apply diffs.")
    except Exception as exc:
        raise SystemExit(f"Unexpected error during patch check: {exc}") from exc

    print("🧩 Applying diff...")
    apply_diff_with_git(diff_text, repo_root=workspace_root)

    print("✅ Diff applied successfully.")
    print("💡 Next steps:")
    print("  - Review: git diff")
    print("  - Run tests: uv run test (if you have a task) / pytest")
    print("  - Commit: git commit -am \"...\"")


if __name__ == "__main__":
    main()
