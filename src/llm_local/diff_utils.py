from __future__ import annotations

import subprocess
from pathlib import Path


def apply_diff_with_git(diff_text: str, repo_root: str | Path = ".") -> None:
    """
    Apply a unified diff to a git repository using `git apply`.

    Parameters
    ----------
    diff_text :
        The unified diff / patch text.
    repo_root :
        Path to the root of the git repository (defaults to current directory).

    Raises
    ------
    RuntimeError
        If `git apply` fails.
    """
    repo_root_path = Path(repo_root)

    # We invoke `git apply` and pass the diff via stdin.
    process = subprocess.run(
        ["git", "apply", "-"],
        input=diff_text.encode("utf-8"),
        cwd=repo_root_path,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"git apply failed with code {process.returncode}:\n"
            f"STDOUT:\n{process.stdout.decode('utf-8')}\n"
            f"STDERR:\n{process.stderr.decode('utf-8')}"
        )
