from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from llm_local.code_service import CodeService
from llm_local.llm_client import LocalLLM
from llm_local.workspace_index import ClassMetadataStore

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceAnalyzer:
    """
    Analyze a workspace (directory tree) and build an index of class metadata.

    This class walks through all Python files under a given root directory,
    extracts class definitions, and for each class:

        - extracts the interface (using static analysis where possible)
        - generates a natural-language summary (using LocalLLM via CodeService)

    The results are stored in a ClassMetadataStore, which persists them to
    a JSON file.

    Parameters
    ----------
    root_dir :
        The root directory of the workspace to analyze.
    llm :
        A LocalLLM instance used by CodeService for LLM-powered tasks.
    metadata_store :
        The store responsible for persisting and retrieving class metadata.
    repo_context :
        Optional repository-level context string (e.g. from a README or
        architecture document) that may help the LLM generate better summaries.
    """

    root_dir: Path
    llm: LocalLLM
    metadata_store: ClassMetadataStore
    repo_context: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> None:
        """
        Analyze all Python files under the root directory and update the
        metadata store with interface and summary information for each class.

        Existing entries in the metadata store are overwritten for any
        (file_path, class_name) pairs that are re-analyzed.
        """
        logger.info("Starting workspace analysis under %s", self.root_dir)

        for file_path in self._iter_python_files():
            try:
                self._analyze_file(file_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to analyze file %s: %s", file_path, exc)

        logger.info("Saving metadata to %s", self.metadata_store.json_path)
        self.metadata_store.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_python_files(self) -> Iterable[Path]:
        """
        Yield all Python files under the root directory.

        This is intentionally simple; you can customize it later to ignore
        certain directories (e.g. .venv, build, dist) if desired.
        """
        for path in self.root_dir.rglob("*.py"):
            # Skip common virtualenv or cache directories
            if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            yield path

    def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a single Python file: extract classes, interfaces, and summaries.
        """
        logger.debug("Analyzing file %s", file_path)

        code = file_path.read_text(encoding="utf-8")
        class_names = self._extract_class_names(code)

        if not class_names:
            logger.debug("No classes found in %s", file_path)
            return

        relative_path = str(file_path.relative_to(self.root_dir))

        # Build file-specific context for the LLM: repo-level + file path
        file_context = self._build_file_context(relative_path)

        code_service = CodeService(
            code=code,
            context=file_context,
            llm=self.llm,
            language="python",
            file_path=relative_path,
        )

        for class_name in class_names:
            logger.debug("Processing class %s in %s", class_name, relative_path)

            # Interface: static analysis where possible (no LLM for Python)
            interface = code_service.get_class_interface(class_name)

            # Summary: LLM-powered
            summary = code_service.describe_class(class_name)

            self.metadata_store.set(
                file_path=relative_path,
                class_name=class_name,
                interface=interface,
                summary=summary,
            )

    def _extract_class_names(self, code: str) -> List[str]:
        """
        Extract class names from Python source code using the `ast` module.

        Only top-level class definitions are considered here. You can extend
        this later to include nested classes if needed.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            logger.warning("Failed to parse Python file for class extraction: %s", exc)
            return []

        class_names: List[str] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)

        return class_names

    def _build_file_context(self, relative_path: str) -> str:
        """
        Build a context string passed into CodeService for a given file.

        This combines repository-level context with file-specific information
        such as relative path. You can extend this later to include, for
        example, module docstrings or related files.
        """
        base = self.repo_context.strip()
        if base:
            base += "\n\n"

        base += f"File: {relative_path}"
        return base
