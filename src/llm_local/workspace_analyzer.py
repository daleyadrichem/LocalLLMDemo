"""
Workspace Analyzer

This module analyzes a Python workspace (directory tree) and generates structured
metadata for:

- Classes (interfaces + LLM summaries)
- Top-level (module) functions that are not part of any class (interfaces + LLM summaries)
- Module-level overview (interface + LLM summary)

It produces two outputs:

1) Hierarchical per-folder JSON:
   - Writes a metadata.json into every folder
   - Each folder JSON embeds its subfolders under `subfolders`
   - Each folder JSON includes metadata for files directly in that folder

2) Flat ClassMetadataStore (backward compatibility):
   - Stores entries keyed by (file_path, class_name)
   - Non-class symbols are stored using reserved names/prefixes:
       "__module__"                -> module-level
       "func:<function_name>"      -> top-level functions
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from llm_local.code_service import CodeService
from llm_local.llm_client import LocalLLM
from llm_local.workspace_index import ClassMetadataStore

logger = logging.getLogger(__name__)

#: Reserved symbol name used for module-level metadata
MODULE_KEY = "__module__"

#: Reserved symbol name used to group top-level functions inside per-folder JSON
FUNCTIONS_KEY = "__functions__"

#: Prefix used when storing top-level functions into the flat metadata store
FUNCTION_STORE_PREFIX = "func:"


@dataclass
class FolderNode:
    """
    Represents a single folder in the workspace hierarchy.

    Attributes
    ----------
    path :
        Relative path from the workspace root ('.' for root).
    files :
        Mapping:
            filename -> symbol_name -> {"interface": ..., "summary": ...}

        `symbol_name` can be:
          - a class name
          - "__module__" for module-level analysis
          - "__functions__" containing a dict of functions, where each function
            maps to {"interface", "summary"}.
    subfolders :
        Mapping of subfolder name -> FolderNode.
    """

    path: str
    files: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    subfolders: Dict[str, "FolderNode"] = field(default_factory=dict)


@dataclass
class WorkspaceAnalyzer:
    """
    Analyze a workspace (directory tree) and build structured metadata.

    For each Python file:
      - Always generates module-level interface + summary
      - Extracts and analyzes each top-level class
      - Extracts and analyzes each top-level function not inside a class

    Parameters
    ----------
    root_dir :
        Root directory of the workspace to analyze.
    llm :
        LocalLLM instance used for LLM-powered summaries.
    metadata_store :
        Flat metadata store (backward compatibility).
    repo_context :
        Optional repository-level context text for improving LLM summaries.
    folder_metadata_filename :
        Filename to write into each folder (default: "metadata.json").
    """

    root_dir: Path
    llm: LocalLLM
    metadata_store: ClassMetadataStore
    repo_context: str = ""
    folder_metadata_filename: str = "metadata.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> None:
        """
        Analyze all Python files under `root_dir`.

        Produces:
          - Hierarchical per-folder metadata JSON files
          - Flat metadata store saved to disk
        """
        logger.info("Starting workspace analysis under %s", self.root_dir)

        tree = FolderNode(path=".")

        for file_path in self._iter_python_files():
            try:
                self._analyze_file_into_tree(file_path, tree)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to analyze file %s: %s", file_path, exc)

        self._write_folder_jsons(tree)

        logger.info("Saving flat metadata to %s", self.metadata_store.json_path)
        print(f"💾 Writing flat metadata to {self.metadata_store.json_path}")
        self.metadata_store.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_python_files(self) -> Iterable[Path]:
        """
        Yield all Python source files under the root directory.

        Skips common virtualenv and cache directories.
        """
        for path in self.root_dir.rglob("*.py"):
            if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            yield path

    def _analyze_file_into_tree(self, file_path: Path, tree: FolderNode) -> None:
        """
        Analyze a single Python file and store results in:
          - hierarchical folder tree
          - flat metadata store

        Analysis includes:
          - module-level interface + summary (always)
          - class-level interface + summary (for each top-level class)
          - function-level interface + summary (for each top-level function)
        """
        print(f"📄 Analyzing file: {file_path}")

        code = file_path.read_text(encoding="utf-8")
        class_names, function_names = self._extract_top_level_symbols(code)

        relative_path = file_path.relative_to(self.root_dir)
        rel_file_str = str(relative_path)
        rel_dir_str = str(relative_path.parent) if relative_path.parent != Path(".") else "."

        file_context = self._build_file_context(rel_file_str)

        code_service = CodeService(
            code=code,
            context=file_context,
            llm=self.llm,
            language="python",
            file_path=rel_file_str,
        )

        node = self._get_or_create_folder_node(tree, rel_dir_str)
        file_key = relative_path.name
        node.files.setdefault(file_key, {})

        # --------------------------------------------------------------
        # Module-level analysis (ALWAYS)
        # --------------------------------------------------------------
        module_interface = code_service.get_module_interface()
        module_summary = code_service.describe_module()

        node.files[file_key][MODULE_KEY] = {
            "interface": module_interface,
            "summary": module_summary,
        }
        self.metadata_store.set(
            file_path=rel_file_str,
            class_name=MODULE_KEY,
            interface=module_interface,
            summary=module_summary,
        )
        print("  └─ 📦 Indexed module")

        # --------------------------------------------------------------
        # Class-level analysis (if any)
        # --------------------------------------------------------------
        for class_name in class_names:
            interface = code_service.get_class_interface(class_name)
            summary = code_service.describe_class(class_name)

            node.files[file_key][class_name] = {
                "interface": interface,
                "summary": summary,
            }

            self.metadata_store.set(
                file_path=rel_file_str,
                class_name=class_name,
                interface=interface,
                summary=summary,
            )

            print(f"  └─ 🧩 Indexed class: {class_name}")

        # --------------------------------------------------------------
        # Top-level function analysis (if any)
        # --------------------------------------------------------------
        if function_names:
            node.files[file_key].setdefault(FUNCTIONS_KEY, {})
            functions_bucket: Dict[str, Dict[str, str]] = node.files[file_key][FUNCTIONS_KEY]  # type: ignore[assignment]

            for fn_name in function_names:
                fn_interface = code_service.get_function_interface(fn_name)
                fn_summary = code_service.describe_function(fn_name)

                functions_bucket[fn_name] = {
                    "interface": fn_interface,
                    "summary": fn_summary,
                }

                # Store into the flat store using a reserved name
                self.metadata_store.set(
                    file_path=rel_file_str,
                    class_name=f"{FUNCTION_STORE_PREFIX}{fn_name}",
                    interface=fn_interface,
                    summary=fn_summary,
                )

                print(f"  └─ 🧰 Indexed function: {fn_name}")

    def _extract_top_level_symbols(self, code: str) -> tuple[List[str], List[str]]:
        """
        Extract top-level class names and top-level function names from Python source.

        Notes
        -----
        - Only considers top-level definitions (direct children of module body).
        - Functions inside classes are NOT included (methods).
        - Nested functions are NOT included.

        Parameters
        ----------
        code :
            Python source code.

        Returns
        -------
        (classes, functions) :
            Two lists containing names of top-level classes and top-level functions.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            logger.warning("Failed to parse Python file: %s", exc)
            return ([], [])

        classes: List[str] = []
        functions: List[str] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        return classes, functions

    def _build_file_context(self, relative_path: str) -> str:
        """
        Build a context string passed into CodeService for a given file.

        Combines repository-level context with file-specific information.
        """
        base = self.repo_context.strip()
        if base:
            base += "\n\n"
        return f"{base}File: {relative_path}"

    def _get_or_create_folder_node(self, root: FolderNode, rel_dir: str) -> FolderNode:
        """
        Retrieve or create a FolderNode for a relative directory path like:
          ".", "pkg", "pkg/subpkg", etc.
        """
        if rel_dir in (".", ""):
            return root

        parts = Path(rel_dir).parts
        cur = root
        running: List[str] = []

        for part in parts:
            running.append(part)
            if part not in cur.subfolders:
                cur.subfolders[part] = FolderNode(path=str(Path(*running)))
            cur = cur.subfolders[part]

        return cur

    def _write_folder_jsons(self, tree: FolderNode) -> None:
        """
        Write a metadata JSON file into every folder of the workspace.

        Each folder's JSON includes:
          - `path`
          - `files` for that folder
          - `subfolders` mapping immediate subfolder name -> nested metadata
        """

        def to_dict(node: FolderNode) -> dict:
            return {
                "path": node.path,
                "files": node.files,
                "subfolders": {name: to_dict(child) for name, child in node.subfolders.items()},
            }

        def write_node(node: FolderNode) -> None:
            folder_path = self.root_dir if node.path == "." else (self.root_dir / node.path)
            out_path = folder_path / self.folder_metadata_filename

            folder_path.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(to_dict(node), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"💾 Wrote folder metadata to {out_path}")

            for child in node.subfolders.values():
                write_node(child)

        write_node(tree)
