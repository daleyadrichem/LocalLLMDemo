from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ClassMetadata:
    """
    Metadata about a single class in the codebase.

    Parameters
    ----------
    interface :
        A string representation of the class interface (e.g. header, public
        methods, docstrings). Typically produced by CodeService.get_class_interface.
    summary :
        A natural-language summary / description of the class. Typically
        produced by CodeService.describe_class.
    """

    interface: str
    summary: str


# Type alias for the internal index structure:
# index[file_path][class_name] = ClassMetadata
ClassIndex = Dict[str, Dict[str, ClassMetadata]]


@dataclass
class ClassMetadataStore:
    """
    JSON-backed store for class-level metadata across a workspace.

    The stored JSON has the following structure::

        {
          "<file_path>": {
            "<class_name>": {
              "interface": "...",
              "summary": "..."
            }
          }
        }

    Notes
    -----
    Paths are stored as strings (typically relative to the workspace root).
    """

    json_path: Path
    _index: ClassIndex = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """
        Load existing data from disk if the JSON file exists.
        """
        if self.json_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the metadata index from the JSON file.

        Raises
        ------
        json.JSONDecodeError
            If the JSON file exists but contains invalid JSON.
        """
        raw = json.loads(self.json_path.read_text(encoding="utf-8"))
        index: ClassIndex = {}

        for file_path, classes in raw.items():
            index[file_path] = {}
            for class_name, data in classes.items():
                index[file_path][class_name] = ClassMetadata(
                    interface=data.get("interface", ""),
                    summary=data.get("summary", ""),
                )

        self._index = index

    def save(self) -> None:
        """
        Persist the metadata index to the JSON file.

        The saved structure is a dict of dicts, where each innermost value
        is a JSON-serializable dict with 'interface' and 'summary' keys.
        """
        serializable: Dict[str, Dict[str, Dict[str, str]]] = {}

        for file_path, classes in self._index.items():
            serializable[file_path] = {}
            for class_name, meta in classes.items():
                serializable[file_path][class_name] = {
                    "interface": meta.interface,
                    "summary": meta.summary,
                }

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get(
        self,
        file_path: str,
        class_name: str,
    ) -> Optional[ClassMetadata]:
        """
        Retrieve metadata for a given class in a given file.

        Parameters
        ----------
        file_path :
            Path of the file as stored in the index (typically relative to
            the workspace root).
        class_name :
            Name of the class.

        Returns
        -------
        ClassMetadata or None
            The metadata if present, None otherwise.
        """
        return self._index.get(file_path, {}).get(class_name)

    def set(
        self,
        file_path: str,
        class_name: str,
        interface: str,
        summary: str,
    ) -> None:
        """
        Set or update the metadata for a given class in a given file.

        Parameters
        ----------
        file_path :
            Path of the file, typically relative to workspace root.
        class_name :
            Name of the class.
        interface :
            Interface representation for the class.
        summary :
            Natural-language summary for the class.
        """
        if file_path not in self._index:
            self._index[file_path] = {}

        self._index[file_path][class_name] = ClassMetadata(
            interface=interface,
            summary=summary,
        )

    def all(self) -> ClassIndex:
        """
        Return a shallow copy of the entire index.
        """
        # Shallow copy is enough for read-only use
        return dict(self._index)
