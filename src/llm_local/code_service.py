from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import List, Optional

from llm_local.llm_client import LocalLLM


@dataclass
class CodeService:
    """
    Service for code-related operations backed by a LocalLLM instance.

    This class does NOT manage the LLM runtime itself; it assumes an already
    configured LocalLLM instance and focuses purely on code-centric tasks.

    Parameters
    ----------
    code :
        The primary code content (e.g. one file) as a string.
    context :
        Additional context from the wider repository (other files, README
        snippets, architecture notes, etc.) as a string. This can be empty
        if no extra context is needed.
    llm :
        An initialized LocalLLM client used to perform LLM-powered tasks.
    language :
        Optional hint about the programming language, used for static
        analysis. Currently, "python" enables AST-based interface extraction.
    file_path :
        Optional path of the file within the repository. Used only for diff
        headers so that the patch can be applied with tools like `git apply`.
    """

    code: str
    context: str
    llm: LocalLLM
    language: str = "python"
    file_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Small helper for diff vs full-output instructions
    # ------------------------------------------------------------------

    def _build_output_instruction(self, return_diff: bool) -> str:
        """
        Build an instruction describing whether the LLM should return
        a full updated file or a diff.

        For diff output we ask for a unified diff that can be applied with
        `git apply`. The file name in the diff is derived from `file_path`
        if available, otherwise a generic name is used.
        """
        if not return_diff:
            return (
                "Return the FULL updated code file as plain text. "
                "Do NOT include any explanation outside of comments or docstrings.\n"
            )

        file_label = self.file_path or "code.py"
        return textwrap.dedent(
            f"""
            Return ONLY a unified diff (patch) that can be applied with `git apply`
            against the original code shown above.

            Requirements for the diff:
            - Use standard unified diff format.
            - Include `---` and `+++` headers using '{file_label}' as the file path.
            - Make sure the diff can be applied cleanly to the original content.
            - Do NOT include any prose explanation before or after the diff.
            """
        )

    # ------------------------------------------------------------------
    # High-level LLM operations
    # ------------------------------------------------------------------

    def write_unit_tests(
        self,
        class_name: Optional[str] = None,
        testing_framework: str = "pytest",
        include_comments: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        target_description = (
            f"the class '{class_name}'"
            if class_name
            else "the main public classes or functions in the file"
        )

        comments_instruction = (
            "Include clear comments that explain each test's purpose.\n"
            if include_comments
            else "Focus on concise tests with minimal comments.\n"
        )

        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer and test writer.

            You are given the code of a file and some additional repository context.
            Your task is to write unit tests for {target_description}.

            Use the testing framework: {testing_framework}.

            {comments_instruction}
            Follow these rules:
            - Return ONLY valid {testing_framework} test code as plain text.
            - Do NOT include explanation outside of comments in the code.
            - Make sure tests are realistic and cover both typical and edge cases.

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def describe_class(
        self,
        class_name: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given a code file and some additional repository context.
            Describe the class '{class_name}' in clear, concise language.

            Focus on:
            - What the class does.
            - Its key responsibilities.
            - How it likely fits into the bigger system.
            - Any important design decisions or patterns that stand out.

            Keep the description suitable for a developer reading documentation.

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def add_functionality_to_class(
        self,
        class_name: str,
        description: str,
        max_tokens: Optional[int] = None,
        return_diff: bool = False
    ) -> str:
        """
        Add functionality to a class based on a natural language description.

        If return_diff is False (default), the full updated file is returned.
        If return_diff is True, a unified diff patch is returned.
        """
        output_instruction = self._build_output_instruction(return_diff)

        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given the code of a file and some additional repository context.
            Your task is to add new functionality to the class '{class_name}'.

            New functionality description:
            --------------------
            {description}
            --------------------

            Requirements:
            - Modify ONLY the relevant parts of the code needed to support
              the new functionality.
            - Preserve existing behavior unless the description explicitly
              states otherwise.
            - Follow the existing coding style and conventions (naming,
              formatting, docstrings, etc.).

            {output_instruction}

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Original code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    def refactor_class(
        self,
        class_name: str,
        description: str,
        max_tokens: Optional[int] = None,
        return_diff: bool = False
    ) -> str:
        """
        Refactor a class based on a high-level description.

        If return_diff is False (default), the full updated file is returned.
        If return_diff is True, a unified diff patch is returned.
        """
        output_instruction = self._build_output_instruction(return_diff)

        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given the code of a file and some additional repository context.
            Your task is to refactor the class '{class_name}'.

            Refactor goals:
            --------------------
            {description}
            --------------------

            Requirements:
            - Preserve the public behavior and interface of the class unless the
              description explicitly allows changes.
            - Improve readability and maintainability.
            - Keep or improve type hints and docstrings where appropriate.
            - Follow the existing coding style and conventions.

            {output_instruction}

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Original code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Docstrings, improvements, method explanation, usage examples
    # ------------------------------------------------------------------

    def generate_docstrings(
        self,
        class_name: str,
        max_tokens: Optional[int] = None,
        return_diff: bool = False,
    ) -> str:
        """
        Add or improve docstrings for all methods in the class.

        If return_diff is False (default), the full updated file is returned.
        If return_diff is True, a unified diff patch is returned.
        """
        output_instruction = self._build_output_instruction(return_diff)

        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given the code of a file and some additional repository context.
            Your task is to add or improve docstrings for the class '{class_name}'.

            Requirements:
            - Ensure the class itself and all its public methods have clear,
              informative docstrings.
            - Follow the existing docstring style if one is already present
              (e.g. Google, NumPy, or reStructuredText style).
            - Preserve existing behavior and signatures.

            {output_instruction}

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Original code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    def suggest_improvements(
        self,
        class_name: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Ask the LLM for a review and improvement suggestions for a class.

        Returns a textual report with suggestions (not modified code).
        """
        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer performing a code review.

            You are given the code of a file and some additional repository context.
            Provide a list of concrete improvement suggestions for the class
            '{class_name}'.

            Focus on:
            - Readability and maintainability.
            - Naming and structure.
            - Testability and separation of concerns.
            - Error handling and edge cases.
            - Type hints and documentation.

            Format your answer as bullet points grouped by theme. Do NOT return
            modified code, only suggestions.

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    def explain_method(
        self,
        class_name: str,
        method_name: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Ask the LLM to explain a single method inside a class.

        Returns a human-readable explanation focused on what the method does,
        how it works, and any important edge cases.
        """
        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given the code of a file and some additional repository context.
            Explain the method '{method_name}' inside the class '{class_name}'.

            Focus on:
            - What the method does and when it should be used.
            - Its parameters and return value.
            - Important branches or edge cases.
            - Any side effects or interactions with other components.

            Keep the explanation suitable for a developer reading documentation.

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    def generate_usage_examples(
        self,
        class_name: str,
        context: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate usage examples for a class, using additional context to steer
        the examples toward realistic scenarios.

        Returns example code snippets as a string.
        """
        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer.

            You are given the code of a file, some repository context, and some
            additional usage context.

            Your task is to generate realistic usage examples for the class
            '{class_name}'.

            Usage context (what the caller is trying to do, constraints, etc.):
            --------------------
            {context}
            --------------------

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Code file:
            --------------------
            {self.code}
            --------------------

            Requirements:
            - Provide one or more concise code examples showing how to instantiate
              and use the class '{class_name}'.
            - Use realistic data and function calls.
            - If relevant, show how this class interacts with other components
              hinted at in the context.
            - Return ONLY code examples as plain text (with comments allowed),
              no prose explanation around them.
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    def review_diff_for_merge_request(
        self,
        class_name: str,
        diff: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Perform a code-review-style analysis of a diff for the given class and
        suggest what to write in a merge request description and review comments.

        The 'code' stored on this service is considered the "before" version,
        and the diff describes the proposed changes.

        Parameters
        ----------
        class_name :
            Name of the class being changed (or primarily affected) by the diff.
        diff :
            A unified diff (git-style) or similar patch text showing the changes.
        max_tokens :
            Optional maximum number of tokens for the LLM response.

        Returns
        -------
        str
            A textual summary including:
            - Suggested merge request title and description.
            - Key changes and rationale (as inferred from the diff).
            - Potential risks or concerns.
            - Suggested inline review comments (e.g. what a reviewer might say).
        """
        prompt = textwrap.dedent(
            f"""
            You are an expert software engineer performing a code review.

            You are given:
            - The previous version of a code file (as stored in the repository).
            - A diff describing the proposed changes to that file.
            - Some additional repository context.

            The main focus is on the class '{class_name}'.

            Previous code file (before changes):
            --------------------
            {self.code}
            --------------------

            Diff (proposed changes):
            --------------------
            {diff}
            --------------------

            Repository context (may be partial, use only if helpful):
            --------------------
            {self.context}
            --------------------

            Your task:
            - Act as a reviewer doing a code review for a merge request / pull request.
            - Identify what has changed and why it might have been changed.
            - Call out potential issues, edge cases, or design concerns.
            - Highlight improvements or positive aspects where relevant.

            Output format:
            1. A suggested merge request title (one line).
            2. A short merge request description (2–5 bullet points).
            3. A section "Review comments" with bullet points of concrete comments
               a reviewer might leave (both positive and critical).

            Do NOT output any modified code, only the review content.
            """
        ).strip()

        return self.llm.generate(prompt=prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Static analysis: class interface (no LLM if possible)
    # ------------------------------------------------------------------

    def get_class_interface(self, class_name: str) -> str:
        """
        Extract the interface (class header, public methods, docstrings)
        for the given class.

        For Python code, this uses the standard library `ast` module and
        does NOT call the LLM. For other languages, this method falls
        back to an LLM-based extraction.
        """
        if self.language.lower() == "python":
            return self._get_python_class_interface(class_name)

        prompt = textwrap.dedent(
            f"""
            Extract the public interface for the class '{class_name}' from the code below.

            Requirements:
            - Include the class declaration line (with bases if present).
            - Include the class docstring if present.
            - Include all public method signatures (methods that are part
              of the public API), with their docstrings if present.
            - Do NOT include method bodies; replace with '...' or an empty body.
            - Return valid code in the same language as the original.

            Code file:
            --------------------
            {self.code}
            --------------------
            """
        ).strip()

        return self.llm.generate(prompt=prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_python_class_interface(self, class_name: str) -> str:
        try:
            tree = ast.parse(self.code)
        except SyntaxError as exc:
            raise ValueError(f"Failed to parse Python code: {exc}") from exc

        class_node: Optional[ast.ClassDef] = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            raise ValueError(f"Class '{class_name}' not found in code.")

        lines: List[str] = []
        class_header = self._extract_node_header(class_node) or f"class {class_name}:"
        lines.append(class_header)

        class_doc = ast.get_docstring(class_node, clean=False)
        if class_doc:
            class_doc_indented = textwrap.indent(
                f'"""%s"""' % class_doc, " " * (class_node.col_offset + 4)
            )
            lines.append(class_doc_indented)

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name not in ("__init__",):
                    continue

                method_header = self._extract_node_header(node)
                if not method_header:
                    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    method_header = (
                        f"{' ' * node.col_offset}{async_prefix}def {node.name}(...):"
                    )

                lines.append(method_header)

                method_doc = ast.get_docstring(node, clean=False)
                body_indent = " " * (node.col_offset + 4)

                if method_doc:
                    doc_block = textwrap.indent(
                        f'"""%s"""' % method_doc,
                        body_indent,
                    )
                    lines.append(doc_block)

                lines.append(f"{body_indent}...")

        return "\n".join(lines)

    def _extract_node_header(self, node: ast.AST) -> Optional[str]:
        try:
            source_segment = ast.get_source_segment(self.code, node)
        except TypeError:
            source_segment = None

        if not source_segment:
            return None

        first_line = source_segment.splitlines()[0].rstrip()
        return first_line
