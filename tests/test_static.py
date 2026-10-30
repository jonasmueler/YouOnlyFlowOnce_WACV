"""Static rules of the repository: no path hacks, no print statements, imports at the top."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = sorted(ROOT.glob("*.py")) + sorted((ROOT / "yofo").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py"))


def test_no_sys_path_or_legacy_config_imports() -> None:
    for path in PY_FILES:
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text, path
        assert not re.search(r"^\s*import config\b", text, re.M), path


def test_no_print_in_library() -> None:
    for path in (ROOT / "yofo").rglob("*.py"):
        assert not re.search(r"^\s*print\(", path.read_text(encoding="utf-8"), re.M), path


def test_imports_only_at_module_level() -> None:
    for path in PY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for inner in ast.walk(node):
                    assert not isinstance(inner, (ast.Import, ast.ImportFrom)), f"import inside {node.name} in {path}"


def test_every_function_and_class_has_a_docstring() -> None:
    for path in (ROOT / "yofo").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("__"):
                assert ast.get_docstring(node), f"{node.name} in {path} has no docstring"
