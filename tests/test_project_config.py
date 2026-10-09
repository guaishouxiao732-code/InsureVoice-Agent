"""Portable configuration tests: no model loading or credentials required."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from project_config import PROJECT_ROOT, cosyvoice_python, project_path


class ProjectConfigTests(unittest.TestCase):
    def test_default_relative_path_is_project_relative(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(project_path("QWEN_MODEL_PATH", "models/example"),
                             (PROJECT_ROOT / "models/example").resolve())

    def test_absolute_override(self):
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"QWEN_MODEL_PATH": directory}):
                self.assertEqual(project_path("QWEN_MODEL_PATH", "unused"), Path(directory))

    def test_relative_override_ignores_terminal_directory(self):
        with patch.dict(os.environ, {"RAG_INDEX_DIR": "custom/index"}):
            self.assertEqual(project_path("RAG_INDEX_DIR", "unused"),
                             (PROJECT_ROOT / "custom/index").resolve())

    def test_expand_user(self):
        with patch.dict(os.environ, {"RAG_INDEX_DIR": "~/insurevoice-test-index"}):
            self.assertEqual(project_path("RAG_INDEX_DIR", "unused"),
                             (Path.home() / "insurevoice-test-index").resolve())

    def test_blank_override_uses_default(self):
        with patch.dict(os.environ, {"KNOWLEDGE_SOURCE": "  "}):
            self.assertEqual(project_path("KNOWLEDGE_SOURCE", "knowledge"),
                             (PROJECT_ROOT / "knowledge").resolve())

    def test_explicit_cosyvoice_python_has_priority(self):
        with patch.dict(os.environ, {"COSYVOICE_PYTHON": "runtime/tts/bin/python"}):
            self.assertEqual(cosyvoice_python(),
                             (PROJECT_ROOT / "runtime/tts/bin/python").resolve())

    def test_discovers_sibling_conda_environment(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "envs/qwen3-vl"
            python = prefix.parent / "cosyvoice/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {}, clear=True), patch("project_config.sys.prefix", str(prefix)):
                self.assertEqual(cosyvoice_python(), python.resolve())

    def test_discovers_conda_exe_base(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            python = base / "envs/cosyvoice/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {"CONDA_EXE": str(base / "bin/conda")}, clear=True), \
                    patch("project_config.sys.prefix", str(base / "unrelated/current")):
                self.assertEqual(cosyvoice_python(), python.resolve())

    def test_discovers_custom_conda_envs_path(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            python = base / "custom/cosyvoice/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {"CONDA_ENVS_PATH": str(base / "custom")}, clear=True), \
                    patch("project_config.sys.prefix", str(base / "unrelated/current")):
                self.assertEqual(cosyvoice_python(), python.resolve())

    def test_missing_interpreter_returns_predictable_candidate(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "envs/qwen3-vl"
            with patch.dict(os.environ, {}, clear=True), patch("project_config.sys.prefix", str(prefix)):
                self.assertEqual(cosyvoice_python(),
                                 (prefix.parent / "cosyvoice/bin/python").resolve())


if __name__ == "__main__":
    unittest.main()
