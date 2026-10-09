"""Portable paths shared by CLI, web and the isolated TTS worker.

Configuration is read from environment variables at process startup. Relative
paths are anchored to this project, not the terminal's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(variable: str, default: str | Path) -> Path:
    value = os.environ.get(variable, "").strip()
    path = Path(value if value else default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def cosyvoice_python() -> Path:
    """Find the sibling Conda environment without running Conda or downloading."""
    if os.environ.get("COSYVOICE_PYTHON", "").strip():
        return project_path("COSYVOICE_PYTHON", "")
    prefix = Path(sys.prefix)
    candidates = []
    if prefix.name == "cosyvoice":
        candidates.append(Path(sys.executable))
    candidates.extend([
        prefix.parent / "cosyvoice" / "bin" / "python",
        prefix / "envs" / "cosyvoice" / "bin" / "python",
    ])
    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe:
        candidates.append(Path(conda_exe).expanduser().resolve().parent.parent
                          / "envs" / "cosyvoice" / "bin" / "python")
    for directory in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep):
        if directory.strip():
            candidates.append(Path(directory).expanduser() / "cosyvoice" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # TTSClient emits an actionable error if automatic startup is needed.
    return candidates[0].resolve()
