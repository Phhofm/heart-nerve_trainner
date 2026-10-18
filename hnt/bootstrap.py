"""First-run bootstrap: create an isolated venv and install pinned deps via `uv`.

The user's system Python is never touched. Internet is expected for setup.
In dev (running inside the repo's own venv) callers can pass ``use_current_env``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import Paths

# Runtime deps for the app + training. torch wheel chosen by CUDA availability.
COMMON_DEPS = [
    "torchvision",
    "onnx>=1.16",
    "onnxsim>=0.4",
    "onnxruntime>=1.17",
    "onnxconverter-common>=1.14",
    "pnnx",
    "ncnn",
    "safetensors>=0.4",
    "huggingface_hub>=0.20",
    "PySide6>=6.6",
    "psutil>=5.9",
    "pyyaml>=6",
]
# Allow a pinned torch index for CUDA builds; falls back to default PyPI.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"


@dataclass
class Deps:
    full_set: list[str]
    torch_spec: str


def _has_cuda() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def required_deps() -> Deps:
    torch_spec = "torch" if _has_cuda() else "torch>=2"
    return Deps(full_set=[torch_spec, *COMMON_DEPS], torch_spec=torch_spec)


def uv_path() -> str | None:
    return shutil.which("uv") or os.environ.get("HNT_UV")


def is_setup_complete(paths: Paths) -> bool:
    return (paths.venv / ".hnt-setup-complete").is_file()


def python_exe(paths: Paths, use_current_env: bool = False) -> str:
    """The interpreter to run training/export with."""
    if use_current_env:
        return sys.executable
    for cand in ("bin/python", "Scripts/python.exe"):
        p = paths.venv / cand
        if p.is_file():
            return str(p)
    return sys.executable


def bootstrap(paths: Paths, on_log=lambda m: None, cuda: bool | None = None) -> None:
    """Create the venv and install deps. Idempotent."""
    uv = uv_path()
    if uv is None:
        raise RuntimeError(
            "`uv` not found. Install it (https://astral.sh/uv) or bundle it with the "
            "installer; set HNT_UV to its path."
        )
    on_log(f"Creating isolated environment at {paths.venv} …")
    subprocess.run([uv, "venv", str(paths.venv), "--python", "3.12"], check=True)

    deps = required_deps()
    on_log("Installing packages (torch, onnxruntime, UI, …) …")
    cmd = [uv, "pip", "install", "--python", python_exe(paths), *deps.full_set]
    if _has_cuda() if cuda is None else cuda:
        cmd += ["--index-url", TORCH_CUDA_INDEX, "--extra-index-url", "https://pypi.org/simple"]
    subprocess.run(cmd, check=True)

    on_log("Verifying installation …")
    rc = subprocess.run([python_exe(paths), "-c", "import torch, onnxruntime, PySide6"],
                        capture_output=True).returncode
    if rc != 0:
        raise RuntimeError("Dependency verification failed (imports).")
    (paths.venv / ".hnt-setup-complete").write_text("ok")
    on_log("Setup complete.")
