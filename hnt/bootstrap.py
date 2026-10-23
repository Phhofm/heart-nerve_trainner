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
# CUDA wheel index. cu128 covers Blackwell (RTX 50xx) and is backward compatible.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


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


def trainer_deps(repo: Path) -> list[str]:
    """The trainer's own declared dependencies (source of truth for training)."""
    try:
        import tomllib

        data = tomllib.loads((repo / "pyproject.toml").read_text())
        return list(data.get("project", {}).get("dependencies", []))
    except Exception:  # noqa: BLE001
        return []


def uv_path() -> str | None:
    """Find uv: $HNT_UV → bundled next to the executable → PATH."""
    env = os.environ.get("HNT_UV")
    if env and Path(env).is_file():
        return env
    exe_dir = Path(sys.executable).resolve().parent
    for cand in (exe_dir / "uv", exe_dir / "uv.exe"):
        if cand.is_file():
            return str(cand)
    return shutil.which("uv")


def is_setup_complete(paths: Paths) -> bool:
    return (paths.venv / ".hnt-setup-complete").is_file()


def python_exe(paths: Paths, use_current_env: bool = False) -> str:
    """The interpreter to run training/export with.

    Returns "" when the isolated venv is not set up yet. It must NEVER return the
    frozen app binary as a fallback: launching `app -c "import torch"` would start
    another copy of the GUI (a launch loop on first run).
    """
    if use_current_env:
        return sys.executable
    for cand in ("bin/python", "Scripts/python.exe"):
        p = paths.venv / cand
        if p.is_file():
            return str(p)
    if is_frozen():
        return ""
    return sys.executable


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def runtime_python(paths: Paths, use_current_env: bool | None = None) -> str:
    """Interpreter for training/export/inference.

    In a frozen build the app binary is NOT a Python interpreter, so heavy work must
    run with the isolated venv Python. In dev, the current env is already correct.
    """
    if use_current_env is None:
        use_current_env = not is_frozen()
    return python_exe(paths, use_current_env=use_current_env)


def app_source_dir() -> Path:
    """Directory containing the ``hnt`` package (for ``python -m hnt`` subprocesses)."""
    if is_frozen():
        base = Path(sys.executable).resolve().parent
        if (base / "hnt").is_dir():
            return base
        return Path(getattr(sys, "_MEIPASS", base))
    return Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def _system_pythons() -> list[str]:
    """Interpreters we can create the venv from without downloading one."""
    names = ("python3.12", "python3.11", "python3.10", "python3", "python", "py")
    if os.name == "nt":  # Windows: versioned exes have no '3' suffix
        names = ("python.exe", "py", "python3.12", "python3.11", "python3")
    found: list[str] = []
    for name in names:
        p = shutil.which(name)
        if p and p not in found:
            found.append(p)
    return found


def _venv_python(paths: Paths) -> Path | None:
    for cand in ("bin/python", "Scripts/python.exe"):
        p = paths.venv / cand
        if p.is_file():
            return p
    return None


def create_venv(uv: str, paths: Paths) -> None:
    """Create the isolated venv, falling back to a system interpreter.

    `uv venv --python 3.12` may try to download a managed Python; on offline or
    restricted machines that fails (exit 2). Try the system Pythons too, then
    `python -m venv` directly.
    """
    paths.venv.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[str] = []
    specs = ["3.12", *_system_pythons()]
    for spec in specs:
        shutil.rmtree(paths.venv, ignore_errors=True)
        cmd = [uv, "venv", str(paths.venv), "--python", spec]
        rc, out = _run(cmd)
        if rc == 0 and _venv_python(paths):
            return
        attempts.append(f"$ {' '.join(cmd)}\n{out[-1200:]}")

    for py in _system_pythons():
        shutil.rmtree(paths.venv, ignore_errors=True)
        cmd = [py, "-m", "venv", str(paths.venv)]
        rc, out = _run(cmd)
        if rc == 0 and _venv_python(paths):
            return
        attempts.append(f"$ {' '.join(cmd)}\n{out[-1200:]}")

    raise RuntimeError(
        "Could not create the training environment.\n\n"
        + "\n\n".join(attempts)
        + "\n\nInstall Python 3.10+ (python3) and try again, or report this message."
    )


def bootstrap(paths: Paths, on_log=lambda m: None, cuda: bool | None = None) -> None:
    """Create the venv and install deps. Idempotent."""
    uv = uv_path()
    if uv is None:
        raise RuntimeError(
            "`uv` not found. Install it (https://astral.sh/uv) or bundle it with the "
            "installer; set HNT_UV to its path."
        )
    on_log(f"Creating isolated environment at {paths.venv} …")
    create_venv(uv, paths)

    deps = required_deps()
    # Install the trainer's own dependency list too — it declares antialiased_cnns,
    # einops, ema-pytorch, pyvips, spandrel, timm, etc. Missing any of these makes
    # train.py abort at its dependency check.
    all_deps = list(dict.fromkeys([*deps.full_set, *trainer_deps(paths.repo)]))
    on_log("Installing packages (torch, trainer deps, UI, …) …")
    cmd = [uv, "pip", "install", "--python", python_exe(paths), *all_deps]
    if _has_cuda() if cuda is None else cuda:
        cmd += ["--index-url", TORCH_CUDA_INDEX, "--extra-index-url", "https://pypi.org/simple"]
    rc, out = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"Dependency install failed:\n\n{out[-2000:]}")

    on_log("Verifying installation …")
    p = subprocess.run(
        [python_exe(paths), "-c",
         "from traiNNer.check.check_dependencies import check_dependencies; "
         "check_dependencies()"],
        cwd=str(paths.repo), capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Dependency verification failed:\n\n" + (p.stdout + p.stderr)[-1500:]
        )
    (paths.venv / ".hnt-setup-complete").write_text("ok")
    on_log("Setup complete.")
