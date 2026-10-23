"""Diagnostics + the "ask your AI" helper text (copy-to-clipboard, no network)."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from .hardware import Hardware, quiet, summary

_PREAMBLE = (
    "You are helping a beginner who is using a super-resolution model training app "
    "(HEART / NERVE Trainer). They hit a problem. Guide them step by step, in simple "
    "language, to fix it. Assume they are not a programmer. Ask for more info only if "
    "truly needed. Here is the technical context:"
)

_VERSION_CODE = """
import json
import importlib.metadata as m

out = {}
for pkg in ("torch", "torchvision", "onnxruntime", "onnx", "safetensors",
            "antialiased_cnns", "spandrel", "pyvips"):
    try:
        out[pkg] = m.version(pkg)
    except Exception:
        out[pkg] = "NOT INSTALLED"
try:
    import torch
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_build"] = torch.version.cuda
except Exception:
    pass
print(json.dumps(out))
"""


def _versions(py: str | None) -> str:
    lines = [f"Platform: {platform.platform()}"]
    if not py:
        lines.append("Training environment: NOT SET UP")
        return "\n".join(lines)
    try:
        p = subprocess.run([py, "-c", _VERSION_CODE], capture_output=True, text=True,
                           timeout=30, **quiet())
        data = json.loads([ln for ln in p.stdout.splitlines() if ln.strip()][-1])
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        if p.returncode != 0:
            lines.append(f"version check stderr: {(p.stdout + p.stderr)[-500:]}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"Training environment: could not inspect ({e})")
    return "\n".join(lines)


def system_report(hw: Hardware | None = None, python_exe: str | None = None) -> str:
    head = summary(hw) if hw is not None else "(hardware not detected)"
    return f"{head}\n{_versions(python_exe)}"


def help_prompt(failing_step: str, error: str = "", log_tail: str = "",
                hw: Hardware | None = None, python_exe: str | None = None) -> str:
    """Build the copy-pasteable help text for an AI/forum."""
    parts = [
        _PREAMBLE, "", "== System ==", system_report(hw, python_exe), "",
        "== What failed ==", failing_step.strip() or "(unknown step)",
    ]
    if error.strip():
        parts += ["", "== Error ==", error.strip()[:4000]]
    if log_tail.strip():
        parts += ["", "== Log tail ==", log_tail.strip()[-4000:]]
    parts += ["", "Please explain what's wrong and give numbered, step-by-step fix instructions."]
    return "\n".join(parts)


def save_report(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
