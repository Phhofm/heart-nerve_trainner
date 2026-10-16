"""Diagnostics + the "ask your AI" helper text (copy-to-clipboard, no network)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from .hardware import Hardware, summary

_PREAMBLE = (
    "You are helping a beginner who is using a super-resolution model training app "
    "(HEART / NERVE Trainer). They hit a problem. Guide them step by step, in simple "
    "language, to fix it. Assume they are not a programmer. Ask for more info only if "
    "truly needed. Here is the technical context:"
)


def _versions() -> str:
    lines = [f"Python: {sys.version.split()[0]}", f"Platform: {platform.platform()}"]
    for mod in ("torch", "onnxruntime", "onnx", "safetensors"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"{mod}: NOT INSTALLED")
    try:
        import torch

        lines.append(f"torch CUDA available: {torch.cuda.is_available()} "
                     f"(built {getattr(torch.version, 'cuda', '?')})")
    except Exception:
        pass
    return "\n".join(lines)


def system_report(hw: Hardware | None = None) -> str:
    head = summary(hw) if hw is not None else "(hardware not detected)"
    return f"{head}\n{_versions()}"


def help_prompt(failing_step: str, error: str = "", log_tail: str = "",
                hw: Hardware | None = None) -> str:
    """Build the copy-pasteable help text for an AI/forum."""
    parts = [
        _PREAMBLE, "", "== System ==", system_report(hw), "",
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
