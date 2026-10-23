"""Hardware + environment detection (GPU/VRAM, CPU, RAM, versions).

Everything degrades gracefully: no pynvml / no GPU ⇒ CPU-only plan with a
clear warning (see ``Plan.cpu_fallback``).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class GPU:
    name: str
    vram_gb: float
    driver: str | None = None
    cuda_runtime: str | None = None
    compute_cap: float | None = None


@dataclass
class Hardware:
    os: str
    os_version: str
    python: str
    cpu: str
    ram_gb: float
    gpus: list[GPU] = field(default_factory=list)

    @property
    def has_cuda_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def best_vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)

    @property
    def supports_bf16(self) -> bool:
        """Ampere+ (compute 8.0+) has fast bf16; older cards should use fp16.

        If the driver does not report a capability we keep the bf16 default.
        """
        caps = [g.compute_cap for g in self.gpus if g.compute_cap]
        return True if not caps else max(caps) >= 8.0


def quiet() -> dict:
    """subprocess flags that stop console windows flashing on Windows."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _smi(exe: str, fields: str) -> str:
    return subprocess.run(
        [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True, **quiet(),
    ).stdout.strip()


def _nvidia_smi_query() -> list[GPU]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = _smi(exe, "name,memory.total,driver_version,compute_cap")
        has_cap = True
    except (subprocess.SubprocessError, OSError):
        try:
            out = _smi(exe, "name,memory.total,driver_version")  # older driver
            has_cap = False
        except (subprocess.SubprocessError, OSError):
            return []
    gpus: list[GPU] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                vram = float(parts[1]) / 1024.0  # MiB -> GiB
            except ValueError:
                vram = 0.0
            cap = None
            if has_cap and len(parts) >= 4:
                try:
                    cap = float(parts[3])
                except ValueError:
                    cap = None
            gpus.append(GPU(name=parts[0], vram_gb=round(vram, 1), driver=parts[2],
                            compute_cap=cap))
    return gpus


def _ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return 0.0


def detect() -> Hardware:
    return Hardware(
        os=platform.system(),
        os_version=platform.release(),
        python=sys.version.split()[0],
        cpu=platform.processor() or platform.machine(),
        ram_gb=_ram_gb(),
        gpus=_nvidia_smi_query(),
    )


def summary(hw: Hardware) -> str:
    gpu = ", ".join(f"{g.name} ({g.vram_gb} GB, driver {g.driver})" for g in hw.gpus) or "none"
    return (
        f"OS: {hw.os} {hw.os_version}\n"
        f"CPU: {hw.cpu}\nRAM: {hw.ram_gb} GB\n"
        f"GPU: {gpu}\nPython: {hw.python}"
    )
