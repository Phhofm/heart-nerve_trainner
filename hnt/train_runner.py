"""Training runner: launch ``train.py``, stream/parse the log, sample telemetry.

BEGINNER-FIRST behaviours:
* run deprioritized (nice/ionice) so the user's machine stays usable;
* parse iter / loss / ETA / PSNR from the log into structured progress;
* **auto-repair**: if torch.compile blows up, transparently relaunch with
  ``use_compile: false``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .config_gen import write_config

_ITER_RE = re.compile(r"iter:\s*([0-9]+).*?performance:\s*([0-9.]+)\s*it/s.*?eta:\s*([0-9:\- ]+)",
                      re.IGNORECASE)
_LOSS_RE = re.compile(r"l_g_total:\s*([0-9.eE+-]+)")
# traiNNer validation lines look like:  "\t # psnr : 12.3456\tBest: ... @ 25000 iter"
_METRIC_RE = re.compile(r"#\s*([a-zA-Z_]+)\s*:\s*([0-9.]+)")
_ANSI = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

# Metrics the GUI knows how to plot (traiNNer computes these when enabled in the config).
METRIC_NAMES = ("psnr", "ssim", "dists", "lpips", "topiq")


def _clean(s: str) -> str:
    return _ANSI.sub("", s)


def parse_metrics(line: str) -> dict[str, float]:
    """Pull ``# <name> : <value>`` validation metrics out of one log line.

    Returns only known metric names, so unrelated ``# ...`` lines are ignored.
    """
    return {
        name: float(val)
        for name, val in _METRIC_RE.findall(line)
        if name.lower() in METRIC_NAMES
    }


@dataclass
class Progress:
    iter: int = 0
    iters: int = 0
    it_s: float = 0.0
    eta: str = ""
    loss: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Telemetry:
    gpu_util: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    gpu_temp: float = 0.0
    cpu: float = 0.0
    ram: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)  # (vram, gpu_util)


def read_telemetry() -> Telemetry:
    t = Telemetry()
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if out:
            u, used, total, temp = [p.strip() for p in out[0].split(",")]
            t.gpu_util, t.vram_used_gb = float(u), round(float(used) / 1024, 2)
            t.vram_total_gb, t.gpu_temp = round(float(total) / 1024, 2), float(temp)
    except Exception:
        pass
    try:
        import psutil

        t.cpu = psutil.cpu_percent(interval=None)
        t.ram = round(psutil.virtual_memory().percent, 1)
    except Exception:
        pass
    return t


class TrainRunner:
    def __init__(self, repo: Path, logs_dir: Path, nice: bool = True):
        self.repo = repo
        self.logs_dir = logs_dir
        self.nice = nice
        self.proc: subprocess.Popen | None = None

    def _cmd(self, cfg_path: Path) -> list[str]:
        py = sys.executable
        base = [py, "train.py", "-opt", str(cfg_path), "--auto_resume"]
        if self.nice and os.name != "nt":
            return ["nice", "-n", "10", "ionice", "-c", "3", *base]
        return base

    def start(self, cfg: dict, name: str) -> Path:
        cfg_path = write_config(cfg, self.repo / "options/train/ABLATION" / f"{name}.yml")
        log = self.logs_dir / f"{name}.log"
        self.proc = subprocess.Popen(
            self._cmd(cfg_path), cwd=self.repo, stdout=log.open("w"),
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        return log

    def watch(self, log: Path, total_iters: int,
              on_progress: Callable[[Progress], None] | None = None,
              poll: float = 2.0) -> Iterator[Progress]:
        """Yield Progress as the run advances (tails the log)."""
        prog = Progress(iters=total_iters)
        pos = 0
        while True:
            if log.exists():
                with log.open(errors="ignore") as f:
                    f.seek(pos)
                    for line in f:
                        line = _clean(line)
                        if m := _ITER_RE.search(line):
                            prog.iter, prog.it_s, prog.eta = int(m.group(1)), float(m.group(2)), m.group(3)
                        if m := _LOSS_RE.search(line):
                            prog.loss = float(m.group(1))
                        for name, value in parse_metrics(line).items():
                            prog.metrics[name] = value
                        prog.psnr = prog.metrics.get("psnr", prog.psnr)
                        prog.ssim = prog.metrics.get("ssim", prog.ssim)
                    pos = f.tell()
            alive = self.proc is not None and self.proc.poll() is None
            if on_progress:
                on_progress(prog)
            yield prog
            if not alive:
                break
            time.sleep(poll)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                self.proc.terminate()

    def compile_failed(self, log: Path) -> bool:
        try:
            txt = _clean(log.read_text(errors="ignore")).lower()
        except OSError:
            return False
        return any(k in txt for k in ("inductor", "torch.compile", "dynamo", "compile error"))
