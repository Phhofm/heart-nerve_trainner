"""Headless CLI — lets us build/test everything without the GUI.

    python -m hnt detect
    python -m hnt config --model NERVE --scale 2 --profile FIDELITY \
        --hr /data/hr --out /tmp/run.yml
    python -m hnt selftest
    python -m hnt help-text --step train --error "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config_gen import TrainRequest, build_config, write_config
from .diagnostics import help_prompt
from .hardware import detect, summary
from .presets import PROFILES, SCALES, auto_params


def _cmd_detect(_: argparse.Namespace) -> int:
    hw = detect()
    print(summary(hw))
    return 0


def _cmd_config(a: argparse.Namespace) -> int:
    from .paths import find_repo

    repo = find_repo()
    ap = auto_params(a.model, a.scale, a.profile, a.vram)
    req = TrainRequest(
        model=a.model, scale=a.scale, profile=a.profile, data_mode=a.data_mode,
        hr_dir=a.hr, lr_dir=a.lr or "", val_hr_dir=a.val_hr or "", val_lr_dir=a.val_lr or "",
        degradation=a.degradation, batch=a.batch or ap.batch, lq_size=a.lq or ap.lq_size,
        iters=a.iters, name=a.name, val_perceptual_metrics=a.perceptual_metrics,
    )
    cfg = build_config(req, repo)
    out = write_config(cfg, Path(a.out))
    print(f"wrote {out}")
    return 0


def _cmd_selftest(_: argparse.Namespace) -> int:
    from .paths import get_paths
    from .selftest import run_selftest

    res = run_selftest(get_paths(), on_log=lambda m: print(f"  {m}", file=sys.stderr))
    print(json.dumps({"ok": res.ok, "stage": res.stage, "detail": res.detail}, indent=2))
    return 0 if res.ok else 1


def _cmd_infer(a: argparse.Namespace) -> int:
    from .infer import infer_paths

    res = infer_paths(a.model, list(a.input), a.out, arch=a.arch, scale=a.scale,
                      device=a.device, tile=a.tile)
    print(f"wrote {len(res.outputs)} file(s) to {a.out}")
    for e in res.errors:
        print("ERR", e, file=sys.stderr)
    return 0 if not res.errors else 1


def _cmd_help_text(a: argparse.Namespace) -> int:
    print(help_prompt(a.step, a.error, hw=detect()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hnt", description="HEART / NERVE Trainer (headless)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("detect").set_defaults(fn=_cmd_detect)

    c = sub.add_parser("config")
    c.add_argument("--model", default="NERVE", choices=["HEART", "NERVE"])
    c.add_argument("--scale", type=int, default=2, choices=SCALES)
    c.add_argument("--profile", default="FIDELITY", choices=list(PROFILES))
    c.add_argument("--data-mode", default="OTF", choices=["PAIRED", "OTF"])
    c.add_argument("--hr", required=True)
    c.add_argument("--lr", default="")
    c.add_argument("--val-hr", default="")
    c.add_argument("--val-lr", default="")
    c.add_argument("--degradation", default="")
    c.add_argument("--batch", type=int, default=0)
    c.add_argument("--lq", type=int, default=0)
    c.add_argument("--vram", type=float, default=8.0)
    c.add_argument("--iters", type=int, default=100_000)
    c.add_argument("--perceptual-metrics", action="store_true",
                   help="also validate with DISTS/LPIPS (slower)")
    c.add_argument("--name", default="hnt_run")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=_cmd_config)

    sub.add_parser("selftest").set_defaults(fn=_cmd_selftest)

    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True, help=".safetensors/.pth HEART or NERVE model")
    inf.add_argument("--arch", default="NERVE", choices=["HEART", "NERVE"])
    inf.add_argument("--scale", type=int, default=2, choices=SCALES)
    inf.add_argument("--input", nargs="+", required=True, help="image files and/or folders")
    inf.add_argument("--out", required=True)
    inf.add_argument("--device", default="cuda")
    inf.add_argument("--tile", type=int, default=0, help="0 = whole image; else tile size (e.g. 256)")
    inf.set_defaults(fn=_cmd_infer)

    h = sub.add_parser("help-text")
    h.add_argument("--step", default="unknown")
    h.add_argument("--error", default="")
    h.set_defaults(fn=_cmd_help_text)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
