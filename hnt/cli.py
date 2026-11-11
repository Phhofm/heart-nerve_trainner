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
from .video import infer_video


def _cmd_detect(_: argparse.Namespace) -> int:
    hw = detect()
    print(summary(hw))
    return 0


def _cmd_config(a: argparse.Namespace) -> int:
    from .hardware import detect
    from .paths import find_repo

    repo = find_repo()
    ap = auto_params(a.model, a.scale, a.profile, a.vram)
    req = TrainRequest(
        model=a.model, scale=a.scale, profile=a.profile, data_mode=a.data_mode,
        hr_dir=a.hr, lr_dir=a.lr or "", val_hr_dir=a.val_hr or "", val_lr_dir=a.val_lr or "",
        degradation=a.degradation, batch=a.batch or ap.batch, lq_size=a.lq or ap.lq_size,
        iters=a.iters, name=a.name, val_perceptual_metrics=a.perceptual_metrics,
        amp_bf16=detect().supports_bf16, val_tile_size=a.val_tile,
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
                      device=a.device, tile=a.tile, skip_existing=a.skip_existing,
                      on_progress=lambda done, total, path: print(
                          f"progress {done}/{total} {path}", flush=True))
    for p in res.outputs:
        print(f"output {p}")
    print(f"wrote {len(res.outputs)} file(s) to {a.out}")
    for e in res.errors:
        print("ERR", e, file=sys.stderr)
    return 0 if not res.errors else 1


def _cmd_video(a: argparse.Namespace) -> int:
    res = infer_video(
        a.model,
        a.input,
        a.out,
        arch=a.arch,
        scale=a.scale,
        device=a.device,
        tile=a.tile,
        codec=a.codec,
        crf=a.crf,
        overwrite=a.overwrite,
        deinterlace=a.deinterlace,
        on_progress=lambda done, total: print(
            f"frame {done}/{total if total is not None else '?'}", flush=True
        ),
    )
    print(f"output {res.output}")
    print(f"frames {res.frames}")
    return 0


def _cmd_check_dataset(a: argparse.Namespace) -> int:
    from .dataset_check import check_dataset

    rep = check_dataset(a.hr, a.lr or "", a.scale)
    if rep.ok:
        print(f"ok — HR: {rep.hr_count} image(s)"
              + (f", LR: {rep.lq_count}" if rep.lq_count >= 0 else ""))
        return 0
    print(rep.error_text(), file=sys.stderr)
    return 1


def _cmd_export(a: argparse.Namespace) -> int:
    from .exporter import export_checkpoint
    from .paths import find_repo
    from .runs import find_run, parse_run_name, pick_checkpoint

    repo = Path(a.repo) if a.repo else find_repo()
    run = find_run(repo, a.run)
    if run is None:
        print(f"no run named {a.run} under {repo / 'experiments'}", file=sys.stderr)
        return 1
    ck = pick_checkpoint(run, a.iter)
    if ck is None:
        which = f" @ iter {a.iter}" if a.iter else ""
        print(f"no checkpoint{which} in run {a.run}", file=sys.stderr)
        return 1
    ms = parse_run_name(a.run)
    model, scale = ms if ms else (a.model, a.scale)
    res = export_checkpoint(repo, ck.path, Path(a.out), model, scale,
                            fp32=a.fp32, fp16=a.fp16, ncnn=a.ncnn,
                            python_exe=a.python or None)
    for f in (res.safetensors, res.onnx_fp32, res.onnx_fp16, res.ncnn_dir):
        if f:
            print(f"wrote {f}")
    for e in res.errors:
        print("ERR", e, file=sys.stderr)
    return 0 if not res.errors else 1


def _cmd_catalog(_: argparse.Namespace) -> int:
    from .catalog import load_catalog

    print(json.dumps([e.__dict__ for e in load_catalog()], indent=2))
    return 0


def _cmd_help_text(a: argparse.Namespace) -> int:
    print(help_prompt(a.step, a.error, hw=detect()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hnt", description="BODY Studio (headless)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("detect").set_defaults(fn=_cmd_detect)

    c = sub.add_parser("config")
    c.add_argument("--model", default="NERVE", choices=["nerve", "heart", "pulse", "pulse_fast"])
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
    c.add_argument("--val-tile", type=int, default=256,
                   help="tile size for validation (0 = whole image)")
    c.add_argument("--name", default="hnt_run")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=_cmd_config)

    sub.add_parser("selftest").set_defaults(fn=_cmd_selftest)

    cd = sub.add_parser("check-dataset", help="validate HR/LR folders before training")
    cd.add_argument("--hr", required=True)
    cd.add_argument("--lr", default="", help="empty = OTF mode (LR generated on the fly)")
    cd.add_argument("--scale", type=int, default=4, choices=SCALES)
    cd.set_defaults(fn=_cmd_check_dataset)

    sub.add_parser("catalog").set_defaults(fn=_cmd_catalog)

    ex = sub.add_parser("export", help="export a training run's checkpoint")
    ex.add_argument("--run", required=True, help="experiment name under experiments/")
    ex.add_argument("--out", required=True)
    ex.add_argument("--iter", type=int, default=None,
                    help="checkpoint iteration (default: latest)")
    ex.add_argument("--model", default="NERVE", choices=["HEART", "NERVE", "PULSE", "PULSE_fast"],
                    help="only used when the run name doesn't encode the model")
    ex.add_argument("--scale", type=int, default=2, choices=SCALES,
                    help="only used when the run name doesn't encode the scale")
    ex.add_argument("--no-fp32", dest="fp32", action="store_false")
    ex.add_argument("--no-fp16", dest="fp16", action="store_false")
    ex.add_argument("--ncnn", action="store_true", help="NERVE only")
    ex.add_argument("--python", default="", help="python executable for export subprocesses")
    ex.add_argument("--repo", default="", help="traiNNer-redux checkout (default: auto-detect)")
    ex.set_defaults(fn=_cmd_export)

    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True, help=".safetensors/.pth HEART or NERVE model")
    inf.add_argument("--arch", default="NERVE", choices=["nerve", "heart", "pulse", "pulse_fast"])
    inf.add_argument("--scale", type=int, default=2, choices=SCALES)
    inf.add_argument("--input", nargs="+", required=True, help="image files and/or folders")
    inf.add_argument("--out", required=True)
    inf.add_argument("--device", default="cuda")
    inf.add_argument("--tile", type=int, default=0, help="0 = auto by image size; -1 = whole image; else tile px")
    inf.add_argument("--skip-existing", action="store_true")
    inf.set_defaults(fn=_cmd_infer)

    h = sub.add_parser("help-text")
    h.add_argument("--step", default="unknown")
    h.add_argument("--error", default="")
    h.set_defaults(fn=_cmd_help_text)

    video = sub.add_parser("video", help="enhance a video with a BODY model")
    video.add_argument("--model", required=True, help=".safetensors/.pth BODY model")
    video.add_argument("--arch", default="NERVE", choices=["nerve", "heart", "pulse", "pulse_fast"])
    video.add_argument("--scale", type=int, default=2, choices=SCALES)
    video.add_argument("--input", required=True)
    video.add_argument("--out", required=True)
    video.add_argument("--device", default="cuda")
    video.add_argument("--tile", type=int, default=0, help="0 = auto by frame size; -1 = whole frame; else tile px")
    video.add_argument("--codec", default="libx264")
    video.add_argument("--crf", type=int, default=18)
    video.add_argument("--deinterlace", action="store_true")
    video.add_argument("--overwrite", action="store_true")
    video.set_defaults(fn=_cmd_video)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
