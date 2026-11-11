from pathlib import Path

import numpy as np

from hnt.infer import _destination_for, infer_paths


def test_destination_for_preserves_folder_structure(tmp_path: Path):
    root = tmp_path / "root"
    src = root / "a" / "b" / "img.png"

    dst = _destination_for([root], src, tmp_path / "out", "model")

    assert dst == tmp_path / "out" / "a" / "b" / "img_model.png"


def test_destination_for_uses_filename_outside_roots(tmp_path: Path):
    src = tmp_path / "img.png"

    dst = _destination_for([tmp_path / "elsewhere"], src, tmp_path / "out", "model")

    assert dst == tmp_path / "out" / "img_model.png"


def test_destination_for_uses_filename_for_explicit_file_input(tmp_path: Path):
    src = tmp_path / "img.png"

    dst = _destination_for([src], src, tmp_path / "out", "model")

    assert dst == tmp_path / "out" / "img_model.png"


def test_infer_paths_can_skip_existing(tmp_path: Path, monkeypatch):
    src = tmp_path / "img.png"
    src.write_bytes(b"stub")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "img_model.png"
    existing.write_bytes(b"old")

    monkeypatch.setattr("hnt.infer.load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "hnt.infer._load_image",
        lambda path: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "hnt.infer._run_full",
        lambda *args, **kwargs: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    result = infer_paths(
        "model.safetensors",
        [src],
        out_dir,
        suffix="model",
        skip_existing=True,
    )

    assert result.outputs == []
    assert existing.read_bytes() == b"old"


def _stub_model(monkeypatch, calls):
    monkeypatch.setattr("hnt.infer.load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        "hnt.infer._load_image",
        lambda path: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    def fake_run_full(*a, **k):
        calls.append(1)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("hnt.infer._run_full", fake_run_full)


def test_infer_paths_reports_per_file_progress(tmp_path: Path, monkeypatch):
    calls = []
    _stub_model(monkeypatch, calls)
    imgs = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"stub")
        imgs.append(p)

    seen = []
    infer_paths("m.safetensors", imgs, tmp_path / "out", suffix="model",
                on_progress=lambda done, total, path: seen.append((done, total, path.name)))

    assert calls  # model ran
    assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)]
    assert seen[0][2] == "img0.png" and seen[1][2] == "img1.png"


def test_skip_existing_avoids_model_and_still_reports(tmp_path: Path, monkeypatch):
    calls = []
    _stub_model(monkeypatch, calls)
    src = tmp_path / "img.png"
    src.write_bytes(b"stub")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "img_model.png").write_bytes(b"old")

    seen = []
    infer_paths("m.safetensors", [src], out_dir, suffix="model",
                skip_existing=True,
                on_progress=lambda done, total, path: seen.append((done, total, path.name)))

    assert calls == []  # existing output: model never ran
    assert seen == [(1, 1, "img.png")]


def test_cli_infer_prints_progress_lines(tmp_path: Path, monkeypatch, capsys):
    from types import SimpleNamespace

    from hnt import cli

    def fake_infer_paths(model, inputs, out, on_progress=None, **kw):
        for i, (d, t) in enumerate([(1, 2), (2, 2)], 1):
            if on_progress:
                on_progress(d, t, tmp_path / f"img{i}.png")
        return SimpleNamespace(outputs=[tmp_path / "out" / "a.png",
                                        tmp_path / "out" / "b.png"], errors=[])

    monkeypatch.setattr("hnt.infer.infer_paths", fake_infer_paths)
    rc = cli.main(["infer", "--model", "m.safetensors", "--input", str(tmp_path),
                   "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "progress 1/2" in out and "progress 2/2" in out
    assert "wrote 2 file(s)" in out


def test_match_input_for_output(tmp_path: Path):
    from hnt.infer import match_input_for_output

    root = tmp_path / "in"
    (root / "sub").mkdir(parents=True)
    a = root / "photo1.png"
    b = root / "sub" / "photo2.png"
    c = root / "unrelated.png"
    for p in (a, b, c):
        p.write_bytes(b"stub")

    out1 = tmp_path / "out" / "photo1_x4.png"
    out2 = tmp_path / "out" / "photo2_model.png"

    assert match_input_for_output(out1, [root]) == a
    assert match_input_for_output(out2, [root]) == b
    assert match_input_for_output(tmp_path / "out" / "ghost_x4.png", [root]) is None
    assert match_input_for_output(out1, [a]) == a  # direct file inputs are checked too
