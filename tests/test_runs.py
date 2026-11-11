"""Run discovery — list training runs, checkpoints, and resumable state."""

from __future__ import annotations

from pathlib import Path

from hnt.runs import find_run, list_runs


def _mk_run(repo: Path, name: str, *, ema_iters=(), plain_iters=(), states=(),
            config=True) -> None:
    models = repo / "experiments" / name / "models"
    models.mkdir(parents=True)
    for i in ema_iters:
        (models / f"net_g_ema_{i}.safetensors").write_bytes(b"x")
    for i in plain_iters:
        (models / f"net_g_{i}.safetensors").write_bytes(b"x")
    if states:
        tstate = repo / "experiments" / name / "training_states"
        tstate.mkdir(parents=True)
        for i in states:
            (tstate / f"{i}.state").write_bytes(b"x")
    if config:
        (repo / "experiments" / name / f"{name}.yml").write_text("name: x\n")


def test_empty_repo(tmp_path: Path):
    (tmp_path / "experiments").mkdir()
    assert list_runs(tmp_path) == []


def test_missing_repo(tmp_path: Path):
    assert list_runs(tmp_path / "nope") == []


def test_lists_checkpoints_sorted(tmp_path: Path):
    _mk_run(tmp_path, "run1", ema_iters=(50000, 5000, 100000))
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert [c.iter for c in runs[0].checkpoints] == [5000, 50000, 100000]
    assert all(c.ema for c in runs[0].checkpoints)
    assert runs[0].checkpoints[-1].path.name == "net_g_ema_100000.safetensors"


def test_falls_back_to_plain_when_no_ema(tmp_path: Path):
    _mk_run(tmp_path, "run1", plain_iters=(5000, 10000))
    runs = list_runs(tmp_path)
    assert [c.iter for c in runs[0].checkpoints] == [5000, 10000]
    assert not any(c.ema for c in runs[0].checkpoints)


def test_prefers_ema_over_plain(tmp_path: Path):
    _mk_run(tmp_path, "run1", ema_iters=(10000,), plain_iters=(10000,))
    runs = list_runs(tmp_path)
    assert [c.iter for c in runs[0].checkpoints] == [10000]
    assert runs[0].checkpoints[0].ema


def test_states_and_resumable(tmp_path: Path):
    _mk_run(tmp_path, "run1", ema_iters=(10000,), states=(5000, 10000))
    run = list_runs(tmp_path)[0]
    assert run.states == [5000, 10000]
    assert run.resumable
    assert run.latest_state.name == "10000.state"


def test_not_resumable_without_state(tmp_path: Path):
    _mk_run(tmp_path, "run1", ema_iters=(10000,))
    run = list_runs(tmp_path)[0]
    assert not run.resumable
    assert run.latest_state is None


def test_config_discovered(tmp_path: Path):
    _mk_run(tmp_path, "my_run", ema_iters=(1000,))
    run = list_runs(tmp_path)[0]
    assert run.config is not None and run.config.name == "my_run.yml"


def test_run_without_models_listed_empty(tmp_path: Path):
    (tmp_path / "experiments" / "archived").mkdir(parents=True)
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].name == "archived"
    assert runs[0].checkpoints == []
    assert not runs[0].resumable


def test_find_run(tmp_path: Path):
    _mk_run(tmp_path, "a", ema_iters=(1000,))
    _mk_run(tmp_path, "b", ema_iters=(2000,))
    assert find_run(tmp_path, "b").name == "b"
    assert find_run(tmp_path, "zz") is None


def test_parse_run_name():
    from hnt.runs import parse_run_name

    assert parse_run_name("NERVE_4x_run") == ("NERVE", 4)
    assert parse_run_name("pulse_fast_2x_run") == ("PULSE_fast", 2)
    assert parse_run_name("HEART_2x_run") == ("HEART", 2)
    assert parse_run_name("something_else") is None


def test_pick_checkpoint(tmp_path: Path):
    from hnt.runs import find_run, pick_checkpoint

    _mk_run(tmp_path, "r", ema_iters=(5000, 10000, 20000))
    run = find_run(tmp_path, "r")
    assert pick_checkpoint(run).iter == 20000
    assert pick_checkpoint(run, 5000).iter == 5000
    assert pick_checkpoint(run, 7000) is None
    assert pick_checkpoint(find_run(tmp_path, "gone")) is None


def test_cli_export_picks_latest_and_reports(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from hnt import cli

    _mk_run(tmp_path, "NERVE_4x_run", ema_iters=(5000, 10000))
    captured = {}

    def fake_export(repo, checkpoint, out_dir, model, scale, **kw):
        captured.update(model=model, scale=scale, ckpt=checkpoint)
        return SimpleNamespace(safetensors=out_dir / "m.safetensors", onnx_fp32=None,
                               onnx_fp16=None, ncnn_dir=None, errors=[])

    monkeypatch.setattr("hnt.exporter.export_checkpoint", fake_export)
    rc = cli.main(["export", "--run", "NERVE_4x_run", "--out", str(tmp_path / "out"),
                   "--repo", str(tmp_path)])
    assert rc == 0
    assert captured["model"] == "NERVE" and captured["scale"] == 4
    assert captured["ckpt"].name == "net_g_ema_10000.safetensors"
    assert "wrote" in capsys.readouterr().out


def test_cli_export_missing_run(tmp_path, capsys):
    from hnt import cli

    assert cli.main(["export", "--run", "nope", "--out", str(tmp_path),
                     "--repo", str(tmp_path)]) == 1
    assert "no run" in capsys.readouterr().err


def test_cli_export_specific_iter(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hnt import cli

    _mk_run(tmp_path, "NERVE_4x_run", ema_iters=(5000, 10000))
    captured = {}

    def fake_export(repo, checkpoint, out_dir, model, scale, **kw):
        captured["ckpt"] = checkpoint
        return SimpleNamespace(safetensors=None, onnx_fp32=None, onnx_fp16=None,
                               ncnn_dir=None, errors=["boom"])

    monkeypatch.setattr("hnt.exporter.export_checkpoint", fake_export)
    rc = cli.main(["export", "--run", "NERVE_4x_run", "--out", str(tmp_path),
                   "--repo", str(tmp_path), "--iter", "5000"])
    assert rc == 1  # exporter reported errors
    assert captured["ckpt"].name == "net_g_ema_5000.safetensors"


def test_archive_run_renames_and_keeps_contents(tmp_path: Path):
    from hnt.runs import archive_run, find_run, list_runs

    _mk_run(tmp_path, "myrun", ema_iters=(1000,), states=(1000,))
    new = archive_run(tmp_path, "myrun")
    assert new is not None and "_archived_" in new.name
    assert new.is_dir() and (new / "models" / "net_g_ema_1000.safetensors").is_file()
    assert find_run(tmp_path, "myrun") is None
    assert any(r.name == new.name for r in list_runs(tmp_path))


def test_archive_run_missing(tmp_path: Path):
    from hnt.runs import archive_run

    assert archive_run(tmp_path, "nope") is None


def test_archive_run_rejects_path_tricks(tmp_path: Path):
    from hnt.runs import archive_run

    assert archive_run(tmp_path, "../escape") is None
    assert archive_run(tmp_path, "a/b") is None


def test_delete_run_removes_directory(tmp_path: Path):
    from hnt.runs import delete_run, find_run

    _mk_run(tmp_path, "gone", ema_iters=(1000,))
    assert delete_run(tmp_path, "gone") is True
    assert find_run(tmp_path, "gone") is None
    assert delete_run(tmp_path, "gone") is False


def test_delete_run_rejects_path_tricks(tmp_path: Path):
    from hnt.runs import delete_run

    (tmp_path / "experiments").mkdir()
    (tmp_path / "keep").mkdir()
    assert delete_run(tmp_path, "../keep") is False
    assert (tmp_path / "keep").is_dir()
