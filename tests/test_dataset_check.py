"""Dataset pre-flight checks — catch bad HR/LR folders before training starts."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from hnt.dataset_check import check_dataset


def _img(path: Path, size: tuple[int, int] = (16, 16), color=(120, 60, 30)) -> None:
    Image.new("RGB", size, color).save(path)


@pytest.fixture()
def paired(tmp_path: Path) -> tuple[Path, Path]:
    hr, lr = tmp_path / "hr", tmp_path / "lr"
    hr.mkdir()
    lr.mkdir()
    for i in range(3):
        _img(hr / f"img{i}.png")
        _img(lr / f"img{i}x4.png")
    return hr, lr


def test_ok_paired_with_scale_suffix(paired):
    hr, lr = paired
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert rep.ok, [i.message for i in rep.issues]
    assert rep.hr_count == 3
    assert rep.lq_count == 3


def test_ok_paired_identical_names(paired):
    hr, lr = paired
    for f in list(lr.glob("*.png")):
        f.rename(f.with_name(f.name.replace("x4", "")))
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert rep.ok, [i.message for i in rep.issues]


def test_missing_hr_dir(tmp_path: Path):
    rep = check_dataset(str(tmp_path / "nope"), "", scale=4)
    assert not rep.ok
    assert "HR" in rep.issues[0].message


def test_empty_hr_dir(tmp_path: Path):
    hr = tmp_path / "hr"
    hr.mkdir()
    rep = check_dataset(str(hr), "", scale=4)
    assert not rep.ok


def test_missing_lr_dir(paired):
    hr, _ = paired
    rep = check_dataset(str(hr), str(paired[0].parent / "gone"), scale=4)
    assert not rep.ok
    assert any("LR" in i.message for i in rep.issues)


def test_count_mismatch(paired):
    hr, lr = paired
    (lr / "img2x4.png").unlink()
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert not rep.ok
    assert any("3" in i.message and "2" in i.message for i in rep.issues)


def test_missing_pair_named(paired):
    hr, lr = paired
    (lr / "img1x4.png").unlink()
    (lr / "img2x4.png").unlink()
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert not rep.ok
    joined = "\n".join(i.message for i in rep.issues)
    assert "img1" in joined


def test_extra_lq_named(paired):
    hr, lr = paired
    _img(lr / "only_lqx4.png")
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert not rep.ok
    joined = "\n".join(i.message for i in rep.issues)
    assert "only_lq" in joined


def test_corrupt_image_flagged(paired):
    hr, lr = paired
    (hr / "img1.png").write_bytes(b"not a png at all")
    rep = check_dataset(str(hr), str(lr), scale=4)
    assert not rep.ok
    assert any("img1" in i.message for i in rep.issues)


def test_otf_mode_needs_only_hr(tmp_path: Path):
    hr = tmp_path / "hr"
    hr.mkdir()
    _img(hr / "a.png")
    rep = check_dataset(str(hr), "", scale=4)
    assert rep.ok, [i.message for i in rep.issues]


def test_subdirs_are_searched(tmp_path: Path):
    hr = tmp_path / "hr"
    (hr / "sub").mkdir(parents=True)
    _img(hr / "sub" / "a.png")
    rep = check_dataset(str(hr), "", scale=4)
    assert rep.ok
    assert rep.hr_count == 1
