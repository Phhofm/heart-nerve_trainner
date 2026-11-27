"""CI parity guard: the suite must COLLECT without torch.

CI installs only ``.[dev]`` (no torch); any unguarded ``import torch``
— in a test file or a first-party module pulled in by one — fails at
collection and turns every push red. The D-NERVE tests did exactly
that (onboarding finding, 2026-11-27). This guard blocks torch in a
subprocess collection run, reproducing CI's import environment, so the
next unguarded import fails here instead of in CI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_suite_collects_without_torch(tmp_path: Path):
    # Simulate torch ABSENCE (what CI sees), not a broken torch: pytest 9
    # skips on ModuleNotFoundError only, so a blocker that raises a plain
    # ImportError from module code would turn clean skips into errors.
    # sitecustomize installs a meta_path finder before any test imports.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "\n"
        "\n"
        "class _BlockTorch:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise ModuleNotFoundError(\n"
        "                \"No module named 'torch'\", name='torch')\n"
        "        return None\n"
        "\n"
        "\n"
        "sys.meta_path.insert(0, _BlockTorch())\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path), QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "collection must succeed without torch (CI has none):\n"
        + result.stdout[-2000:]
    )
