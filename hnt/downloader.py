"""Downloads via ``huggingface_hub`` — public, resumable, login optional.

Design: **anonymous by default** (Lucid is public CC0). ``hf_login`` is only for
higher rate limits / gated repos, and is never required to proceed.
"""

from __future__ import annotations

from pathlib import Path

from .manifest import Pretrain

LUCID_REPO = "Phips/lucid-cc0-v2-hc-512"


def lucid_home(data_dir: Path) -> Path:
    """Where the Lucid dataset is kept inside the app data dir."""
    return Path(data_dir) / "datasets" / "lucid"


def lucid_hr_dir(dest: Path) -> Path:
    """HR folder to train from — the repo's ``hr/`` if present, else the root."""
    dest = Path(dest)
    return dest / "hr" if (dest / "hr").is_dir() else dest


def hf_login(token: str | None = None) -> None:
    """Optional sign-in (browser device-code OAuth when ``token`` is None)."""
    from huggingface_hub import login  # lazy: only when the user opts in

    login(token=token, add_to_git_credential=False)


def download_lucid(dest: Path, on_log=lambda m: None) -> Path:
    """Download + extract the Lucid CC0 dataset (resumable)."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    on_log(f"Downloading Lucid (CC0) → {dest} … (large; resumable)")
    path = snapshot_download(repo_id=LUCID_REPO, repo_type="dataset", local_dir=str(dest))
    on_log("Lucid ready.")
    return Path(path)


def resolve_pretrain(pretrain: Pretrain | None, cache_dir: Path) -> str | None:
    """Turn a manifest entry into a local path (download if ``huggingface://``)."""
    if pretrain is None:
        return None
    src = pretrain.path_or_url
    if not src.startswith("huggingface://"):
        return src  # already a local path
    from huggingface_hub import hf_hub_download

    parts = src.replace("huggingface://", "").split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"bad huggingface:// URL: {src}")
    repo_id, filename = "/".join(parts[:2]), parts[2]
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(cache_dir))
