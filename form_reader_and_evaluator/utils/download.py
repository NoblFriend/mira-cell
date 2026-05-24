"""Data fetching: prefer DVC pull, fall back to a direct download stub."""

from __future__ import annotations

from pathlib import Path

import fire


def download_data(target: str = "data", repo_root: str | None = None) -> None:
    """Pull a DVC-tracked artefact into the working copy.

    Falls back to a plain message if dvc is not available — wire your own
    open-source mirror here if you need a no-DVC bootstrap.
    """
    try:
        from dvc.repo import Repo
    except ImportError:
        print("dvc is not installed; install it and rerun, or download data manually.")
        return

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    with Repo(str(root)) as repo:
        repo.pull(targets=[target])


def cli() -> None:
    fire.Fire(download_data)


if __name__ == "__main__":
    cli()
