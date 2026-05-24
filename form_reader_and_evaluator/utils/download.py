from __future__ import annotations

from pathlib import Path

import fire


def download_data(target: str = "data", repo_root: str | None = None) -> None:
    from dvc.repo import Repo

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    with Repo(str(root)) as repo:
        repo.pull(targets=[target])


def cli() -> None:
    fire.Fire(download_data)


if __name__ == "__main__":
    cli()
