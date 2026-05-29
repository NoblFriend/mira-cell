from __future__ import annotations

import subprocess
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS: dict[str, dict[str, object]] = {
    "data": {
        "dvc_files": [
            "data/train.dvc",
            "data/val.dvc",
            "data/cell_variants.dvc",
        ],
        "marker": REPO_ROOT / "data" / "cell_variants",
    },
    "models": {
        "dvc_files": ["models/best.ckpt.dvc"],
        "marker": REPO_ROOT / "models" / "best.ckpt",
    },
}


def download_data(target: str = "data", force: bool = False) -> None:
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}, expected one of {list(TARGETS)}")

    spec = TARGETS[target]
    marker: Path = spec["marker"]  # type: ignore[assignment]
    dvc_files: list[str] = spec["dvc_files"]  # type: ignore[assignment]

    if marker.exists() and not force:
        print(f"[{target}] already present at {marker}, skipping (pass force=True to re-pull)")
        return

    print(f"[{target}] dvc pull {' '.join(dvc_files)}")
    subprocess.run(["dvc", "pull", *dvc_files], cwd=REPO_ROOT, check=True)
    print(f"[{target}] done")


def cli() -> None:
    fire.Fire(download_data)


if __name__ == "__main__":
    cli()
