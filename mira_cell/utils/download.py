from __future__ import annotations

import tarfile
from pathlib import Path

import fire
import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]

RELEASE_TAG = "v0.1"
RELEASE_URL_TEMPLATE = (
    "https://github.com/NoblFriend/form-reader-and-evaluator/releases/download/{tag}/{asset}"
)

TARGETS: dict[str, dict[str, object]] = {
    "data": {
        "asset": "data.tar.gz",
        "extract_to": REPO_ROOT,
        "marker": REPO_ROOT / "data" / "cell_variants",
    },
    "models": {
        "asset": "models.tar.gz",
        "extract_to": REPO_ROOT,
        "marker": REPO_ROOT / "models" / "best.ckpt",
    },
}


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"  {downloaded >> 20} / {total >> 20} MB ({pct:.1f}%)", end="\r")
    print()


def download_data(target: str = "data", force: bool = False) -> None:
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}, expected one of {list(TARGETS)}")

    spec = TARGETS[target]
    marker: Path = spec["marker"]  # type: ignore[assignment]
    asset: str = spec["asset"]  # type: ignore[assignment]
    extract_to: Path = spec["extract_to"]  # type: ignore[assignment]

    if marker.exists() and not force:
        print(f"[{target}] already present at {marker}, skipping (pass force=True to re-download)")
        return

    url = RELEASE_URL_TEMPLATE.format(tag=RELEASE_TAG, asset=asset)
    archive = REPO_ROOT / asset
    print(f"[{target}] downloading {url}")
    _stream_download(url, archive)

    print(f"[{target}] extracting into {extract_to}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_to)
    archive.unlink()
    print(f"[{target}] done")


def cli() -> None:
    fire.Fire(download_data)


if __name__ == "__main__":
    cli()
