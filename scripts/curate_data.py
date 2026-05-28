"""Apply a manual-curation manifest (relabel / delete) to the dataset.

The manifest is a CSV produced during manual review (one row per reviewed
cell). Expected columns:

- ``action``    — ``relabel`` or ``delete``
- ``rel_path``  — path to the image relative to the repo root,
                  e.g. ``data/train/A/foo.png`` (portable across machines)
- ``new_label`` — target class folder for ``relabel`` (ignored for delete)

``relabel`` moves the file into the sibling ``<split>/<new_label>/`` folder;
``delete`` moves it into ``<split>/__deleted/`` (quarantine, never erased).
Re-running is safe: rows whose source is already gone are skipped.

Dry-run by default — pass ``--apply`` to actually move files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import fire

from mira_cell.constants import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
QUARANTINE_NAME = "__deleted"


def _safe_target(dst: Path) -> Path:
    if not dst.exists():
        return dst
    i = 1
    while True:
        cand = dst.parent / f"{dst.stem}__dup{i}{dst.suffix}"
        if not cand.exists():
            return cand
        i += 1


def _plan_row(rec: dict, repo_root: Path) -> tuple[str, Path | None, Path | None]:
    action = str(rec.get("action", "")).strip().lower()
    rel = str(rec.get("rel_path", "")).strip()
    if not rel:
        return "skipped_no_rel_path", None, None

    src = (repo_root / rel).resolve()
    if not src.exists():
        return "skipped_missing_source", src, None

    split_root = src.parent.parent  # <repo>/data/<split>/<class>/file -> <split>
    if action == "relabel":
        new_label = str(rec.get("new_label", "")).strip()
        if new_label not in CLASS_NAMES:
            return "skipped_invalid_label", src, None
        return "relabel", src, _safe_target(split_root / new_label / src.name)
    if action == "delete":
        return "delete", src, _safe_target(split_root / QUARANTINE_NAME / src.name)
    return "skipped_unknown_action", src, None


def curate(
    manifest: str,
    repo_root: str = str(REPO_ROOT),
    apply: bool = False,
    log_csv: str = "curation_applied_log.csv",
) -> None:
    root = Path(repo_root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(newline="") as handle:
        actions = list(csv.DictReader(handle))

    logs: list[dict] = []
    counts = {"relabel": 0, "delete": 0, "skipped": 0}

    for rec in actions:
        status, src, dst = _plan_row(rec, root)
        if status in ("relabel", "delete"):
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
            counts[status] += 1
        else:
            counts["skipped"] += 1
        logs.append(
            {"status": status, "src": str(src) if src else "", "dst": str(dst) if dst else ""}
        )

    mode = "APPLIED" if apply else "DRY-RUN (no files moved; pass --apply)"
    print(f"[curate] {mode}")
    print(
        f"[curate] relabel={counts['relabel']} delete={counts['delete']} skipped={counts['skipped']}"
    )

    if apply:
        with Path(log_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status", "src", "dst"])
            writer.writeheader()
            writer.writerows(logs)
        print(f"[curate] log written to {log_csv}")


def cli() -> None:
    fire.Fire(curate)


if __name__ == "__main__":
    cli()
