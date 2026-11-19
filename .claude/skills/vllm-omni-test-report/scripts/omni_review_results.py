#!/usr/bin/env python3
"""Locate the latest vllm-omni Buildkite and local result archives in omni-reviewbot.

The JiusiServe/omni-reviewbot repo stores synced metric JSON under
``kanban/data/vllm_omni/buildkite_nightly_raw/<YYYYMMDD>/`` and
``kanban/data/vllm_omni/local_nightly_raw/manual_<YYYYMMDD>/``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

BK_REL = Path("kanban/data/vllm_omni/buildkite_nightly_raw")
LOCAL_REL = Path("kanban/data/vllm_omni/local_nightly_raw")
_DATE_DIR_RE = re.compile(r"^20\d{6}$")
_MANUAL_RE = re.compile(r"^manual_(20\d{6})$")

_DEFAULT_CANDIDATES = (
    Path("/workspace/omni-reviewbot"),
    Path("/home/wy/omni-reviewbot"),
    Path.home() / "omni-reviewbot",
)


def resolve_omni_review_repo(explicit: Path | str | None = None) -> Path | None:
    """Return the omni-reviewbot checkout, or ``None`` if it cannot be found."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("OMNI_REVIEW_REPO_ROOT") or os.environ.get("OMNI_REVIEWBOT_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(_DEFAULT_CANDIDATES)
    seen: set[Path] = set()
    for raw in candidates:
        try:
            path = raw.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        if (path / BK_REL).is_dir() or (path / LOCAL_REL).is_dir() or (path / ".git").exists():
            if path.is_dir():
                return path
    return None


def maybe_ff_pull_omni_review(repo: Path) -> bool:
    """Fast-forward ``repo`` to ``origin/main`` when the tree is clean.

    Failures are printed to stderr and never raised — the caller can still
    use whatever is already checked out.
    """
    if not (repo / ".git").exists():
        return False
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        if dirty.stdout.strip():
            print(
                f"omni-reviewbot at {repo} has a dirty tree; using the checked-out copy",
                file=sys.stderr,
            )
            return False
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["git", "merge", "--ff-only", "origin/main"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"omni-reviewbot ff-only pull skipped: {(result.stderr or result.stdout).strip()}",
                file=sys.stderr,
            )
            return False
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"omni-reviewbot update failed — {exc}; using the checked-out copy", file=sys.stderr)
        return False


def _latest_named_dir(root: Path, *, pattern: re.Pattern[str], name_key=lambda p: p.name) -> Path | None:
    if not root.is_dir():
        return None
    matches = [path for path in root.iterdir() if path.is_dir() and pattern.match(path.name)]
    if not matches:
        return None
    return max(matches, key=name_key)


def latest_buildkite_raw_dir(repo: Path) -> Path | None:
    return _latest_named_dir(repo / BK_REL, pattern=_DATE_DIR_RE)


def latest_local_raw_dir(repo: Path) -> Path | None:
    return _latest_named_dir(
        repo / LOCAL_REL,
        pattern=_MANUAL_RE,
        name_key=lambda p: _MANUAL_RE.match(p.name).group(1) if _MANUAL_RE.match(p.name) else p.name,
    )


def latest_vllm_omni_result_dirs(repo: Path) -> dict[str, Path | str | None]:
    """Return the latest Buildkite and Local archive directories under ``repo``."""
    bk = latest_buildkite_raw_dir(repo)
    local = latest_local_raw_dir(repo)
    return {
        "buildkite": bk,
        "buildkite_date": bk.name if bk is not None else None,
        "local": local,
        "local_date": _MANUAL_RE.match(local.name).group(1) if local is not None else None,
    }


def perf_acc_log_dirs_from_omni_review(repo: Path) -> list[tuple[str, Path]]:
    """``(Host label, directory)`` pairs for the Performance / Accuracy chapters."""
    found = latest_vllm_omni_result_dirs(repo)
    rows: list[tuple[str, Path]] = []
    if isinstance(found["buildkite"], Path):
        rows.append(("Buildkite", found["buildkite"]))
    if isinstance(found["local"], Path):
        rows.append(("Local", found["local"]))
    return rows
