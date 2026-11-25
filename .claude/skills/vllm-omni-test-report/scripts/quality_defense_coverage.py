#!/usr/bin/env python3
"""Coverage percentages for the Quality Defense Radar (stdlib only).

Performance and Precision
-------------------------
Denominator: rows in the Model care scenario table whose ``kind`` is
``performance`` or ``accuracy``. Source file, in order:

1. ``$MODEL_CARE_SCENARIOS_JSON`` when it points at a file
2. ``kanban/docs/assets/care/model_care_scenarios.json`` inside a
   JiusiServe/omni-reviewbot checkout (``$OMNI_REVIEW_REPO_ROOT``,
   ``$OMNI_REVIEWBOT_ROOT``, ``/workspace/omni-reviewbot``,
   ``/home/wy/omni-reviewbot``, ``~/omni-reviewbot``)
3. The snapshot shipped next to this skill
   (``references/model_care_scenarios.json``)

The checkout wins over the snapshot, so a newly added scenario is picked
up without editing this module. A flagship model with **no** rows for an
axis is 100% on a hardware side only when this repo has a real test of
that type on that side (performance JSON for Performance, an accuracy
test for Precision). A side with no such test stays at 0%. Matching uses
the radar keyword list; the three ids in today's JSON are not hardcoded.

A row with hardware ``GPU`` counts on the GPU half only, ``NPU`` on the
NPU half only, and ``GPU/NPU`` on both halves. The numerator is a real
test case under ``tests/dfx/perf/tests/*.json`` (performance) or
``tests/e2e/accuracy/**/*.py`` (accuracy) that matches the row's model,
dataset, full-duplex vs non-full-duplex mode, async-chunk feature, and
hardware side. Traffic strings and deploy card counts are not extra
splits. ``async chunk + prefix`` matches a case that enables async chunk;
the bundled "prefix" token is not required on its own. ``default``
matches only an explicit non-async case (``--no-async-chunk`` or
``async_chunk: false``). Feature ``None`` matches a case that does not
enable async chunk. A row whose metrics mention COT is covered only by
a case that mentions COT. GPU is the default hardware of these tests;
NPU requires an ``npu`` / ``ascend`` marker or a reference from an NPU
Buildkite YAML. No pass/fail score and no metric value is invented.

Reliability
-----------
Denominator: the in-scope fault-mode cells from
https://github.com/JiusiServe/vllm-omni-project-manage/issues/133
section 「可靠性测试需求规划（检测 / 响应 / 恢复）」 and the 0.34.0
必达 rows of the 场景×模型缺口 table. Each scenario is scored on
detect / respond / recover (优雅退出的「清理」counts as recover).
Timings the issue prints as 「拉起/空闲」are separate cells, because the
issue says an idle case does not cover startup. Host memory, SIGSTOP
hang, overload, and dependency faults are out of scope and are not in
the denominator.

Numerator: non-skipped tests under ``tests/dfx/reliability/test_reliability_*.py``.
An unconditional ``@pytest.mark.skip`` (or a skipped ``pytest.param`` for
the required signal) does not count. Crash cells require an active
SIGKILL parameter; SIGTERM / SIGINT do not count as crash coverage.
``no_load`` cases map to 空闲 (the server fixture is already up), not
拉起. Graceful-exit cells require a test whose name is the dedicated
SIGTERM exit scenario; a multi-signal kill matrix is not that scenario.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

RELIABILITY_ISSUE_URL = "https://github.com/JiusiServe/vllm-omni-project-manage/issues/133"

_MODEL_CARE_REL = Path("kanban/docs/assets/care/model_care_scenarios.json")
_BUNDLED_MODEL_CARE = (
    Path(__file__).resolve().parent.parent / "references" / "model_care_scenarios.json"
)

# In-scope cells. id suffix is the timing. Three dimensions each.
# Kept in this module (with the issue URL) so report generation does not
# fetch GitHub.
_RELIABILITY_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("oom_startup", "GPU OOM at startup"),
    ("oom_under_load", "GPU OOM under load"),
    ("tree_startup", "Full-tree SIGKILL at startup"),
    ("tree_idle", "Full-tree SIGKILL while idle"),
    ("tree_under_load", "Full-tree SIGKILL under load"),
    ("stage_startup", "Partial stage SIGKILL at startup"),
    ("stage_idle", "Partial stage SIGKILL while idle"),
    ("stage_under_load", "Partial stage SIGKILL under load"),
    ("stage_during_exit", "Partial stage SIGKILL during exit"),
    ("replica_startup", "Partial replica SIGKILL at startup"),
    ("replica_idle", "Partial replica SIGKILL while idle"),
    ("replica_under_load", "Partial replica SIGKILL under load"),
    ("replica_during_exit", "Partial replica SIGKILL during exit"),
    ("root_startup", "Serve-root SIGKILL at startup"),
    ("root_idle", "Serve-root SIGKILL while idle"),
    ("root_under_load", "Serve-root SIGKILL under load"),
    ("exit_single", "Graceful SIGTERM exit, single replica"),
    ("exit_multi", "Graceful SIGTERM exit, multi replica"),
)
_RELIABILITY_DIMENSIONS = ("detect", "respond", "recover")

_SKIP_ASSIGN_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*pytest\.mark\.skip\(",
    re.M,
)
_SIG_RE = re.compile(r"SIG(?:KILL|TERM|INT)")
_DEF_RE = re.compile(r"^def\s+(test_[A-Za-z0-9_]+)\(", re.M)
_ANY_DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\(", re.M)
_PARAM_LIST_RE = re.compile(
    r"""@pytest\.mark\.parametrize\(\s*['"](\w+)['"]\s*,\s*([A-Z][A-Z0-9_]*)"""
)
_ASYNC_TRUE_RE = re.compile(r"async_chunk[\"']?\s*[:=]\s*true\b")
_ASYNC_FALSE_RE = re.compile(r"async_chunk[\"']?\s*[:=]\s*false\b")


def reliability_cell_count() -> int:
    """Number of in-scope detect/respond/recover cells per radar model."""
    return len(_RELIABILITY_SCENARIOS) * len(_RELIABILITY_DIMENSIONS)


def coverage_percent(covered: int, total: int) -> int:
    """Integer percent, half-up. ``total <= 0`` is an empty requirement (100)."""
    if total <= 0:
        return 100
    if covered < 0:
        covered = 0
    if covered > total:
        covered = total
    return (100 * covered + total // 2) // total


def resolve_model_care_json(explicit: Path | str | None = None) -> Path | None:
    """Locate the Model care scenario JSON. Checkout beats the snapshot."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_file = os.environ.get("MODEL_CARE_SCENARIOS_JSON")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    for root_env in ("OMNI_REVIEW_REPO_ROOT", "OMNI_REVIEWBOT_ROOT"):
        raw = os.environ.get(root_env)
        if raw:
            candidates.append(Path(raw).expanduser() / _MODEL_CARE_REL)
    for root in (
        Path("/workspace/omni-reviewbot"),
        Path("/home/wy/omni-reviewbot"),
        Path.home() / "omni-reviewbot",
    ):
        candidates.append(root / _MODEL_CARE_REL)
    candidates.append(_BUNDLED_MODEL_CARE)
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def load_model_care_scenarios(path: Path | None) -> list[dict]:
    """Return the ``scenarios`` list, or ``[]`` when the file is missing."""
    if path is None or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        return []
    return [item for item in scenarios if isinstance(item, dict)]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _scenario_blob(scenario: dict) -> str:
    parts = [str(scenario.get("id") or ""), str(scenario.get("title") or "")]
    aliases = scenario.get("aliases") or []
    if isinstance(aliases, list):
        parts.extend(str(alias) for alias in aliases)
    return _norm(" ".join(parts))


def matching_scenarios(
    scenarios: list[dict],
    model_id: str,
    keywords: tuple[str, ...],
) -> list[dict]:
    """Scenarios whose id, title, or alias contains a radar keyword."""
    del model_id  # matching is keyword-based so new JSON ids attach themselves
    norms = [_norm(kw) for kw in keywords if _norm(kw)]
    matched: list[dict] = []
    for scenario in scenarios:
        blob = _scenario_blob(scenario)
        if not blob:
            continue
        if any(kw in blob for kw in norms):
            matched.append(scenario)
    return matched


def _hardware_sides(hardware: str) -> tuple[str, ...]:
    text = (hardware or "").upper().replace(" ", "")
    gpu = "GPU" in text
    npu = "NPU" in text
    if gpu and npu:
        return ("gpu", "npu")
    if npu:
        return ("npu",)
    if gpu:
        return ("gpu",)
    return ("gpu", "npu")


def _row_kind(row: dict) -> str | None:
    kind = str(row.get("kind") or "").strip().lower()
    if kind in {"performance", "perf"}:
        return "perf"
    if kind in {"accuracy", "precision", "prec"}:
        return "prec"
    return None


def _row_datasets(dataset: str) -> frozenset[str]:
    compact = _norm(dataset)
    if "duplexeval" in compact and "omniinteract" in compact:
        return frozenset({"duplexeval", "omniinteract"})
    if "duplexeval" in compact:
        return frozenset({"duplexeval"})
    if "omniinteract" in compact:
        return frozenset({"omniinteract"})
    if "dailyomni" in compact:
        return frozenset({"daily-omni"})
    if "seedtts" in compact:
        return frozenset({"seed-tts"})
    if compact == "random":
        return frozenset({"random"})
    if compact == "custom":
        return frozenset({"custom"})
    if "singleimage" in compact:
        return frozenset({"single-image"})
    return frozenset({compact}) if compact else frozenset()


def _row_is_full_duplex(row: dict) -> bool:
    mode = str(row.get("mode") or "").lower()
    if "non-full" in mode or "non full" in mode:
        return False
    return "full duplex" in mode or "fullduplex" in mode


def _row_feature(row: dict) -> str:
    feature = str(row.get("feature") or "").strip().lower()
    if feature in {"", "none", "null"}:
        return "none"
    if "async" in feature:
        return "async"
    if feature == "default":
        return "default"
    return feature


def _row_requires_cot(row: dict) -> bool:
    try:
        blob = json.dumps(row, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        blob = str(row).lower()
    return re.search(r"\bcot\b", blob) is not None


def _strip_docstrings(text: str) -> str:
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return re.sub(r"'''[\s\S]*?'''", "", text)


def _async_flag(text: str) -> bool | None:
    low = text.lower()
    negative = "--no-async-chunk" in low or _ASYNC_FALSE_RE.search(low) is not None
    positive = (
        ("--async-chunk" in low and "--no-async-chunk" not in low)
        or _ASYNC_TRUE_RE.search(low) is not None
        or "get_chunk_config" in low
    )
    if positive and negative:
        return None
    if positive:
        return True
    if negative:
        return False
    return None


def _datasets_in_text(text: str) -> set[str]:
    low = text.lower()
    compact = _norm(text)
    found: set[str] = set()
    if "seed-tts" in low or "seed_tts" in low:
        found.add("seed-tts")
    if "daily-omni" in low or "daily_omni" in low:
        found.add("daily-omni")
    if "duplexeval" in compact or "duplex_eval" in low or "duplex-eval" in low:
        found.add("duplexeval")
    if "omniinteract" in compact:
        found.add("omniinteract")
    if re.search(r"\bpsnr\b", low) and re.search(r"\bssim\b", low):
        found.add("single-image")
    if "pixel_accuracy" in low or "pixel accuracy" in low:
        found.add("single-image")
    return found


def _tags_in_text(text: str) -> set[str]:
    low = text.lower()
    tags: set[str] = set()
    if re.search(r"\bcot\b", low):
        tags.add("cot")
    return tags


def _is_duplex_text(text: str, datasets: set[str]) -> bool:
    if "duplexeval" in datasets or "omniinteract" in datasets:
        return True
    low = text.lower()
    return (
        "full duplex" in low
        or "full_duplex" in low
        or "fullduplex" in low
        or "realtime-duplex" in low
        or "realtime_duplex" in low
    )


def _sides_from_marks(item: dict) -> tuple[bool, bool] | None:
    """Read ``mark[].hardware_marks.res`` (``cuda``/``rocm`` → GPU, ``npu`` → NPU)."""
    marks = item.get("mark")
    if not isinstance(marks, list):
        return None
    gpu = False
    npu = False
    found = False
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        hardware = mark.get("hardware_marks")
        res = hardware.get("res") if isinstance(hardware, dict) else None
        if not isinstance(res, dict):
            continue
        found = True
        keys = {str(key).lower() for key in res}
        if keys & {"cuda", "rocm", "gpu", "xpu"}:
            gpu = True
        if "npu" in keys:
            npu = True
    if not found:
        return None
    return gpu, npu


def _hardware_flags(text: str, path: Path, npu_yaml: str) -> tuple[bool, bool]:
    low = text.lower()
    npu = bool(re.search(r"\bnpu\b", low) or "ascend" in low)
    gpu = bool(re.search(r"\bcuda\b|\bgpu\b|\bh100\b|\brocm\b", low))
    if npu_yaml:
        name = path.name
        rel = str(path).replace("\\", "/")
        if name and name in npu_yaml:
            npu = True
        elif rel in npu_yaml.replace("\\", "/"):
            npu = True
    if npu and not gpu:
        return False, True
    if gpu or not npu:
        return True, npu
    return True, True


def _models_in_text(text: str, keywords: dict[str, tuple[str, ...]]) -> set[str]:
    blob = _norm(text)
    found: set[str] = set()
    for model_id, kws in keywords.items():
        if any(_norm(kw) and _norm(kw) in blob for kw in kws):
            found.add(model_id)
    return found


def _read_npu_yaml(repo_root: Path) -> str:
    chunks: list[str] = []
    for rel in (
        ".buildkite/test-npu-nightly.yaml",
        ".buildkite/test-npu-ready.yaml",
        ".buildkite/npu/test-npu-nightly.yml",
        ".buildkite/npu/test-npu-ready.yml",
    ):
        path = repo_root / rel
        if path.is_file():
            try:
                chunks.append(path.read_text(errors="ignore"))
            except OSError:
                continue
    npu_dir = repo_root / ".buildkite" / "npu"
    if npu_dir.is_dir():
        for path in sorted(npu_dir.glob("*.y*ml")):
            try:
                chunks.append(path.read_text(errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


def _match_bracket(text: str, open_at: int) -> int:
    depth = 0
    for idx in range(open_at, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return idx
    return len(text) - 1


def _top_level_params(body: str) -> list[str]:
    """Split a ``[ pytest.param(...), ... ]`` body into param chunks."""
    chunks: list[str] = []
    depth = 0
    start: int | None = None
    for idx, char in enumerate(body):
        if char in "([{":
            if depth == 1 and char == "(" and start is None:
                start = idx
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 1 and start is not None and char == ")":
                chunks.append(body[start : idx + 1])
                start = None
            if depth < 0:
                break
    return chunks


def _signal_lists(text: str) -> dict[str, set[str]]:
    skip_names = set(_SKIP_ASSIGN_RE.findall(text))
    found: dict[str, set[str]] = {}
    for match in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*\[", text, re.M):
        open_at = match.end() - 1
        close_at = _match_bracket(text, open_at)
        body = text[open_at : close_at + 1]
        active: set[str] = set()
        params = _top_level_params(body)
        pieces = params or [body]
        for piece in pieces:
            if re.search(r"pytest\.mark\.skip\(", piece):
                continue
            if any(re.search(rf"\b{re.escape(name)}\b", piece) for name in skip_names):
                continue
            active.update(_SIG_RE.findall(piece))
        if active:
            found[match.group(1)] = active
    return found


def _iter_test_functions(text: str) -> list[tuple[str, str, str]]:
    """Return ``(name, decorators, body)`` for each ``def test_*``."""
    matches = list(_DEF_RE.finditer(text))
    items: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        prev = matches[index - 1].end() if index else 0
        preamble = text[prev : match.start()]
        deco_lines: list[str] = []
        started = False
        for line in reversed(preamble.splitlines()):
            stripped = line.strip()
            if not stripped:
                if started:
                    break
                continue
            if line.startswith(("def ", "class ")):
                break
            deco_lines.append(line)
            started = True
        decorators = "\n".join(reversed(deco_lines))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start() : end]
        body = re.sub(r"\n(?:[ \t]*@.*\n)+\s*$", "\n", body)
        items.append((match.group(1), decorators, body))
    return items


def _function_skipped(decorators: str) -> bool:
    return re.search(r"@pytest\.mark\.skip\(", decorators) is not None


def _active_signals(decorators: str, body: str, lists: dict[str, set[str]]) -> set[str]:
    signals: set[str] = set()
    for _arg, list_name in _PARAM_LIST_RE.findall(decorators):
        signals |= lists.get(list_name, set())
    if signals:
        return signals
    return set(_SIG_RE.findall(decorators + "\n" + body))


def _timing(name: str) -> str:
    if "with_load" in name or "under_load" in name:
        return "under_load"
    if any(token in name for token in ("startup", "bringup", "before_ready", "during_launch")):
        return "startup"
    if "during_exit" in name or "on_exit" in name:
        return "during_exit"
    return "idle"


def _dimensions(body: str) -> set[str]:
    low = body.lower()
    detect = (
        "assert_fault" in low
        or "failure_observed" in low
        or "fault_observed" in low
        or ("expected" in low and "fail" in low)
        or "status >= 500" in low
        or "status_code >= 500" in low
        or "get_health" in low
        or "/health" in low
        or "health" in low
        or "fast_fail" in low
    )
    respond = (
        "fast_fail" in low
        or "elapsed" in low
        or "hung" in low
        or "not hang" in low
        or "pending" in low
        or "failure_observed" in low
        or "status >= 500" in low
        or "status_code >= 500" in low
        or "assert_fault" in low
    )
    # Cleanup / restart assertions only. "converge" and "convergence" in a
    # fast-fail bound are response-time checks, not the recover dimension.
    recover = (
        "gpu_release" in low
        or re.search(r"\bresidual\b", low) is not None
        or re.search(r"\brecover(?:y|s|ed|ing)?\b", low) is not None
        or re.search(r"\brestart(?:ed|s|ing)?\b", low) is not None
    )
    dims: set[str] = set()
    if detect:
        dims.add("detect")
    if respond:
        dims.add("respond")
    if recover:
        dims.add("recover")
    return dims


def _cells_for_test(name: str, body: str, signals: set[str]) -> set[tuple[str, str]]:
    """Map one active test to ``(scenario_id, dimension)`` cells."""
    lname = name.lower()
    dims = _dimensions(body)
    if not dims:
        return set()
    crash_ok = "SIGKILL" in signals

    def pack(scenario_id: str) -> set[tuple[str, str]]:
        return {(scenario_id, dim) for dim in dims}

    if "oom" in lname:
        scenario = "oom_startup" if _timing(lname) == "startup" else "oom_under_load"
        return pack(scenario)

    if "graceful" in lname or "normal_exit" in lname:
        if "SIGTERM" not in signals and "sigterm" not in lname:
            return set()
        scenario = "exit_multi" if ("multi" in lname or "replica" in lname) else "exit_single"
        return pack(scenario)

    if not crash_ok:
        return set()

    if "replica" in lname:
        return pack(f"replica_{_timing(lname)}")
    if "tree" in lname:
        return pack(f"tree_{_timing(lname)}")
    if "serve_root" in lname or "server_root" in lname:
        return pack(f"root_{_timing(lname)}")
    if "during_exit" in lname or "on_exit" in lname:
        return pack(f"stage_{_timing(lname)}")
    return pack(f"stage_{_timing(lname)}")


def _file_matches_model(filename: str, keywords: tuple[str, ...]) -> bool:
    blob = _norm(filename)
    return any(_norm(kw) and _norm(kw) in blob for kw in keywords)


def scan_reliability_cells(
    repo_root: Path | None,
    *,
    keywords: dict[str, tuple[str, ...]],
) -> dict[str, set[tuple[str, str]]]:
    """Active fault-mode cells per radar model id."""
    found = {model_id: set() for model_id in keywords}
    if repo_root is None or not repo_root.is_dir():
        return found
    root = repo_root / "tests" / "dfx" / "reliability"
    if not root.is_dir():
        return found
    known = {scenario_id for scenario_id, _label in _RELIABILITY_SCENARIOS}
    for path in sorted(root.glob("test_reliability_*.py")):
        owners = [mid for mid, kws in keywords.items() if _file_matches_model(path.name, kws)]
        if not owners:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lists = _signal_lists(text)
        for func_name, decorators, body in _iter_test_functions(text):
            if _function_skipped(decorators):
                continue
            signals = _active_signals(decorators, body, lists)
            cells = {(sid, dim) for sid, dim in _cells_for_test(func_name, body, signals) if sid in known}
            for model_id in owners:
                found[model_id] |= cells
    return found


def _helper_sources(text: str) -> dict[str, str]:
    matches = list(_ANY_DEF_RE.finditer(text))
    sources: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sources[match.group(1)] = text[match.start() : end]
    return sources


def _expand_callees(body: str, helpers: dict[str, str]) -> str:
    extra: list[str] = []
    for name, source in helpers.items():
        if name.startswith("test_"):
            continue
        if re.search(rf"\b{re.escape(name)}\b", body):
            extra.append(source)
    return body + "\n" + "\n".join(extra)


def _case_record(
    *,
    kind: str,
    models: set[str],
    datasets: set[str],
    tags: set[str],
    async_chunk: bool | None,
    duplex: bool,
    gpu: bool,
    npu: bool,
    source: str,
) -> dict:
    return {
        "kind": kind,
        "models": models,
        "datasets": datasets,
        "tags": tags,
        "async_chunk": async_chunk,
        "duplex": duplex,
        "gpu": gpu,
        "npu": npu,
        "source": source,
    }


def _header_before_first_test(text: str) -> str:
    match = _DEF_RE.search(text)
    header = text[: match.start()] if match else text
    return _strip_docstrings(header)


def scan_model_care_cases(
    repo_root: Path | None,
    *,
    keywords: dict[str, tuple[str, ...]],
) -> list[dict]:
    """Performance and accuracy cases that can cover a Model care row."""
    if repo_root is None or not repo_root.is_dir():
        return []
    npu_yaml = _read_npu_yaml(repo_root)
    cases: list[dict] = []
    perf_dir = repo_root / "tests" / "dfx" / "perf" / "tests"
    if perf_dir.is_dir():
        for path in sorted(perf_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                server = item.get("server_params") if isinstance(item.get("server_params"), dict) else {}
                model_text = " ".join(
                    [
                        path.name,
                        str(item.get("test_name") or ""),
                        str(server.get("model") or ""),
                        json.dumps(server, ensure_ascii=False),
                    ]
                )
                models = _models_in_text(model_text, keywords)
                if not models:
                    continue
                async_chunk = _async_flag(json.dumps(server, ensure_ascii=False))
                if async_chunk is None:
                    async_chunk = _async_flag(json.dumps(item.get("server_params") or {}, ensure_ascii=False))
                benches = item.get("benchmark_params")
                bench_rows = benches if isinstance(benches, list) and benches else [item]
                for bench in bench_rows:
                    if not isinstance(bench, dict):
                        continue
                    raw_dataset = ""
                    for key in ("dataset", "dataset_name", "dataset-name"):
                        value = bench.get(key)
                        if isinstance(value, str) and value.strip():
                            raw_dataset = value.strip()
                            break
                    datasets = set(_row_datasets(raw_dataset))
                    marked = _sides_from_marks(item)
                    if marked is None:
                        gpu, npu = _hardware_flags(
                            model_text + "\n" + json.dumps(bench, ensure_ascii=False),
                            path,
                            npu_yaml,
                        )
                    else:
                        gpu, npu = marked
                    test_name = str(item.get("test_name") or "")
                    if re.search(r"(^|_)npu($|_)", test_name.lower()):
                        npu = True
                    duplex_blob = " ".join(
                        [
                            raw_dataset,
                            test_name,
                            str(bench.get("backend") or ""),
                            json.dumps(bench.get("extra_body") or {}),
                        ]
                    )
                    label = str(bench.get("name") or test_name or path.stem)
                    cases.append(
                        _case_record(
                            kind="perf",
                            models=models,
                            datasets=datasets,
                            tags=_tags_in_text(json.dumps(bench)),
                            async_chunk=async_chunk,
                            duplex=_is_duplex_text(duplex_blob, datasets),
                            gpu=gpu,
                            npu=npu,
                            source=f"{path.relative_to(repo_root).as_posix()}:{label}",
                        )
                    )
    acc_root = repo_root / "tests" / "e2e" / "accuracy"
    if acc_root.is_dir():
        for path in sorted(acc_root.rglob("*.py")):
            if path.name == "conftest.py":
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not _models_in_text(str(path.relative_to(repo_root)), keywords) and not _models_in_text(
                path.name, keywords
            ):
                # File path did not name a radar model. Still accept the file
                # when a test function's own text names one (rare).
                file_models: set[str] = set()
            else:
                file_models = _models_in_text(str(path.relative_to(repo_root)) + "\n" + path.name, keywords)
            header = _header_before_first_test(text)
            header_async = _async_flag(header)
            helpers = _helper_sources(text)
            rel = path.relative_to(repo_root).as_posix()
            for func_name, decorators, body in _iter_test_functions(text):
                expanded = _expand_callees(decorators + "\n" + body, helpers)
                models = set(file_models) | _models_in_text(func_name + "\n" + expanded, keywords)
                if not models:
                    continue
                func_async = _async_flag(expanded)
                async_chunk = func_async if func_async is not None else header_async
                datasets = _datasets_in_text(func_name + "\n" + expanded)
                gpu, npu = _hardware_flags(decorators + "\n" + expanded, path, npu_yaml)
                cases.append(
                    _case_record(
                        kind="prec",
                        models=models,
                        datasets=datasets,
                        tags=_tags_in_text(expanded),
                        async_chunk=async_chunk,
                        duplex=_is_duplex_text(expanded, datasets),
                        gpu=gpu,
                        npu=npu,
                        source=f"{rel}:{func_name}",
                    )
                )
    return cases


def _case_covers(case: dict, row: dict, side: str, *, dataset_token: str | None = None) -> bool:
    kind = _row_kind(row)
    if kind is None or case["kind"] != kind:
        return False
    if side == "gpu" and not case["gpu"]:
        return False
    if side == "npu" and not case["npu"]:
        return False
    needed = frozenset({dataset_token}) if dataset_token else _row_datasets(str(row.get("dataset") or ""))
    if not needed or not needed <= case["datasets"]:
        return False
    if _row_requires_cot(row) and "cot" not in case["tags"]:
        return False
    full = _row_is_full_duplex(row)
    if full and not case["duplex"]:
        return False
    if not full and case["duplex"]:
        return False
    feature = _row_feature(row)
    # A full-duplex bench that does not repeat async_chunk still covers the
    # duplex care row. An explicit ``async_chunk: false`` does not.
    if feature == "async" and case["async_chunk"] is not True:
        if not (full and case["async_chunk"] is None):
            return False
    if feature == "default" and case["async_chunk"] is not False:
        return False
    if feature == "none" and case["async_chunk"] is True:
        return False
    return True


def _empty_side() -> dict:
    return {"covered": 0, "total": 0, "pct": 0, "rows": []}


def _presence_side(cases: list[dict], kind: str, side: str) -> dict:
    """100% when any real case of ``kind`` exists on ``side``, else 0%."""
    hits = [case["source"] for case in cases if case["kind"] == kind and case.get(side)]
    present = bool(hits)
    return {
        "covered": 1 if present else 0,
        "total": 1 if present else 0,
        "pct": 100 if present else 0,
        "rows": [],
        "by": hits[0] if hits else "",
    }


def score_model_care(
    scenarios: list[dict],
    cases: list[dict],
    *,
    model_id: str,
    keywords: tuple[str, ...],
) -> dict[str, dict]:
    """Performance and Precision fractions for one radar model.

    No Model care rows for an axis → ``basis="repo-tests"``: 100% on a
    hardware side only when a real case of that kind exists there, else 0%.
    """
    matched = matching_scenarios(scenarios, model_id, keywords)
    model_cases = [case for case in cases if model_id in case["models"]]
    result = {
        "perf": _empty_side() | {"basis": "repo-tests", "gpu": _empty_side(), "npu": _empty_side()},
        "prec": _empty_side() | {"basis": "repo-tests", "gpu": _empty_side(), "npu": _empty_side()},
    }
    if not matched:
        for axis, kind in (("perf", "perf"), ("prec", "prec")):
            gpu = _presence_side(model_cases, kind, "gpu")
            npu = _presence_side(model_cases, kind, "npu")
            single = 100 if gpu["pct"] == 100 or npu["pct"] == 100 else 0
            result[axis] = {
                "basis": "repo-tests",
                "gpu": gpu,
                "npu": npu,
                "covered": gpu["covered"] + npu["covered"],
                "total": gpu["total"] + npu["total"],
                "pct": single,
                "single": single,
            }
        return result
    for axis in ("perf", "prec"):
        result[axis] = {
            "basis": "model-care",
            "gpu": {"covered": 0, "total": 0, "rows": []},
            "npu": {"covered": 0, "total": 0, "rows": []},
        }
    for scenario in matched:
        for row in scenario.get("rows") or []:
            if not isinstance(row, dict):
                continue
            axis = _row_kind(row)
            if axis is None:
                continue
            for side in _hardware_sides(str(row.get("hardware") or "")):
                bucket = result[axis][side]
                needed = _row_datasets(str(row.get("dataset") or ""))
                # A care row that names two datasets (duplexEval-OmniInteract)
                # is covered when each dataset has its own matching case.
                # The repo records those benches in separate JSON files.
                if len(needed) > 1:
                    hits: list[str] = []
                    covered = True
                    for token in sorted(needed):
                        token_hits = [
                            case["source"]
                            for case in model_cases
                            if _case_covers(case, row, side, dataset_token=token)
                        ]
                        if not token_hits:
                            covered = False
                            hits = []
                            break
                        hits.append(token_hits[0])
                else:
                    matched_cases = [case for case in model_cases if _case_covers(case, row, side)]
                    if not _row_requires_cot(row):
                        tighter = [case for case in matched_cases if "cot" not in case["tags"]]
                        if tighter:
                            matched_cases = tighter
                    hits = [case["source"] for case in matched_cases]
                    covered = bool(hits)
                bucket["total"] += 1
                if covered:
                    bucket["covered"] += 1
                bucket["rows"].append(
                    {
                        "mode": row.get("mode"),
                        "feature": row.get("feature"),
                        "dataset": row.get("dataset"),
                        "hardware": row.get("hardware"),
                        "covered": covered,
                        "by": "; ".join(hits),
                    }
                )
    for axis, kind in (("perf", "perf"), ("prec", "prec")):
        gpu = result[axis]["gpu"]
        npu = result[axis]["npu"]
        # A side the care table does not mention is presence-based, same as
        # a model with no rows: full only when a real case exists.
        if gpu["total"] == 0:
            present = _presence_side(model_cases, kind, "gpu")
            gpu["pct"] = present["pct"]
            gpu["by"] = present["by"]
        else:
            gpu["pct"] = coverage_percent(gpu["covered"], gpu["total"])
        if npu["total"] == 0:
            present = _presence_side(model_cases, kind, "npu")
            npu["pct"] = present["pct"]
            npu["by"] = present["by"]
        else:
            npu["pct"] = coverage_percent(npu["covered"], npu["total"])
        combined_cov = gpu["covered"] + npu["covered"]
        combined_total = gpu["total"] + npu["total"]
        result[axis]["covered"] = combined_cov
        result[axis]["total"] = combined_total
        if combined_total == 0:
            result[axis]["pct"] = 100 if gpu["pct"] == 100 or npu["pct"] == 100 else 0
        else:
            result[axis]["pct"] = coverage_percent(combined_cov, combined_total)
        result[axis]["single"] = result[axis]["pct"]
        result[axis]["basis"] = "model-care"
    return result


def score_reliability(cells: set[tuple[str, str]]) -> dict:
    """Reliability fraction against the issue 133 library."""
    total = reliability_cell_count()
    known = {scenario_id for scenario_id, _label in _RELIABILITY_SCENARIOS}
    hit = sorted(f"{sid}:{dim}" for sid, dim in cells if sid in known and dim in _RELIABILITY_DIMENSIONS)
    covered = len(hit)
    return {
        "basis": "issue-133",
        "issue": RELIABILITY_ISSUE_URL,
        "covered": covered,
        "total": total,
        "pct": coverage_percent(covered, total),
        "hits": hit,
    }


def attach_quality_percentages(
    coverage: dict[str, dict],
    repo_root: Path | None,
    *,
    has_rel: frozenset[str] | set[str],
    single_mode: frozenset[str] | set[str],
    keywords: dict[str, tuple[str, ...]],
    model_care_path: Path | str | None = None,
) -> dict[str, dict]:
    """Add a ``pct`` block to each model's coverage dict.

    Binary axes (``gpu`` / ``npu`` sets, ``rel`` bool) are left to the caller.
    Performance, Precision, and Reliability percentages live under ``pct``.
    """
    care_path = resolve_model_care_json(model_care_path)
    scenarios = load_model_care_scenarios(care_path)
    cases = scan_model_care_cases(repo_root, keywords=keywords)
    rel_cells = scan_reliability_cells(repo_root, keywords=keywords)
    for model_id, cov in coverage.items():
        care = score_model_care(scenarios, cases, model_id=model_id, keywords=keywords.get(model_id, ()))
        if model_id in single_mode:
            for axis in ("perf", "prec"):
                care[axis]["display"] = "single"
        else:
            for axis in ("perf", "prec"):
                care[axis]["display"] = "split"
        block = {"perf": care["perf"], "prec": care["prec"], "model_care": str(care_path or "")}
        if model_id in has_rel:
            block["rel"] = score_reliability(rel_cells.get(model_id, set()))
        cov["pct"] = block
    return coverage
