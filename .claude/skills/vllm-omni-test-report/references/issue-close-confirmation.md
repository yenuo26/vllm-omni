# Issue close confirmation

Nightly HTML chapter that scans every still-open `ci-failure` issue and confirms it against the corresponding CUDA / NPU scheduled nightlies already loaded for the report. Advisory only — confirm before closing. Implemented in `scripts/issue_close_confirm.py` and wired from `nightly_local_log_report.py::emit_report_html` immediately after Issue Alert, as a `bk-section panel nightly-root` card.

## Source

`GET /repos/vllm-project/vllm-omni/issues?state=open&labels=ci-failure&per_page=100&page=…` (PRs excluded). `bug` is not required. Uses `GITHUB_TOKEN` / `GH_TOKEN` the same way as Issue Alert and the Outstanding DI card; unauthenticated REST is allowed and the chapter degrades with a note if listing fails.

## Conservative confirmation rule

Each issue is classified as **theoretically closable**, **still failing**, or **insufficient evidence**:

1. **Identity.** Distinctive job name in the title/body (emoji aliases stripped; `·` / `-` / `/` treated as spaces; at least 3 tokens and length ≥18, or a near-verbatim job title) and/or an exact pytest node id (`[path/]foo.py::test_…`). Missing or weak identity → insufficient evidence.
2. **Scope.** NPU / CUDA / H100 / H200 / H800 / L4 mentions (and labels) decide which sides must be green. If the issue is silent, use the pipelines whose jobs actually matched. Still unknown → insufficient evidence. Do not mark an NPU-only issue closable because CUDA is green, or vice versa. If both sides are implicated, both must be green.
3. **Job result (latest nightly).** Required sides must have loaded nightly jobs. Theoretically closable: each identified job has Buildkite `state == passed` and a fully `ok` pytest summary (no failed/error nodes, no `log_error`, no skipped/unknown kind). A clearly red job → still failing. No clear pass/fail evidence → insufficient evidence.
4. **Test nodes.** Any extracted exact node must *not* appear in the current failed/error node set on that side. `parse_pytest_log` does not populate `passed_nodes`, so job-level full green plus “node absent from failures” is the positive evidence used here.

## Consecutive clean runs

The table column **Consecutive clean runs** counts how many successive scheduled nightlies (newest first, up to 14), starting from this report’s CUDA / NPU builds, have not reproduced the failure.

- The latest entry is the nightly already loaded for the report.
- Older entries use Buildkite job *state* only (no extra log download), via `BUILDKITE_TOKEN` / `BUILDKITE_API_TOKEN`.
- A build counts as clean only when every identified job on that side has `state == passed` (or `finished`). A missing, skipped, or failed job breaks the streak.
- For issues that involve both CUDA and NPU, the number shown is the **minimum** streak across required sides.
- Still-failing issues show 0. Insufficient-evidence issues show n/a.

## Output

The table lists every scanned issue (sorted theoretically closable, then still failing, then insufficient evidence). Columns: issue (link + title), scope, tracked test/job, latest CUDA, latest NPU, consecutive clean runs, confirmation (status + reason). Disable with `--no-issue-close-confirm`.
