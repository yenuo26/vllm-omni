# Issue 关闭确认

Nightly HTML chapter that lists *theoretically closable* open `ci-failure` issues, based on the latest CUDA / NPU scheduled nightlies already loaded for the report. Advisory only — confirm before closing. Implemented in `scripts/issue_close_confirm.py` and wired from `nightly_local_log_report.py::emit_report_html` immediately after Issue Alert.

## Source

`GET /repos/vllm-project/vllm-omni/issues?state=open&labels=ci-failure&per_page=100&page=…` (PRs excluded). `bug` is not required. Uses `GITHUB_TOKEN` / `GH_TOKEN` the same way as Issue Alert and the Outstanding DI card; unauthenticated REST is allowed and the chapter degrades with a note if listing fails.

## Conservative close-candidate rule

An issue is listed only with **positive evidence** that the previously failing test/job is now passing on every relevant pipeline:

1. **Identity.** Distinctive job name in the title/body (emoji aliases stripped; `·` / `-` / `/` treated as spaces; at least 3 tokens and length ≥18, or a near-verbatim job title) and/or an exact pytest node id (`[path/]foo.py::test_…`). Missing or weak identity → not listed.
2. **Scope.** NPU / CUDA / H100 / H200 / H800 / L4 mentions (and labels) decide which sides must be green. If the issue is silent, use the pipelines whose jobs actually matched. Still unknown → not listed. Do not close an NPU-only issue because CUDA is green, or vice versa. If both sides are implicated, both must be green.
3. **Job result.** Required sides must have loaded nightly jobs. Each identified job: Buildkite `state == passed`, pytest summary fully `ok` (no failed/error nodes, no `log_error`, no skipped/unknown kind).
4. **Test nodes.** Any extracted exact node must *not* appear in the current failed/error node set on that side. `parse_pytest_log` does not populate `passed_nodes`, so job-level full green plus “node absent from failures” is the positive evidence used here.

Missing data, skipped/not-run, weak match, or a CUDA/NPU mismatch must **not** inflate the closable table.

## Output

HTML table columns: issue (link + title), scope, tracked test/job, latest CUDA, latest NPU, why closable. Disable with `--no-issue-close-confirm`.
