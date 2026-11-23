# Nightly Issue section

The nightly HTML report body is three top-level parts, in order: **Issue**, **Buildkite**, **Local test**. Daily focus stays above them as the existing failure-count dashboard.

The Issue section replaces both the old **Issue Alert** chapter and the unmerged **Issue close confirmation** chapter (the conservative evaluator from that work lives in `scripts/issue_close_confirm.py`; the old chapter is not rendered). Implementation: `scripts/issue_section.py`, wired from `nightly_local_log_report.py::emit_report_html`.

## Inputs

- Today's CUDA / NPU scheduled nightlies, CUDA weekly builds that fall in the report window, and local `nightly_jobs` failures.
- Already-open issues. Close / still-failing confirmation uses `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=ci-failure` (PRs excluded). Node-level duplicate matching uses the same read-only issue search as the old Issue Alert classifier.

Nothing is invented. A missing token, a failed fetch, a skipped or not-run job, a weak job name, or a CUDA vs NPU mismatch does not add a closable row.

## Subsections

| Title | Contents | Confirm control |
| --- | --- | --- |
| Summary | Short prose: how many items are in the three buckets, plus what was withheld and why. | — |
| 待新增 | Failures that need a new issue: search verdict `new` or `recurred` (a closed issue matched; the card names it). Each item has the issue-to-file text, the log analysis, and build URL(s). | **提单确认** |
| 待关闭 | Open `ci-failure` issues whose previously failing job is fully green on every relevant pipeline, and whose tracked test nodes are absent from the current failure set. | **关闭确认** |
| 待更新 | Open issues that are still failing, or that a strong duplicate match / distinctive job-name failure still ties to today's run, and are not ready to close. | **更新评论** |

Local-environment signatures (HF cache-lock permission errors) are not listed under 待新增. Groups whose search did not return a confident match are not listed as new issues. A closable nightly reading is dropped from 待关闭 when this report still has a strong match to a live failure.

## Confirm buttons

The buttons are operator confirmations. The static archived page does not call the GitHub API, so it cannot file, close, or comment by itself.

- **提单确认** reuses the failure-table prefilled bug-report URL (`/issues/new?template=400-bug-report.yml&...`). The click opens that form. GitHub creates the issue only after the operator submits it.
- **关闭确认** opens the existing issue and copies a ready-to-paste close note when the clipboard is available. The note is also visible on the card. Closing still happens on GitHub.
- **更新评论** opens the existing issue and copies a ready-to-paste comment the same way. The page does not POST the comment.

`--no-issue-alert` skips the GitHub issue search used for node-level 待新增 and search-based tracked updates. The Issue section still renders, and open `ci-failure` close/update confirmation still runs.
