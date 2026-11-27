# Nightly Issue section

The nightly HTML report body is three top-level parts, in order: **Issue**, **Buildkite**, **Local test**. Daily focus stays above them as the existing failure-count dashboard.

The Issue section replaces both the old **Issue Alert** chapter and the unmerged **Issue close confirmation** chapter (the conservative evaluator from that work lives in `scripts/issue_close_confirm.py`; the old chapter is not rendered). Implementation: `scripts/issue_section.py`, wired from `nightly_local_log_report.py::emit_report_html`.

## Inputs

- Today's CUDA / NPU scheduled nightlies, CUDA weekly builds that fall in the report window, and local `nightly_jobs` failures.
- Already-open issues. Close / still-failing confirmation uses `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=ci-failure` (PRs excluded). Node-level duplicate matching uses the same read-only issue search as the old Issue Alert classifier.
- Open `ci-failure` issues tagged `[B200]` (title, body, or labels) are confirmed against the latest **Scheduled Release B200** build on the release branch named in the issue (`release/vX.Y.Z`, `rc` / `.post` included), or `main` if none is named. The first line of the build message must be `Scheduled Release B200 build` (case-insensitive, ordinary whitespace allowed). `source` is ignored, so a `source=ui` release-branch B200 build still counts. Daily scheduled nightlies are not used as B200 evidence. A dual-tagged `[B200][H100]` (or H200 / H800 / L4) issue also requires the daily CUDA nightly. NPU nightlies are never B200 evidence.

Nothing is invented. A missing token, a failed fetch, a skipped or not-run job, a weak job name, or a CUDA vs NPU mismatch does not add a closable row. A `[B200]` issue with no loaded Scheduled Release B200 build is not closable and is not judged from today's nightly instead.

## Subsections

| Title | Contents | Confirm control |
| --- | --- | --- |
| Summary | Short prose above the cards: how many items are in the three buckets, plus what was withheld and why. | — |
| 待新增 | Collapsed card. The table has one to-be-filed item per row (proposed title, analysis, build URLs). | **开单确认** |
| 待关闭 | Collapsed card. One closable issue per row. | **关闭确认** |
| 待更新 | Collapsed card. One already-open issue per row, and only when its test or job still failed in this run. The comment is today's failure. | **更新评论** |

Empty cards still show the title. Opening one shows the table headers and an empty row, with no confirm button.

**一键操作** is outside the cards. It does not run on load. One click files every 待新增 row, comments and closes every 待关闭 row, and comments on every 待更新 row.

Local-environment signatures (HF cache-lock permission errors) are not listed under 待新增. Groups whose search did not return a confident match are not listed as new issues. A closable nightly reading is dropped from 待关闭 when this report still has a strong match to a live failure.

## Confirm buttons

Row buttons do not call the GitHub API:

- **开单确认** opens that row's prefilled bug-report URL (`/issues/new?template=400-bug-report.yml&...`). GitHub creates the issue only after the operator submits the form.
- **关闭确认** opens the existing issue and copies a ready-to-paste close note. It does not close the issue.
- **更新评论** opens the existing issue and copies a ready-to-paste comment. It does not post the comment.

## 一键操作

The archived page embeds the action payload (`id="issue-bulk-payload"`). When the report is written to a file, the same JSON is saved as `<report>.issue-actions.json`. Neither contains a token.

The click POSTs that payload to `http://127.0.0.1:8931/apply`. Start the executor first:

```bash
export GITHUB_TOKEN=...   # or GH_TOKEN; do not pass the value as an argument
python3 scripts/apply_nightly_issue_actions.py --serve
```

`--serve` binds `127.0.0.1` only and reads the token from the environment. It creates each 待新增 issue, posts the prepared comment and closes each 待关闭 issue, and posts each 待更新 comment. Completed ids are recorded so a second click skips them.

If the executor is not running, the button says GitHub was not changed and shows:

```bash
python3 scripts/apply_nightly_issue_actions.py --payload <report>.issue-actions.json --apply
```

That command performs the same three writes. Without `--apply` it only prints the plan.

`--no-issue-alert` skips the GitHub issue search used for node-level 待新增 and search-based tracked updates. The Issue section still renders, and open `ci-failure` close/update confirmation still runs.
