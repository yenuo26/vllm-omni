#!/usr/bin/env python3
"""Apply a nightly Issue-section payload to GitHub.

The archived HTML does not contain a token. This process reads
``GITHUB_TOKEN`` or ``GH_TOKEN`` from the environment only.

  # While the report is open, then click 一键操作 in the page:
  python3 apply_nightly_issue_actions.py --serve

  # Or run the whole set once, without the page:
  python3 apply_nightly_issue_actions.py --payload report.issue-actions.json --apply

Without ``--apply``, the command prints the plan and does not write.
``--serve`` writes only when the page POSTs /apply (the 一键操作 click).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ALLOWED_REPO = "vllm-project/vllm-omni"
API = "https://api.github.com"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8931
_MAX_BODY = 2_000_000
_LOCK = threading.Lock()


class NoTokenError(RuntimeError):
    pass


class ApiError(RuntimeError):
    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"GitHub API {code}: {detail}")
        self.code = code
        self.detail = detail


def github_token() -> str:
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if not token:
        raise NoTokenError(
            "Set GITHUB_TOKEN or GH_TOKEN in the environment. "
            "The token is not read from the HTML or the payload."
        )
    return token


def _redact(text: str, token: str) -> str:
    if token and text and token in text:
        return text.replace(token, "[redacted]")
    return text


def _load_results(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    actions = data.get("actions") if isinstance(data, dict) else None
    return actions if isinstance(actions, dict) else {}


def _save_results(path: Path | None, actions: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"actions": actions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _api(
    method: str,
    url: str,
    token: str,
    body: dict[str, Any] | None,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "vllm-omni-nightly-report",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with opener(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err = _redact(exc.read().decode("utf-8", "replace")[:300], token)
        raise ApiError(exc.code, err) from None
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _comment(repo: str, number: int, body: str, token: str, opener: Callable[..., Any]) -> None:
    _api(
        "POST",
        f"{API}/repos/{repo}/issues/{number}/comments",
        token,
        {"body": body},
        opener,
    )


def _create(repo: str, action: dict[str, Any], token: str, opener: Callable[..., Any]) -> str:
    payload: dict[str, Any] = {
        "title": action.get("title") or "Nightly failure",
        "body": action.get("body") or "",
    }
    labels = [str(label) for label in (action.get("labels") or []) if str(label).strip()]
    if labels:
        payload["labels"] = labels
    try:
        created = _api("POST", f"{API}/repos/{repo}/issues", token, payload, opener)
    except ApiError as exc:
        if exc.code != 422 or not labels:
            raise
        payload.pop("labels", None)
        created = _api("POST", f"{API}/repos/{repo}/issues", token, payload, opener)
    return str(created.get("html_url") or created.get("number") or "")


def _close(repo: str, number: int, token: str, opener: Callable[..., Any]) -> None:
    _api(
        "PATCH",
        f"{API}/repos/{repo}/issues/{number}",
        token,
        {"state": "closed", "state_reason": "completed"},
        opener,
    )


def _check_payload(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    repo = str(payload.get("repo") or ALLOWED_REPO)
    if repo != ALLOWED_REPO:
        raise ValueError(f"refusing repo {repo}")
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError("payload.actions must be a list")
    return repo


def apply_payload(
    payload: dict[str, Any],
    *,
    write: bool,
    token: str | None = None,
    opener: Callable[..., Any] | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Create, close, and comment. ``write=False`` only describes the plan."""
    repo = _check_payload(payload)
    actions = list(payload.get("actions") or [])
    previous = _load_results(results_path)
    opener = opener or urllib.request.urlopen
    counts = {"created": 0, "closed": 0, "commented": 0, "skipped": 0, "failed": 0}
    results: list[dict[str, Any]] = []
    auth = ""
    if write:
        auth = token if token is not None else github_token()

    for action in actions:
        if not isinstance(action, dict):
            counts["failed"] += 1
            results.append({"id": "", "op": "", "status": "failed", "detail": "action is not an object"})
            continue
        action_id = str(action.get("id") or "")
        op = str(action.get("op") or "")
        prev = previous.get(action_id) if action_id else None
        if isinstance(prev, dict) and prev.get("status") == "ok":
            counts["skipped"] += 1
            results.append({"id": action_id, "op": op, "status": "skipped", "detail": "already applied"})
            continue
        if not write:
            results.append({"id": action_id, "op": op, "status": "planned", "detail": ""})
            continue
        prior = prev if isinstance(prev, dict) else {}
        try:
            detail = _apply_one(repo, action, auth, opener, prior)
        except (ApiError, ValueError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            message = _redact(str(exc), auth)
            counts["failed"] += 1
            record = {"id": action_id, "op": op, "status": "failed", "detail": message}
            if prior.get("commented"):
                record["commented"] = True
            results.append(record)
            if action_id:
                previous[action_id] = record
                _save_results(results_path, previous)
            continue
        counts[detail[0]] += 1
        record = {"id": action_id, "op": op, "status": "ok", "detail": detail[1]}
        results.append(record)
        if action_id:
            previous[action_id] = record
            _save_results(results_path, previous)

    return {"ok": counts["failed"] == 0, **counts, "results": results}


def _apply_one(
    repo: str,
    action: dict[str, Any],
    token: str,
    opener: Callable[..., Any],
    prev: dict[str, Any],
) -> tuple[str, str]:
    op = str(action.get("op") or "")
    if op == "create":
        url = _create(repo, action, token, opener)
        return "created", url
    number = action.get("issue_number")
    if not isinstance(number, int):
        raise ValueError(f"{op} requires an integer issue_number")
    comment = str(action.get("comment") or "")
    if op == "comment":
        if comment and not prev.get("commented"):
            _comment(repo, number, comment, token, opener)
        return "commented", f"https://github.com/{repo}/issues/{number}"
    if op == "close":
        if comment and not prev.get("commented"):
            _comment(repo, number, comment, token, opener)
            prev["commented"] = True
        _close(repo, number, token, opener)
        return "closed", f"https://github.com/{repo}/issues/{number}"
    raise ValueError(f"unknown op {op}")


def _print_plan(result: dict[str, Any]) -> None:
    for row in result["results"]:
        op = row.get("op") or "?"
        ident = row.get("id") or "?"
        status = row.get("status") or ""
        detail = row.get("detail") or ""
        print(f"{status}\t{op}\t{ident}\t{detail}")
    print(
        "created={created} closed={closed} commented={commented} skipped={skipped} failed={failed}".format(
            **{key: result.get(key, 0) for key in ("created", "closed", "commented", "skipped", "failed")}
        )
    )


class _ApplyHandler(BaseHTTPRequestHandler):
    results_path: Path | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?", 1)[0]))

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self._json(404, {"ok": False, "error": "not found"})
            return
        self._json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/apply":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY:
            self._json(400, {"ok": False, "error": "payload size is not allowed"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._json(400, {"ok": False, "error": "payload is not JSON"})
            return
        if isinstance(payload, dict):
            payload.pop("token", None)
            payload.pop("github_token", None)
            payload.pop("authorization", None)
        try:
            with _LOCK:
                result = apply_payload(
                    payload,
                    write=True,
                    results_path=self.results_path,
                )
        except NoTokenError as exc:
            self._json(401, {"ok": False, "error": str(exc)})
            return
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200 if result.get("ok") else 502, result)


def serve(port: int, results_path: Path | None) -> None:
    _ApplyHandler.results_path = results_path
    httpd = ThreadingHTTPServer((LISTEN_HOST, port), _ApplyHandler)
    print(f"Listening on http://{LISTEN_HOST}:{port}/apply", file=sys.stderr)
    print("Token is taken from GITHUB_TOKEN or GH_TOKEN. Click 一键操作 in the report.", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply nightly Issue-section actions to GitHub.")
    parser.add_argument("--payload", type=Path, help="issue-actions JSON written next to the report")
    parser.add_argument("--apply", action="store_true", help="Perform the GitHub writes. Omit to print the plan only.")
    parser.add_argument("--serve", action="store_true", help="Listen on 127.0.0.1 for the report's 一键操作 button.")
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    parser.add_argument("--results", type=Path, default=None, help="JSON file of completed action ids.")
    args = parser.parse_args(argv)

    if args.serve:
        results = args.results or Path.cwd() / "nightly-issue-actions.results.json"
        serve(args.port, results)
        return 0
    if args.payload is None:
        print("Pass --payload, or --serve for the in-page 一键操作 button.", file=sys.stderr)
        return 2
    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read payload: {exc}", file=sys.stderr)
        return 2
    results = args.results or args.payload.with_name(args.payload.name + ".results.json")
    try:
        outcome = apply_payload(payload, write=args.apply, results_path=results if args.apply else None)
    except NoTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _print_plan(outcome)
    if not args.apply:
        print("Dry run only. Re-run with --apply to file, close, and comment.", file=sys.stderr)
        return 0
    return 0 if outcome.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
