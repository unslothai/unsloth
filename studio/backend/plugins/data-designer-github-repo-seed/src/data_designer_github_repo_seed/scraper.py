# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-repo GitHub scraper for the Studio seed plugin.

Drives the GraphQL-based scraper in `scraper_impl/` per repo. Each repo is
scraped with a trial_limits cap so we stop at `limit` items per resource.
After scraping, we read the per-resource JSONL shards and flatten them into
a single unified JSONL with stable columns (`item_type`, `repo`, `number`,
`title`, `body`, ...).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Defer scraper_impl imports until `scrape()` runs with a resolved token.
_IMPL_DIR = Path(__file__).parent / "scraper_impl"


def _ensure_impl_on_path() -> None:
    if str(_IMPL_DIR) not in sys.path:
        sys.path.insert(0, str(_IMPL_DIR))


def _load_impl():
    _ensure_impl_on_path()
    import importlib

    gh_client = importlib.import_module("gh_client")  # type: ignore
    scraper_mod = importlib.import_module("scraper")  # type: ignore
    return gh_client.GitHubClient, scraper_mod.RepoScraper


@dataclass
class ScrapeConfig:
    repos: list[str]
    token: str
    item_types: list[str]
    limit: int
    include_comments: bool
    max_comments_per_item: int


def _resolve_token(token: str) -> str:
    tok = token or os.environ.get("GH_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        raise ValueError(
            "GitHub token is required. Set it in the recipe config or the GH_TOKEN / GITHUB_TOKEN env var."
        )
    return tok


def _read_jsonl(path: Path, max_rows: int | None = None):
    if not path.exists():
        return
    with path.open(encoding = "utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if max_rows is not None and i >= max_rows:
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _flatten_issue_row(r: dict, repo: str, include_comments: bool, max_c: int) -> dict:
    labels = [
        l.get("name")
        for l in (r.get("labels", {}) or {}).get("nodes", [])
        if l.get("name")
    ]
    comments_nodes = (r.get("comments") or {}).get("nodes") or []
    comments_text = ""
    if include_comments and comments_nodes:
        kept = comments_nodes[:max_c]
        comments_text = "\n\n".join(
            f"[{(c.get('author') or {}).get('login', '?')}]: {c.get('body') or ''}"
            for c in kept
        )
    return {
        "item_type": "issue",
        "repo": repo,
        "number": r.get("number"),
        "title": r.get("title") or "",
        "body": r.get("body") or "",
        "state": r.get("state") or "",
        "author": (r.get("author") or {}).get("login", ""),
        "created_at": r.get("createdAt") or "",
        "closed_at": r.get("closedAt") or "",
        "url": r.get("url") or r.get("permalink") or "",
        "labels": labels,
        "comments": comments_text,
    }


def _flatten_pr_row(r: dict, repo: str, include_comments: bool, max_c: int) -> dict:
    labels = [
        l.get("name")
        for l in (r.get("labels", {}) or {}).get("nodes", [])
        if l.get("name")
    ]
    comments_nodes = (r.get("comments") or {}).get("nodes") or []
    comments_text = ""
    if include_comments and comments_nodes:
        kept = comments_nodes[:max_c]
        comments_text = "\n\n".join(
            f"[{(c.get('author') or {}).get('login', '?')}]: {c.get('body') or ''}"
            for c in kept
        )
    return {
        "item_type": "pull",
        "repo": repo,
        "number": r.get("number"),
        "title": r.get("title") or "",
        "body": r.get("body") or "",
        "state": r.get("state") or "",
        "author": (r.get("author") or {}).get("login", ""),
        "created_at": r.get("createdAt") or "",
        "closed_at": r.get("closedAt") or "",
        "url": r.get("url") or r.get("permalink") or "",
        "labels": labels,
        "comments": comments_text,
    }


def _flatten_commit_row(r: dict, repo: str) -> dict:
    msg = r.get("messageHeadline") or r.get("message") or ""
    body = r.get("messageBody") or r.get("message") or msg
    author = r.get("author") or {}
    return {
        "item_type": "commit",
        "repo": repo,
        "number": r.get("oid") or r.get("sha") or "",
        "title": msg,
        "body": body,
        "state": "",
        "author": (author.get("user") or {}).get("login") or author.get("name", ""),
        "created_at": (author.get("date") or r.get("committedDate") or ""),
        "closed_at": "",
        "url": r.get("url") or "",
        "labels": [],
        "comments": "",
    }


def scrape(cfg: ScrapeConfig, base_dir: Path):
    token = _resolve_token(cfg.token)
    GitHubClient, RepoScraper = _load_impl()
    client = GitHubClient(token = token)
    base_dir.mkdir(parents = True, exist_ok = True)

    # Per-resource trial limits. limit <= 0 means "all": use a very large cap.
    effective_limit = cfg.limit if cfg.limit and cfg.limit > 0 else 1_000_000
    trial_limits: dict[str, int] = {}
    if "issues" in cfg.item_types:
        trial_limits["issues"] = effective_limit
    if "pulls" in cfg.item_types:
        trial_limits["pull_requests"] = effective_limit
    if "commits" in cfg.item_types:
        trial_limits["commits"] = effective_limit

    all_rows: list[dict] = []
    for repo in cfg.repos:
        owner, name = repo.split("/", 1)
        scraper = RepoScraper(
            owner = owner,
            name = name,
            base_dir = base_dir,
            client = client,
            trial_limits = trial_limits,
            light = True,
        )
        try:
            repo_meta = scraper.scrape_repo_meta()
            if "issues" in cfg.item_types:
                scraper.scrape_issues()
            if "pulls" in cfg.item_types:
                scraper.scrape_prs()
            if "commits" in cfg.item_types:
                default_ref = repo_meta.get("defaultBranchRef") or {}
                default_branch = (
                    default_ref.get("name") if isinstance(default_ref, dict) else None
                )
                branch = (
                    f"refs/heads/{default_branch}"
                    if default_branch
                    else "refs/heads/main"
                )
                scraper.scrape_commits(branch = branch)
        finally:
            scraper.close()

        read_cap = cfg.limit if cfg.limit and cfg.limit > 0 else None
        repo_dir = base_dir / f"{owner}__{name}"
        if "issues" in cfg.item_types:
            for row in _read_jsonl(repo_dir / "issues.jsonl", read_cap):
                all_rows.append(
                    _flatten_issue_row(
                        row, repo, cfg.include_comments, cfg.max_comments_per_item
                    )
                )
        if "pulls" in cfg.item_types:
            for row in _read_jsonl(repo_dir / "pull_requests.jsonl", read_cap):
                all_rows.append(
                    _flatten_pr_row(
                        row, repo, cfg.include_comments, cfg.max_comments_per_item
                    )
                )
        if "commits" in cfg.item_types:
            for row in _read_jsonl(repo_dir / "commits.jsonl", read_cap):
                all_rows.append(_flatten_commit_row(row, repo))

    return all_rows


def materialize_to_jsonl(cfg: ScrapeConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents = True, exist_ok = True)
    tag = "-".join(r.replace("/", "__") for r in cfg.repos)[:120]
    kinds = "-".join(cfg.item_types)
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:12]}"
    fname = f"github_{tag}__{kinds}__{cfg.limit}_{run_id}.jsonl"
    out = out_dir / fname
    rows = scrape(cfg, out_dir / "raw-runs" / run_id)
    with out.open("w", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")
    return out
