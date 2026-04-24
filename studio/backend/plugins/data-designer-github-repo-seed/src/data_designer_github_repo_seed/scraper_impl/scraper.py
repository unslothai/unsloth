# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Main scraper orchestration. Collects issues, PRs, discussions, commits, releases, etc.

Resumable via state file. Writes JSONL shards under data/{repo}/{resource}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow running as a module or script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gh_client import GitHubClient
from state_store import JsonlWriter, StateStore
import queries as Q

log = logging.getLogger("scraper")


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RepoScraper:
    def __init__(
        self,
        owner: str,
        name: str,
        base_dir: Path,
        client: GitHubClient,
        trial_limits: Optional[Dict[str, int]] = None,
        light: bool = False,
    ):
        self.owner = owner
        self.name = name
        self.base_dir = base_dir
        self.client = client
        self.trial_limits = trial_limits or {}
        # When light=True, use trimmed GraphQL queries (no reviewThreads,
        # reviews, commits, timelineItems, files) so PR pages can be much
        # larger without blowing GitHub's node-count ceiling.
        self.light = light
        self.repo_dir = base_dir / f"{owner}__{name}"
        self.repo_dir.mkdir(parents = True, exist_ok = True)
        self.state = StateStore(base_dir / "state" / f"{owner}__{name}.json")

        # Writers
        self.writers: Dict[str, JsonlWriter] = {}
        for key in (
            "issues",
            "pull_requests",
            "discussions",
            "commits",
            "releases",
            "labels",
            "milestones",
            "pr_extra_comments",
            "pr_extra_timeline",
            "pr_extra_reviews",
            "issue_extra_comments",
            "issue_extra_timeline",
            "discussion_extra_comments",
            "discussion_extra_replies",
            "repo_meta",
        ):
            self.writers[key] = JsonlWriter(self.repo_dir / f"{key}.jsonl")

    # ----- helpers -----
    def _trial_stop(self, key: str, counter: int) -> bool:
        lim = self.trial_limits.get(key)
        if lim is None:
            return False
        return counter >= lim

    def _log_rate(self, where: str, data: Dict[str, Any]) -> None:
        rl = (
            data.get("data", {}).get("rateLimit")
            if isinstance(data.get("data"), dict)
            else None
        )
        if rl:
            log.debug(
                "[%s] rate cost=%s remaining=%s resetAt=%s",
                where,
                rl.get("cost"),
                rl.get("remaining"),
                rl.get("resetAt"),
            )

    # ----- repo meta -----
    def scrape_repo_meta(self) -> Dict[str, Any]:
        data = self.client.graphql(
            Q.REPO_META_QUERY, {"owner": self.owner, "name": self.name}
        )
        self._log_rate("repo_meta", data)
        repo = data.get("data", {}).get("repository") or {}
        repo["_fetchedAt"] = ts()
        self.writers["repo_meta"].write(repo)
        return repo

    # ----- issues -----
    def scrape_issues(self) -> int:
        key = "issues"
        cursor = self.state.get(f"{key}_cursor")
        done = self.state.get(f"{key}_done", False)
        if done:
            log.info("%s/%s issues already complete", self.owner, self.name)
            return 0
        total_new = 0
        page = 0
        # Light query skips heavy nested fields; safe at 50 per page.
        # Clamp by trial_limit so e.g. limit=1 asks GitHub for first:1
        # instead of fetching a full 50-item page and discarding 49.
        page_cap = 50 if self.light else 15
        trial_cap = self.trial_limits.get(key)
        per_page = min(page_cap, trial_cap) if trial_cap and trial_cap > 0 else page_cap
        while True:
            page += 1
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "first": per_page,
                "after": cursor,
            }
            query = Q.ISSUES_PAGE_QUERY_LIGHT if self.light else Q.ISSUES_PAGE_QUERY
            data = self.client.graphql(query, vars_)
            self._log_rate("issues", data)
            repo = (data.get("data") or {}).get("repository") or {}
            issues = repo.get("issues") or {}
            nodes = issues.get("nodes") or []
            for it in nodes:
                it["_owner"] = self.owner
                it["_repo"] = self.name
                it["_fetchedAt"] = ts()
                if not self.light:
                    if it.get("comments", {}).get("pageInfo", {}).get("hasNextPage"):
                        self._paginate_issue_comments(
                            it["number"], it["comments"]["pageInfo"]["endCursor"]
                        )
                    if (
                        it.get("timelineItems", {})
                        .get("pageInfo", {})
                        .get("hasNextPage")
                    ):
                        self._paginate_issue_timeline(
                            it["number"],
                            it["timelineItems"]["pageInfo"]["endCursor"],
                        )
                if self.writers[key].write(it):
                    total_new += 1
            info = issues.get("pageInfo") or {}
            cursor = info.get("endCursor")
            self.state.set(f"{key}_cursor", cursor)
            log.info(
                "[%s/%s] issues page %d (+%d) cursor=%s remaining=%s",
                self.owner,
                self.name,
                page,
                len(nodes),
                str(cursor)[:20],
                self.client.graphql_remaining,
            )
            if self._trial_stop(key, total_new):
                log.info("Trial limit reached for issues (%d)", total_new)
                return total_new
            if not info.get("hasNextPage"):
                self.state.set(f"{key}_done", True)
                break
        return total_new

    def _paginate_issue_comments(self, number: int, after: str) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.ISSUE_COMMENTS_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "issueOrPullRequest"
            ) or {}
            comments = item.get("comments") or {}
            for c in comments.get("nodes") or []:
                c["_owner"] = self.owner
                c["_repo"] = self.name
                c["_issueNumber"] = number
                self.writers["issue_extra_comments"].write(c)
            info = comments.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_issue_timeline(self, number: int, after: str) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.ISSUE_TIMELINE_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}
            tl = item.get("timelineItems") or {}
            for ev in tl.get("nodes") or []:
                ev["_owner"] = self.owner
                ev["_repo"] = self.name
                ev["_issueNumber"] = number
                self.writers["issue_extra_timeline"].write(ev)
            info = tl.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    # ----- PRs -----
    def scrape_prs(self) -> int:
        key = "pull_requests"
        cursor = self.state.get(f"{key}_cursor")
        done = self.state.get(f"{key}_done", False)
        if done:
            log.info("%s/%s PRs already complete", self.owner, self.name)
            return 0
        total_new = 0
        page = 0
        # Heavy nested PR query is capped at 3 per page (GitHub node-count
        # ceiling); light query skips reviewThreads/reviews/commits/etc and
        # can safely go to 25 per page. Clamp by trial_limit for small
        # previews so limit=1 does not fetch a whole 25-item page.
        page_cap = 25 if self.light else 3
        trial_cap = self.trial_limits.get(key)
        per_page = min(page_cap, trial_cap) if trial_cap and trial_cap > 0 else page_cap
        while True:
            page += 1
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "first": per_page,
                "after": cursor,
            }
            query = Q.PRS_PAGE_QUERY_LIGHT if self.light else Q.PRS_PAGE_QUERY
            data = self.client.graphql(query, vars_)
            self._log_rate("prs", data)
            repo = (data.get("data") or {}).get("repository") or {}
            prs = repo.get("pullRequests") or {}
            nodes = prs.get("nodes") or []
            for pr in nodes:
                pr["_owner"] = self.owner
                pr["_repo"] = self.name
                pr["_fetchedAt"] = ts()
                num = pr["number"]
                if not self.light:
                    if pr.get("comments", {}).get("pageInfo", {}).get("hasNextPage"):
                        self._paginate_pr_comments(
                            num, pr["comments"]["pageInfo"]["endCursor"]
                        )
                    if (
                        pr.get("timelineItems", {})
                        .get("pageInfo", {})
                        .get("hasNextPage")
                    ):
                        self._paginate_pr_timeline(
                            num, pr["timelineItems"]["pageInfo"]["endCursor"]
                        )
                    if pr.get("commits", {}).get("pageInfo", {}).get("hasNextPage"):
                        self._paginate_pr_commits(
                            num, pr["commits"]["pageInfo"]["endCursor"]
                        )
                    if pr.get("files", {}).get("pageInfo", {}).get("hasNextPage"):
                        self._paginate_pr_files(
                            num, pr["files"]["pageInfo"]["endCursor"]
                        )
                    if (
                        pr.get("reviewThreads", {})
                        .get("pageInfo", {})
                        .get("hasNextPage")
                    ):
                        self._paginate_pr_review_threads(
                            num, pr["reviewThreads"]["pageInfo"]["endCursor"]
                        )
                if self.writers[key].write(pr):
                    total_new += 1
            info = prs.get("pageInfo") or {}
            cursor = info.get("endCursor")
            self.state.set(f"{key}_cursor", cursor)
            log.info(
                "[%s/%s] PRs page %d (+%d) cursor=%s remaining=%s",
                self.owner,
                self.name,
                page,
                len(nodes),
                str(cursor)[:20],
                self.client.graphql_remaining,
            )
            if self._trial_stop(key, total_new):
                log.info("Trial limit reached for PRs (%d)", total_new)
                return total_new
            if not info.get("hasNextPage"):
                self.state.set(f"{key}_done", True)
                break
        return total_new

    def _paginate_pr_comments(self, number: int, after: str) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.ISSUE_COMMENTS_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "issueOrPullRequest"
            ) or {}
            comments = item.get("comments") or {}
            for c in comments.get("nodes") or []:
                c["_owner"] = self.owner
                c["_repo"] = self.name
                c["_prNumber"] = number
                self.writers["pr_extra_comments"].write(c)
            info = comments.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_pr_timeline(self, number: int, after: str) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.PR_TIMELINE_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "pullRequest"
            ) or {}
            tl = item.get("timelineItems") or {}
            for ev in tl.get("nodes") or []:
                ev["_owner"] = self.owner
                ev["_repo"] = self.name
                ev["_prNumber"] = number
                self.writers["pr_extra_timeline"].write(ev)
            info = tl.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_pr_commits(self, number: int, after: str) -> None:
        cur = after
        out_key = "pr_extra_commits"
        if out_key not in self.writers:
            self.writers[out_key] = JsonlWriter(self.repo_dir / f"{out_key}.jsonl")
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.PR_COMMITS_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "pullRequest"
            ) or {}
            cc = item.get("commits") or {}
            for c in cc.get("nodes") or []:
                c["_owner"] = self.owner
                c["_repo"] = self.name
                c["_prNumber"] = number
                self.writers[out_key].write(c)
            info = cc.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_pr_files(self, number: int, after: str) -> None:
        cur = after
        out_key = "pr_extra_files"
        if out_key not in self.writers:
            self.writers[out_key] = JsonlWriter(self.repo_dir / f"{out_key}.jsonl")
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.PR_FILES_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "pullRequest"
            ) or {}
            ff = item.get("files") or {}
            for f in ff.get("nodes") or []:
                f["_owner"] = self.owner
                f["_repo"] = self.name
                f["_prNumber"] = number
                # files don't have id, synthesize one
                f["_syntheticId"] = f"{self.owner}/{self.name}#{number}:{f.get('path')}"
                self.writers[out_key].write(f)
            info = ff.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_pr_review_threads(self, number: int, after: str) -> None:
        cur = after
        out_key = "pr_extra_review_threads"
        if out_key not in self.writers:
            self.writers[out_key] = JsonlWriter(self.repo_dir / f"{out_key}.jsonl")
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.PR_REVIEW_THREADS_QUERY, vars_)
            item = ((data.get("data") or {}).get("repository") or {}).get(
                "pullRequest"
            ) or {}
            rt = item.get("reviewThreads") or {}
            for th in rt.get("nodes") or []:
                th["_owner"] = self.owner
                th["_repo"] = self.name
                th["_prNumber"] = number
                self.writers[out_key].write(th)
            info = rt.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    # ----- Discussions -----
    def scrape_discussions(self) -> int:
        key = "discussions"
        cursor = self.state.get(f"{key}_cursor")
        done = self.state.get(f"{key}_done", False)
        if done:
            log.info("%s/%s discussions already complete", self.owner, self.name)
            return 0
        total_new = 0
        page = 0
        per_page = 15
        while True:
            page += 1
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "first": per_page,
                "after": cursor,
            }
            data = self.client.graphql(Q.DISCUSSIONS_PAGE_QUERY, vars_)
            self._log_rate("discussions", data)
            repo = (data.get("data") or {}).get("repository") or {}
            dd = repo.get("discussions") or {}
            nodes = dd.get("nodes") or []
            for d in nodes:
                d["_owner"] = self.owner
                d["_repo"] = self.name
                d["_fetchedAt"] = ts()
                num = d["number"]
                if d.get("comments", {}).get("pageInfo", {}).get("hasNextPage"):
                    self._paginate_discussion_comments(
                        num, d["comments"]["pageInfo"]["endCursor"]
                    )
                # paginate replies per comment if needed
                for c in d.get("comments", {}).get("nodes", []) or []:
                    if c.get("replies", {}).get("pageInfo", {}).get("hasNextPage"):
                        self._paginate_discussion_replies(
                            c["id"], c["replies"]["pageInfo"]["endCursor"], num
                        )
                if self.writers[key].write(d):
                    total_new += 1
            info = dd.get("pageInfo") or {}
            cursor = info.get("endCursor")
            self.state.set(f"{key}_cursor", cursor)
            log.info(
                "[%s/%s] discussions page %d (+%d) cursor=%s remaining=%s",
                self.owner,
                self.name,
                page,
                len(nodes),
                str(cursor)[:20],
                self.client.graphql_remaining,
            )
            if self._trial_stop(key, total_new):
                return total_new
            if not info.get("hasNextPage"):
                self.state.set(f"{key}_done", True)
                break
        return total_new

    def _paginate_discussion_comments(self, number: int, after: str) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "after": cur,
            }
            data = self.client.graphql(Q.DISCUSSION_COMMENTS_QUERY, vars_)
            disc = ((data.get("data") or {}).get("repository") or {}).get(
                "discussion"
            ) or {}
            cc = disc.get("comments") or {}
            for c in cc.get("nodes") or []:
                c["_owner"] = self.owner
                c["_repo"] = self.name
                c["_discussionNumber"] = number
                self.writers["discussion_extra_comments"].write(c)
            info = cc.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    def _paginate_discussion_replies(
        self, comment_id: str, after: str, disc_number: int
    ) -> None:
        cur = after
        while cur:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "commentId": comment_id,
                "after": cur,
            }
            data = self.client.graphql(Q.DISCUSSION_REPLIES_QUERY, vars_)
            node = (data.get("data") or {}).get("node") or {}
            replies = node.get("replies") or {}
            for r in replies.get("nodes") or []:
                r["_owner"] = self.owner
                r["_repo"] = self.name
                r["_discussionNumber"] = disc_number
                r["_commentId"] = comment_id
                self.writers["discussion_extra_replies"].write(r)
            info = replies.get("pageInfo") or {}
            cur = info.get("endCursor") if info.get("hasNextPage") else None

    # ----- Commits -----
    def scrape_commits(self, branch: str = "refs/heads/main") -> int:
        key = "commits"
        cursor = self.state.get(f"{key}_cursor")
        done = self.state.get(f"{key}_done", False)
        if done:
            return 0
        total_new = 0
        page = 0
        page_cap = 100
        trial_cap = self.trial_limits.get(key)
        per_page = min(page_cap, trial_cap) if trial_cap and trial_cap > 0 else page_cap
        while True:
            page += 1
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "first": per_page,
                "after": cursor,
                "branch": branch,
            }
            data = self.client.graphql(Q.COMMITS_PAGE_QUERY, vars_)
            self._log_rate("commits", data)
            ref = ((data.get("data") or {}).get("repository") or {}).get("ref") or {}
            tgt = ref.get("target") or {}
            hist = tgt.get("history") or {}
            nodes = hist.get("nodes") or []
            for c in nodes:
                c["_owner"] = self.owner
                c["_repo"] = self.name
                c["_fetchedAt"] = ts()
                if self.writers[key].write(c):
                    total_new += 1
            info = hist.get("pageInfo") or {}
            cursor = info.get("endCursor")
            self.state.set(f"{key}_cursor", cursor)
            log.info(
                "[%s/%s] commits page %d (+%d) remaining=%s",
                self.owner,
                self.name,
                page,
                len(nodes),
                self.client.graphql_remaining,
            )
            if self._trial_stop(key, total_new):
                return total_new
            if not info.get("hasNextPage"):
                self.state.set(f"{key}_done", True)
                break
        return total_new

    # ----- Releases/Labels/Milestones -----
    def scrape_releases(self) -> int:
        return self._scrape_simple("releases", Q.RELEASES_QUERY, "releases")

    def scrape_labels(self) -> int:
        return self._scrape_simple("labels", Q.LABELS_QUERY, "labels")

    def scrape_milestones(self) -> int:
        return self._scrape_simple("milestones", Q.MILESTONES_QUERY, "milestones")

    def _scrape_simple(self, key: str, query: str, field: str) -> int:
        cursor = self.state.get(f"{key}_cursor")
        done = self.state.get(f"{key}_done", False)
        if done:
            return 0
        total_new = 0
        while True:
            vars_ = {
                "owner": self.owner,
                "name": self.name,
                "first": 50,
                "after": cursor,
            }
            data = self.client.graphql(query, vars_)
            repo = (data.get("data") or {}).get("repository") or {}
            col = repo.get(field) or {}
            for it in col.get("nodes") or []:
                it["_owner"] = self.owner
                it["_repo"] = self.name
                it["_fetchedAt"] = ts()
                if self.writers[key].write(it):
                    total_new += 1
            info = col.get("pageInfo") or {}
            cursor = info.get("endCursor")
            self.state.set(f"{key}_cursor", cursor)
            if self._trial_stop(key, total_new):
                return total_new
            if not info.get("hasNextPage"):
                self.state.set(f"{key}_done", True)
                break
        log.info("[%s/%s] %s done +%d", self.owner, self.name, key, total_new)
        return total_new

    def close(self) -> None:
        for w in self.writers.values():
            try:
                w.close()
            except Exception:
                pass


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents = True, exist_ok = True)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode = "a", encoding = "utf-8"),
    ]
    logging.basicConfig(level = logging.INFO, format = fmt, handlers = handlers, force = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir", default = "/mnt/disks/unslothai/ubuntu/workspace_34/github_scraper"
    )
    ap.add_argument(
        "--repos", nargs = "+", default = ["unslothai/unsloth", "unslothai/unsloth-zoo"]
    )
    ap.add_argument("--trial", action = "store_true", help = "Small trial run")
    ap.add_argument(
        "--only",
        nargs = "+",
        default = None,
        help = "Only run these resource keys: issues,pulls,discussions,commits,releases,labels,milestones,meta",
    )
    ap.add_argument(
        "--hf-upload-interval",
        type = int,
        default = 900,
        help = "Seconds between HF uploads (0 to disable)",
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    data_dir = base / "data"
    data_dir.mkdir(parents = True, exist_ok = True)
    setup_logging(base / "logs" / f"scraper_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log.info("Scraper starting: repos=%s trial=%s", args.repos, args.trial)

    client = GitHubClient(min_remaining_graphql = 80, min_remaining_rest = 80)
    rl = client.rate_snapshot()
    log.info(
        "Rate limit snapshot: %s",
        json.dumps(rl.get("resources", {}), default = str)[:400],
    )

    # Start HF uploader in background if requested
    uploader = None
    if args.hf_upload_interval > 0:
        from hf_uploader import HFUploader

        uploader = HFUploader(data_dir, interval_s = args.hf_upload_interval)
        uploader.start()

    trial_limits = None
    if args.trial:
        trial_limits = {
            "issues": 5,
            "pull_requests": 5,
            "discussions": 3,
            "commits": 20,
            "releases": 3,
            "labels": 20,
            "milestones": 20,
        }

    only = set(args.only or [])

    try:
        for repo_spec in args.repos:
            owner, name = repo_spec.split("/")
            scraper = RepoScraper(owner, name, data_dir, client, trial_limits)
            try:
                repo_meta: Dict[str, Any] = {}
                if not only or "meta" in only or "commits" in only:
                    repo_meta = scraper.scrape_repo_meta()
                if not only or "labels" in only:
                    scraper.scrape_labels()
                if not only or "milestones" in only:
                    scraper.scrape_milestones()
                if not only or "releases" in only:
                    scraper.scrape_releases()
                if not only or "discussions" in only:
                    scraper.scrape_discussions()
                if not only or "issues" in only:
                    scraper.scrape_issues()
                if not only or "pulls" in only:
                    scraper.scrape_prs()
                if not only or "commits" in only:
                    default_ref = repo_meta.get("defaultBranchRef") or {}
                    default_branch = (
                        default_ref.get("name")
                        if isinstance(default_ref, dict)
                        else None
                    )
                    branch = (
                        f"refs/heads/{default_branch}"
                        if default_branch
                        else "refs/heads/main"
                    )
                    scraper.scrape_commits(branch = branch)
            finally:
                scraper.close()
    finally:
        if uploader:
            log.info("Stopping uploader and final sync...")
            uploader.stop(final_upload = True)
    log.info(
        "Scraper complete. GraphQL calls=%d REST calls=%d",
        client.calls_graphql,
        client.calls_rest,
    )


if __name__ == "__main__":
    main()
