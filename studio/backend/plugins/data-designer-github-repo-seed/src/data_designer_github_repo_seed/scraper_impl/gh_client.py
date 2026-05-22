# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GitHub API client with rate-limit awareness, retry, and dual REST/GraphQL support."""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

log = logging.getLogger("gh_client")

GRAPHQL_URL = "https://api.github.com/graphql"
REST_BASE = "https://api.github.com"

BASE_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "github-data-gatherer/1.0",
}


class RateLimitError(Exception):
    pass


class GitHubAuthError(RuntimeError):
    """Raised when GitHub returns 401/403 due to invalid or insufficient credentials."""


def _retry_after_seconds(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return max(0, int(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo = timezone.utc)
    return max(0, int(retry_at.timestamp() - time.time()))


class GitHubClient:
    def __init__(
        self,
        min_remaining_graphql: int = 100,
        min_remaining_rest: int = 100,
        token: str | None = None,
        token_source: str | None = None,
    ):
        if token:
            self._token_source = (
                token_source or "explicit token argument (recipe-level field)"
            )
        elif os.environ.get("GH_TOKEN"):
            self._token_source = "GH_TOKEN environment variable"
            token = os.environ["GH_TOKEN"]
        elif os.environ.get("GITHUB_TOKEN"):
            self._token_source = "GITHUB_TOKEN environment variable"
            token = os.environ["GITHUB_TOKEN"]
        else:
            raise RuntimeError("GH_TOKEN or GITHUB_TOKEN not set in environment")
        self.session = requests.Session()
        self.session.headers.update(
            {**BASE_HEADERS, "Authorization": f"Bearer {token}"}
        )
        self.min_remaining_graphql = min_remaining_graphql
        self.min_remaining_rest = min_remaining_rest
        self.graphql_remaining: Optional[int] = None
        self.graphql_reset: Optional[int] = None
        self.rest_remaining: Optional[int] = None
        self.rest_reset: Optional[int] = None
        self.calls_graphql = 0
        self.calls_rest = 0
        self.retry_count = 0

    def _sleep_until(self, reset_ts: int, buffer_s: int = 10) -> None:
        now = int(time.time())
        wait = max(0, reset_ts - now) + buffer_s
        log.warning("Rate limit hit. Sleeping %ds until reset.", wait)
        time.sleep(wait)

    def _is_rate_limit_response(self, r: "requests.Response") -> bool:
        if r.headers.get("Retry-After"):
            return True
        if r.headers.get("X-RateLimit-Remaining") == "0":
            return True
        body = (r.text or "").lower()
        return any(
            marker in body
            for marker in (
                "api rate limit exceeded",
                "rate limit exceeded",
                "secondary rate limit",
                "secondary limit",
                "abuse detection mechanism",
                "abuse detection",
            )
        )

    def _is_auth_failure(self, r: "requests.Response") -> bool:
        """Distinguish auth failures from rate limiting on 401/403 responses.

        - 401: always an auth failure (invalid / expired / wrong-scope token).
        - 403: an auth failure UNLESS the response carries a clear rate-limit signal
          (Retry-After header, X-RateLimit-Remaining: 0, or GitHub's secondary /
          abuse rate-limit response text).
        """
        if r.status_code == 401:
            return True
        if r.status_code == 403:
            return not self._is_rate_limit_response(r)
        return False

    def _raise_auth_error(self, r: "requests.Response", endpoint: str) -> None:
        snippet = (r.text or "").strip()[:200]
        request_id = r.headers.get("X-GitHub-Request-Id")
        request_id_message = f" Request ID: {request_id}." if request_id else ""
        raise GitHubAuthError(
            f"GitHub {endpoint} returned {r.status_code} {r.reason}. "
            f"Token source: {self._token_source}. "
            f"The token is invalid, expired, or missing required scopes — "
            f"retrying will not recover.{request_id_message} Response: {snippet}"
        )

    def _check_rate_and_wait(self, kind: str) -> None:
        if kind == "graphql":
            remaining = self.graphql_remaining
            reset = self.graphql_reset
            min_remaining = self.min_remaining_graphql
        else:
            remaining = self.rest_remaining
            reset = self.rest_reset
            min_remaining = self.min_remaining_rest
        if remaining is not None and remaining < min_remaining:
            if reset:
                self._sleep_until(reset)
                # Reset remaining so we don't spin
                if kind == "graphql":
                    self.graphql_remaining = None
                else:
                    self.rest_remaining = None

    def graphql(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        max_retries: int = 20,
    ) -> Dict[str, Any]:
        self._check_rate_and_wait("graphql")
        backoff = 2
        last_err = None
        for attempt in range(max_retries):
            try:
                r = self.session.post(
                    GRAPHQL_URL,
                    json = {"query": query, "variables": variables or {}},
                    timeout = 120,
                )
                self.calls_graphql += 1
                # Update rate info from response headers
                rem = r.headers.get("X-RateLimit-Remaining")
                rst = r.headers.get("X-RateLimit-Reset")
                if rem is not None:
                    try:
                        self.graphql_remaining = int(rem)
                    except ValueError:
                        pass
                if rst is not None:
                    try:
                        self.graphql_reset = int(rst)
                    except ValueError:
                        pass
                if r.status_code in (502, 503, 504):
                    log.warning("GraphQL %s transient, retrying", r.status_code)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if self._is_auth_failure(r):
                    self._raise_auth_error(r, "GraphQL")
                if r.status_code == 403 or r.status_code == 429:
                    # Check for secondary/abuse
                    retry_after = _retry_after_seconds(r.headers.get("Retry-After"))
                    if retry_after is not None:
                        log.warning("Secondary rate limit. Sleep %ds.", retry_after)
                        time.sleep(retry_after + 2)
                        continue
                    if self.graphql_reset:
                        self._sleep_until(self.graphql_reset)
                        continue
                    time.sleep(60)
                    continue
                r.raise_for_status()
                data = r.json()
                if "errors" in data and data["errors"]:
                    # Surface errors but allow partial data
                    errs = data["errors"]
                    # Retry on RATE_LIMITED
                    for e in errs:
                        if e.get("type") == "RATE_LIMITED":
                            self._sleep_until(
                                (self.graphql_reset or int(time.time()) + 60)
                            )
                            break
                    else:
                        # No rate-limit error, log and return partial
                        log.warning("GraphQL errors: %s", json.dumps(errs)[:400])
                        return data
                    continue
                return data
            except requests.RequestException as e:
                last_err = e
                log.warning("GraphQL network error: %s. Retry.", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        raise RuntimeError(f"GraphQL failed after {max_retries} retries: {last_err}")

    def rest(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 6,
    ) -> requests.Response:
        self._check_rate_and_wait("rest")
        if path.startswith("http"):
            url = path
        else:
            url = REST_BASE + path
        backoff = 2
        last_err = None
        for attempt in range(max_retries):
            try:
                r = self.session.request(
                    method, url, params = params, json = json_body, timeout = 120
                )
                self.calls_rest += 1
                rem = r.headers.get("X-RateLimit-Remaining")
                rst = r.headers.get("X-RateLimit-Reset")
                if rem is not None:
                    try:
                        self.rest_remaining = int(rem)
                    except ValueError:
                        pass
                if rst is not None:
                    try:
                        self.rest_reset = int(rst)
                    except ValueError:
                        pass
                if r.status_code in (502, 503, 504):
                    log.warning("REST %s transient, retrying", r.status_code)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if self._is_auth_failure(r):
                    self._raise_auth_error(r, "REST")
                if r.status_code in (403, 429):
                    retry_after = _retry_after_seconds(r.headers.get("Retry-After"))
                    if retry_after is not None:
                        log.warning(
                            "Secondary rate limit on REST. Sleep %ds.", retry_after
                        )
                        time.sleep(retry_after + 2)
                        continue
                    # Check if primary rate
                    if self.rest_remaining == 0 and self.rest_reset:
                        self._sleep_until(self.rest_reset)
                        continue
                    log.warning("REST 403/429, sleep 60")
                    time.sleep(60)
                    continue
                return r
            except requests.RequestException as e:
                last_err = e
                log.warning("REST network error: %s. Retry.", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        raise RuntimeError(f"REST failed after {max_retries} retries: {last_err}")

    def rest_paginate(
        self, path: str, params: Optional[Dict[str, Any]] = None, per_page: int = 100
    ) -> Iterator[dict]:
        params = dict(params or {})
        params.setdefault("per_page", per_page)
        url = path
        while True:
            r = self.rest("GET", url, params = params if url == path else None)
            if r.status_code != 200:
                log.error(
                    "REST paginate got %s at %s: %s", r.status_code, url, r.text[:200]
                )
                return
            items = r.json()
            if isinstance(items, dict):
                # Some endpoints return dict with list field
                items = items.get("items", [])
            for it in items:
                yield it
            # Follow link header
            link = r.headers.get("Link", "")
            nxt = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    nxt = part.split(";")[0].strip().strip("<>")
                    break
            if not nxt:
                return
            url = nxt
            params = None

    def rate_snapshot(self) -> Dict[str, Any]:
        r = self.rest("GET", "/rate_limit")
        if r.status_code == 200:
            return r.json()
        return {}
