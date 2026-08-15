# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional supervisor jobs: approval voter and episode planner.

Not the MemoryWheels outer wheel (that is B + C). Both jobs are one-shot,
no-tools completes. They do not call admit() or decide().
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from unforgettable.constants import RECORD_BODY_CHARS, RECORD_TITLE_CHARS
from unforgettable.host import SUPERVISE_MAX_TOKENS
from unforgettable.store.records import log_admission

PURPOSE_VOTE = "vote"
PURPOSE_PLAN = "plan"
PURPOSE_MINE = "mine"
SUPERVISE_PURPOSES = frozenset({PURPOSE_VOTE, PURPOSE_PLAN, PURPOSE_MINE})

VOTER_OFF = "off"
VOTER_ADVISORY = "advisory"
VOTER_BINDING = "binding"
VOTER_MODES = frozenset({VOTER_OFF, VOTER_ADVISORY, VOTER_BINDING})

PLANNER_OFF = "off"
PLANNER_ON = "on"

VOTE_ALLOW = "allow"
VOTE_DENY = "deny"
VOTE_ABSTAIN = "abstain"
VOTE_DECISIONS = frozenset({VOTE_ALLOW, VOTE_DENY, VOTE_ABSTAIN})

SKIP_VOTE_KINDS = frozenset({"episode"})
MINE_KINDS = frozenset({"claim", "procedure", "error_fix", "entity", "twin_note"})

VOTER_ENV = "UNFORGETTABLE_VOTER"
PLANNER_ENV = "UNFORGETTABLE_PLANNER"
VOTER_MODEL_ENV = "UNFORGETTABLE_VOTER_MODEL"
PLANNER_MODEL_ENV = "UNFORGETTABLE_PLANNER_MODEL"
SUPERVISOR_URL_ENV = "UNFORGETTABLE_SUPERVISOR_URL"
SUPERVISOR_TIMEOUT_ENV = "UNFORGETTABLE_SUPERVISOR_TIMEOUT"

DEFAULT_SUPERVISOR_TIMEOUT = 30.0
PLANNER_MAX_CHARS = 1200
VOTE_REASON_CHARS = 400
MINE_MAX_ROWS = 20
HTTP_SUPERVISOR_CLIP = 8000

PLANNER_HEADER = (
    "Supervisor plan (this episode only; not durable memory — "
    "do not memory_write this block unless you independently verify it):"
)

VOTER_SYSTEM = (
    "You are an approval voter for a durable memory store. "
    "The local actor already selected this candidate. "
    "Reply with a JSON object only: "
    '{"decision":"allow"|"deny"|"abstain","reason":"<short>"}. '
    "Allow only durable, non-secret, non-one-off lessons. "
    "Deny junk, secrets, sim-only dynamics presented as world truth, "
    "and anything that should stay proposed. Abstain if unsure."
)

PLANNER_SYSTEM = (
    "You are a temporary planner for a local coding actor. "
    "Write a short plan the actor can follow with python/terminal tools. "
    "No tools. No durable memory writes. "
    "At most 12 short lines. Do not paste secrets."
)

MINE_SYSTEM = (
    "You review a memory store for promotion and extra lesson drafts. "
    "Reply with a JSON array only. Each item is either "
    '{"id":"<existing id>","decision":"allow"|"deny"|"abstain","reason":"<short>"} '
    "or a new draft "
    '{"kind":"claim"|"procedure"|"error_fix"|"entity"|"twin_note",'
    '"title":"...","body":"..."} . '
    "New drafts stay proposed. Do not invent secrets or episode transcripts."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class SupervisorConfig:
    voter: str = VOTER_OFF
    planner: str = PLANNER_OFF
    voter_model: Optional[str] = None
    planner_model: Optional[str] = None
    url: Optional[str] = None
    timeout: float = DEFAULT_SUPERVISOR_TIMEOUT


@dataclass(frozen=True)
class Vote:
    decision: str
    reason: str
    raw: str = ""


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _normalize_voter(value: str | None) -> str:
    text = (value or VOTER_OFF).strip().lower()
    if text in {"advise", "advice", "hint"}:
        return VOTER_ADVISORY
    if text in {"bind", "on", "true", "1", "yes"}:
        return VOTER_BINDING
    if text in VOTER_MODES:
        return text
    return VOTER_OFF


def _normalize_planner(value: str | None) -> str:
    if flag_is_on(value):
        return PLANNER_ON
    return PLANNER_OFF


def flag_is_on(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "on", "true", "yes", "external"}


def coerce_planner_flag(value: Any) -> Optional[str]:
    if value is None:
        return None
    if value is True:
        return PLANNER_ON
    if value is False:
        return PLANNER_OFF
    text = str(value).strip()
    if not text:
        return None
    return PLANNER_ON if flag_is_on(text) else PLANNER_OFF


def config_from_env() -> SupervisorConfig:
    timeout_raw = _env(SUPERVISOR_TIMEOUT_ENV)
    timeout = DEFAULT_SUPERVISOR_TIMEOUT
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = DEFAULT_SUPERVISOR_TIMEOUT
        if timeout <= 0:
            timeout = DEFAULT_SUPERVISOR_TIMEOUT
    return SupervisorConfig(
        voter=_normalize_voter(_env(VOTER_ENV)),
        planner=_normalize_planner(_env(PLANNER_ENV)),
        voter_model=_env(VOTER_MODEL_ENV) or None,
        planner_model=_env(PLANNER_MODEL_ENV) or None,
        url=_env(SUPERVISOR_URL_ENV) or None,
        timeout=timeout,
    )


def planner_is_on(request: Any, config: SupervisorConfig | None = None) -> bool:
    """Request-scoped. Studio copies UNFORGETTABLE_PLANNER onto EpisodeRequest."""
    del config
    return flag_is_on(getattr(request, "planner", None))


def should_vote(candidate: dict[str, Any]) -> bool:
    kind = (candidate.get("kind") or "").strip()
    return kind not in SKIP_VOTE_KINDS


def _clip(text: str, limit: int) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "..."


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw or "").strip()


def parse_vote(raw: str) -> Vote:
    text = _strip_fences(raw)
    if not text:
        return Vote(VOTE_ABSTAIN, "empty supervisor reply", raw=raw or "")
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        decision = str(parsed.get("decision") or "").strip().lower()
        reason = _clip(str(parsed.get("reason") or ""), VOTE_REASON_CHARS)
        if decision in VOTE_DECISIONS:
            return Vote(decision, reason or decision, raw=text)
    token = re.split(r"[\s:,]+", text, maxsplit=1)[0].strip().lower()
    rest = text[len(token) :].lstrip(" :,-")
    if token in VOTE_DECISIONS:
        return Vote(token, _clip(rest, VOTE_REASON_CHARS) or token, raw=text)
    return Vote(VOTE_ABSTAIN, "unparsed supervisor reply", raw=text)


def parse_mine(raw: str) -> list[dict[str, Any]]:
    text = _strip_fences(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for item in parsed[:MINE_MAX_ROWS]:
        if not isinstance(item, dict):
            continue
        rec_id = str(item.get("id") or "").strip()
        if rec_id:
            vote = parse_vote(json.dumps(item))
            out.append(
                {
                    "id": rec_id,
                    "decision": vote.decision,
                    "reason": vote.reason,
                    "kind": None,
                    "title": None,
                    "body": None,
                }
            )
            continue
        kind = str(item.get("kind") or "").strip()
        title = _clip(str(item.get("title") or ""), RECORD_TITLE_CHARS)
        body = _clip(str(item.get("body") or ""), RECORD_BODY_CHARS)
        if kind in MINE_KINDS and title:
            out.append(
                {
                    "id": None,
                    "decision": None,
                    "reason": None,
                    "kind": kind,
                    "title": title,
                    "body": body,
                }
            )
    return out


def vote_messages(candidate: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "id": candidate.get("id"),
        "kind": candidate.get("kind"),
        "status": candidate.get("status"),
        "provenance": candidate.get("provenance"),
        "title": candidate.get("title"),
        "body": _clip(str(candidate.get("body") or ""), 1500),
    }
    extra = candidate.get("extra")
    if extra:
        payload["extra"] = extra
    return [
        {"role": "system", "content": VOTER_SYSTEM},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]


def plan_messages(user_text: str, *, extra: str = "") -> list[dict[str, str]]:
    parts = [f"User request:\n{_clip(user_text, 800)}"]
    note = _clip(extra, 800)
    if note:
        parts.append(f"Episode notes:\n{note}")
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def mine_messages(
    *,
    proposed: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "proposed": [
            {
                "id": row.get("id"),
                "kind": row.get("kind"),
                "provenance": row.get("provenance"),
                "title": row.get("title"),
                "body": _clip(str(row.get("body") or ""), 400),
            }
            for row in proposed[:MINE_MAX_ROWS]
        ],
        "rollouts": [
            {
                "contact": row.get("contact"),
                "outcome": row.get("outcome"),
                "summary": _clip(str(row.get("summary") or ""), 200),
            }
            for row in rollouts[:MINE_MAX_ROWS]
        ],
        "admissions": [
            {
                "record_id": row.get("record_id"),
                "decision": row.get("decision"),
                "reason": _clip(str(row.get("reason") or ""), 160),
            }
            for row in admissions[:MINE_MAX_ROWS]
        ],
    }
    return [
        {"role": "system", "content": MINE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]


def planner_block(text: str) -> str:
    body = _clip(text, PLANNER_MAX_CHARS)
    if not body:
        return ""
    return f"{PLANNER_HEADER}\n{body}"


def voter_blocks(vote: Vote, *, force: bool, config: SupervisorConfig) -> bool:
    if force:
        return False
    return config.voter == VOTER_BINDING and vote.decision == VOTE_DENY


async def call_supervise(
    host: Any,
    purpose: str,
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: int = SUPERVISE_MAX_TOKENS,
) -> Optional[str]:
    fn = getattr(host, "supervise", None)
    if fn is None:
        return None
    return await fn(purpose, messages, model=model, max_tokens=max_tokens)


async def request_vote(
    candidate: dict[str, Any],
    *,
    host: Any = None,
    config: SupervisorConfig | None = None,
    db_path=None,
) -> Vote:
    cfg = config or config_from_env()
    if cfg.voter == VOTER_OFF:
        return Vote(VOTE_ABSTAIN, "voter off")
    if not should_vote(candidate):
        return Vote(VOTE_ABSTAIN, "bookkeeping skip")
    if host is None:
        return Vote(VOTE_ABSTAIN, "no supervisor")
    try:
        raw = await call_supervise(
            host,
            PURPOSE_VOTE,
            vote_messages(candidate),
            model=cfg.voter_model,
        )
    except Exception as exc:
        return Vote(VOTE_ABSTAIN, f"supervisor failed: {exc}")
    if raw is None:
        return Vote(VOTE_ABSTAIN, "host has no supervise")
    vote = parse_vote(raw)
    if db_path is not None:
        log_admission(
            record_id=candidate.get("id"),
            decision=f"voter:{vote.decision}",
            reason=vote.reason,
            db_path=db_path,
        )
    return vote


def request_vote_sync(
    candidate: dict[str, Any],
    *,
    host: Any = None,
    config: SupervisorConfig | None = None,
    db_path=None,
) -> Vote:
    return asyncio.run(
        request_vote(candidate, host=host, config=config, db_path=db_path)
    )


async def request_plan(
    host: Any,
    *,
    user_text: str,
    extra: str = "",
    model: Optional[str] = None,
) -> str:
    try:
        raw = await call_supervise(
            host,
            PURPOSE_PLAN,
            plan_messages(user_text, extra=extra),
            model=model,
        )
    except Exception:
        return ""
    if not raw:
        return ""
    return _clip(raw, PLANNER_MAX_CHARS)


async def request_mine(
    host: Any,
    *,
    proposed: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
    config: SupervisorConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or config_from_env()
    try:
        raw = await call_supervise(
            host,
            PURPOSE_MINE,
            mine_messages(
                proposed=proposed, rollouts=rollouts, admissions=admissions
            ),
            model=cfg.voter_model,
        )
    except Exception:
        return []
    if not raw:
        return []
    return parse_mine(raw)


def request_mine_sync(
    host: Any,
    *,
    proposed: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
    config: SupervisorConfig | None = None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        request_mine(
            host,
            proposed=proposed,
            rollouts=rollouts,
            admissions=admissions,
            config=config,
        )
    )


def post_supervisor(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_SUPERVISOR_TIMEOUT,
) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    raw = raw[:HTTP_SUPERVISOR_CLIP]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        for key in ("text", "content"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return raw


class HttpSupervisor:
    """Headless Host-shaped voter/planner: POST JSON to SUPERVISOR_URL."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = DEFAULT_SUPERVISOR_TIMEOUT,
    ) -> None:
        self.url = url
        self.timeout = timeout

    async def supervise(
        self,
        purpose: str,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = SUPERVISE_MAX_TOKENS,
    ) -> str:
        payload = {
            "purpose": purpose,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        try:
            return await asyncio.to_thread(
                post_supervisor, self.url, payload, timeout=self.timeout
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"supervisor http failed: {exc}") from exc


def resolve_supervisor_host(
    config: SupervisorConfig | None = None,
) -> Optional[HttpSupervisor]:
    cfg = config or config_from_env()
    if not cfg.url:
        return None
    return HttpSupervisor(cfg.url, timeout=cfg.timeout)
