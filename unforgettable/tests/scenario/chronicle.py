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

"""Markdown timeline of a ledger-week run. Marketing tape, generated not hand-waved."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from unforgettable.loop.episode import EpisodeOutcome
from unforgettable.store.compile import count_compiled
from unforgettable.store.records import list_inject_stats, list_records, summarize_records
from unforgettable.throne.policy import Action

from .script import Scene


def _kind_counts(db_path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in list_records(db_path = db_path):
        key = f"{rec.get('kind')}:{rec.get('status')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def render_chronicle(
    plays: Iterable[tuple[Scene, EpisodeOutcome]],
    *,
    db_path,
    pack: Optional[Any] = None,
) -> str:
    plays = list(plays)
    lines = ["# Ledger week — Unforgettable", ""]
    summary = summarize_records(db_path = db_path)
    lines.append(f"Episodes: {len(plays)}    Records: {summary['total']}")
    counts = _kind_counts(db_path)
    lines.append(
        "Active playbooks: {procs}    Error-fixes admitted: {fixes}".format(
            procs = counts.get("procedure:active", 0),
            fixes = counts.get("error_fix:active", 0),
        )
    )
    lines.append(
        "Standing: {standing}    Twin notes: {twins}".format(
            standing = count_compiled(db_path = db_path),
            twins = counts.get("twin_note:active", 0),
        )
    )
    if pack is not None:
        lines.append(f"Pack: {pack.n_train} train / {pack.n_holdout} holdout")
    lines.append("")
    lines.append("| Scene | Actions | Notes |")
    lines.append("|-------|---------|-------|")
    for scene, outcome in plays:
        actions = ",".join(outcome.actions) if outcome.actions else "finish"
        note = (outcome.text or "").splitlines()[0] if outcome.text else ""
        if Action.ENTER_SIM in outcome.actions and Action.RETRY_WORLD in outcome.actions:
            if Action.ESCALATE in outcome.actions:
                note = "world fail → sim → world still red"
            else:
                note = "world fail → sim → world retry"
        lines.append(f"| {scene.name} | {actions} | {note} |")
    stats = list_inject_stats(limit = 5, db_path = db_path)
    if stats:
        lines.append("")
        lines.append("Latest inject_stats (standing + retrieve + traj):")
        for row in stats[:5]:
            lines.append(
                "- {contact}: standing={standing} retrieve={retrieve} traj={traj} total={total}".format(
                    contact = row.get("contact"),
                    standing = row.get("standing_chars"),
                    retrieve = row.get("retrieve_chars"),
                    traj = row.get("trajectory_chars"),
                    total = row.get("total_chars"),
                )
            )
    lines.append("")
    return "\n".join(lines)
