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

from __future__ import annotations

from dataclasses import dataclass

from unforgettable.constants import DEFAULT_NAMESPACE_ID
from unforgettable.store.records import ensure_default_namespace, get_namespace, log_admission


@dataclass(frozen = True)
class AdmissionDecision:
    status: str
    reason: str


def admit(
    *,
    kind: str,
    provenance: str,
    explicit: bool,
    namespace_id: str = DEFAULT_NAMESPACE_ID,
    record_id: str | None = None,
    db_path = None,
    bookkeeping: bool = False,
    force_proposed_reason: str | None = None,
    persist_log: bool = True,
) -> AdmissionDecision:
    """Decide status before insert. Logs every decision unless persist_log is false."""
    ensure_default_namespace(db_path = db_path)
    ns = get_namespace(namespace_id, db_path = db_path)
    mode = (ns or {}).get("admission") or "auto"

    if mode == "deny":
        decision = AdmissionDecision("rejected", "namespace denies writes")
    elif mode == "propose":
        decision = AdmissionDecision("proposed", "namespace is propose-only")
    elif force_proposed_reason:
        decision = AdmissionDecision("proposed", force_proposed_reason)
    elif bookkeeping:
        decision = AdmissionDecision("active", "bookkeeping write admitted")
    elif kind in {"claim", "procedure"} and provenance == "sim":
        decision = AdmissionDecision(
            "proposed", "sim-only claims and procedures are not auto-promoted"
        )
    elif not explicit:
        decision = AdmissionDecision("proposed", "auto-extract is proposed until eyes confirm")
    elif provenance == "infer":
        decision = AdmissionDecision("proposed", "infer provenance stays proposed")
    elif kind == "directive" and provenance != "human":
        decision = AdmissionDecision(
            "proposed", "directives stay proposed until a human admits them"
        )
    else:
        decision = AdmissionDecision("active", "explicit write admitted")

    if persist_log:
        log_admission(
            record_id = record_id,
            decision = decision.status,
            reason = decision.reason,
            db_path = db_path,
        )
    return decision
