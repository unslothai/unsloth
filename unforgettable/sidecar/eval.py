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

"""Holdout lean score vs base. Does not call Host.complete.

Prefix/substring is the default. An optional supervisor judge overlays it
when ``judge_model`` is configured.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from typing import Any

from unforgettable.eyes.gate import LogGateEyes
from unforgettable.eyes.probes import run_probes
from unforgettable.sidecar.adapters import get_adapter, set_adapter_metrics
from unforgettable.sidecar.pack import ROLE_HOLDOUT, list_pack_items
from unforgettable.sidecar.train import TrainBackend
from unforgettable.supervisor import SupervisorConfig, config_from_env, request_score_sync

EVAL_CLIP = 200


@dataclass(frozen = True)
class EvalReport:
    adapter_id: str
    n_holdout: int
    adapter_lean: float
    base_lean: float
    probes_pass: int
    probes_fail: int
    passed: bool


def completion_score(
    output: str,
    gold: str,
    *,
    clip: int = EVAL_CLIP,
) -> float:
    """Prefix / substring lean. Used as the default and as the judge fallback."""
    g = (gold or "")[:clip].strip().casefold()
    o = (output or "")[:clip].strip().casefold()
    if not g:
        return 0.0
    if g in o:
        return 1.0
    n = 0
    for a, b in zip(g, o):
        if a != b:
            break
        n += 1
    return n / len(g)


def scored_completion(
    output: str,
    gold: str,
    *,
    clip: int = EVAL_CLIP,
    host = None,
    model: str | None = None,
) -> float:
    algo = completion_score(output, gold, clip = clip)
    if host is None or not model:
        return algo
    judged = request_score_sync(host, output = output, gold = gold, model = model)
    if judged is None:
        return algo
    return judged


def _user_only_and_gold(item: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = item.get("messages") or []
    if isinstance(messages, str):
        messages = json.loads(messages)
    user_only: list[dict[str, str]] = []
    gold = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            user_only.append({"role": "user", "content": content})
        elif role == "assistant":
            gold = content
    return user_only, gold


def eval_adapter(
    adapter_id: str,
    *,
    backend: TrainBackend,
    world = None,
    db_path = None,
    host = None,
    config: SupervisorConfig | None = None,
) -> EvalReport:
    adapter = get_adapter(adapter_id, db_path = db_path)
    if adapter is None:
        raise KeyError(adapter_id)
    items = list_pack_items(adapter["pack_id"], db_path = db_path)
    holdout = [item for item in items if item.get("role") == ROLE_HOLDOUT]
    adapter_scores: list[float] = []
    base_scores: list[float] = []
    adapter_path = adapter.get("path") or None
    cfg = config or config_from_env()
    judge_model = cfg.judge_model
    judge_host = host if judge_model else None
    for item in holdout:
        user_only, gold = _user_only_and_gold(item)
        adapter_out = backend.complete(user_only, adapter_path = adapter_path, max_tokens = 80)
        base_out = backend.complete(user_only, adapter_path = None, max_tokens = 80)
        adapter_scores.append(
            scored_completion(
                adapter_out,
                gold,
                host = judge_host,
                model = judge_model,
            )
        )
        base_scores.append(
            scored_completion(
                base_out,
                gold,
                host = judge_host,
                model = judge_model,
            )
        )
    n_holdout = len(holdout)
    if n_holdout:
        adapter_lean = sum(adapter_scores) / n_holdout
        base_lean = sum(base_scores) / n_holdout
    else:
        adapter_lean = 0.0
        base_lean = 0.0

    probes_pass = 0
    probes_fail = 0
    probes_ran = world is not None
    if probes_ran:
        results = run_probes(world = world, host = None, db_path = db_path)
        if inspect.isawaitable(results):
            raise RuntimeError("eval_adapter cannot await run_probes")
        for row in results:
            if row.get("outcome") == "pass":
                probes_pass += 1
            else:
                probes_fail += 1

    if n_holdout == 0 and not probes_ran:
        passed = False
    elif n_holdout > 0 and adapter_lean == 0.0 and base_lean == 0.0:
        # Empty completions on both sides are not a lean win (Fake holdout
        # titles are not in train gold; Unsloth-without-base also scores 0).
        passed = False
    else:
        passed = adapter_lean >= base_lean and adapter_lean > 0.0 and probes_fail == 0

    report = EvalReport(
        adapter_id = adapter_id,
        n_holdout = n_holdout,
        adapter_lean = adapter_lean,
        base_lean = base_lean,
        probes_pass = probes_pass,
        probes_fail = probes_fail,
        passed = passed,
    )
    set_adapter_metrics(adapter_id, asdict(report), db_path = db_path)
    LogGateEyes().note(f"eval: lean={adapter_lean}", db_path = db_path)
    return report
