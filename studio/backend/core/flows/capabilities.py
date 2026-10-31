# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Blocks that reach outside a flow: Python code, HTTP requests, MCP tools, external providers and triggers.

A spec lists what it needs in ``requires_approval``. Such a block runs only when (1) the Studio owner started the run
in Unsloth Studio, from a trigger the owner set up, or from a player the owner opened, never through an API key or MCP,
and (2) the owner approved exactly that capability for the flow: Python by the hash of its code, HTTP by host. Approvals
are stored beside the flow (``flows_db.get_capabilities``), never in the document, which any editor or import writes.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from urllib.parse import urlsplit

CAPS = ("python", "http", "mcp", "provider", "trigger")
TRUSTED_KINDS = ("ui", "trigger", "player")


def code_hash(code: Any) -> str:
    # The full digest: a truncated one lets a writer craft a colliding program and swap it in after approval.
    return "sha256:" + hashlib.sha256(str(code or "").encode()).hexdigest()


def url_host(url: Any) -> Optional[str]:
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return parts.hostname.lower().rstrip(".")


def node_needs(spec: dict, params: dict) -> list[tuple[str, str]]:
    """(capability, detail) pairs this block needs with these params."""
    found = []
    for cap in spec.get("requires_approval") or []:
        if cap == "python":
            found.append((cap, code_hash(params.get("code"))))
        elif cap == "http":
            found.append((cap, url_host(params.get("url")) or ""))
        elif cap == "provider":
            provider = str(params.get("provider") or "").strip()
            if provider:
                found.append((cap, provider))
        else:
            found.append((cap, cap))
    from core.flows.compiler import RECIPE_PARAM
    from core.flows.recipe_share import uses_mcp

    compiled = params.get(RECIPE_PARAM)
    # A recipe group's MCP servers come in on its compiled recipe, not the step's own spec.
    if (
        isinstance(compiled, dict)
        and uses_mcp(compiled.get("recipe"))
        and ("mcp", "mcp") not in found
    ):
        found.append(("mcp", "mcp"))
    return found


def doc_needs(doc: dict) -> dict:
    """Everything a flow document needs approved, in the shape stored as its approval."""
    from core.flows.engine import param_values
    from core.flows.specs import SPECS_BY_TYPE

    need: dict[str, Any] = {
        "python": [],
        "hosts": [],
        "mcp": False,
        "providers": [],
        "triggers": False,
    }
    for node in doc.get("nodes") or []:
        spec = SPECS_BY_TYPE.get(node.get("type"))
        if spec is None or node.get("disabled"):
            continue
        for cap, detail in node_needs(spec, param_values(spec, node)):
            if cap == "python" and detail not in need["python"]:
                need["python"].append(detail)
            elif cap == "http" and detail and detail not in need["hosts"]:
                need["hosts"].append(detail)
            elif cap == "mcp":
                need["mcp"] = True
            elif cap == "provider" and detail not in need["providers"]:
                need["providers"].append(detail)
            elif cap == "trigger":
                need["triggers"] = True
    # A recipe group's MCP servers live on the group, not on its Recipe Dataset step's params.
    from core.flows.recipe_import import GENERATE_TYPE
    from core.flows.recipe_share import uses_mcp

    running = {
        node.get("group")
        for node in doc.get("nodes") or []
        if node.get("type") == GENERATE_TYPE and not node.get("disabled")
    }
    for group in doc.get("groups") or []:
        if (
            group.get("kind") == "recipe"
            and group.get("id") in running
            and uses_mcp(group.get("config"))
        ):
            need["mcp"] = True
    return need


def is_approved(approved: dict, cap: str, detail: str) -> bool:
    approved = approved or {}
    if cap == "python":
        return detail in (approved.get("python") or [])
    if cap == "http":
        return bool(detail) and detail in (approved.get("hosts") or [])
    if cap == "mcp":
        return approved.get("mcp") is True
    if cap == "provider":
        return detail in (approved.get("providers") or [])
    if cap == "trigger":
        return approved.get("triggers") is True
    return False


def pending(doc: dict, approved: dict) -> dict:
    """The part of ``doc_needs`` not yet approved (empty lists / False when nothing is missing)."""
    need = doc_needs(doc)
    return {
        "python": [h for h in need["python"] if not is_approved(approved, "python", h)],
        "hosts": [h for h in need["hosts"] if not is_approved(approved, "http", h)],
        "mcp": need["mcp"] and not is_approved(approved, "mcp", "mcp"),
        "providers": [p for p in need["providers"] if not is_approved(approved, "provider", p)],
        "triggers": need["triggers"] and not is_approved(approved, "trigger", "trigger"),
    }


def has_pending(found: dict) -> bool:
    return any(bool(v) for v in found.values())


def origin_trusted(origin: Optional[dict]) -> bool:
    return bool(origin) and origin.get("owner") is True and origin.get("kind") in TRUSTED_KINDS


_WHAT = {
    "python": "runs Python code",
    "http": "calls {detail}",
    "mcp": "uses your MCP servers",
    "provider": "uses the {detail} connection",
    "trigger": "starts on its own",
}


def refusal(
    origin: Optional[dict], approved: dict, cap: str, detail: str, title: str
) -> Optional[str]:
    """Why this capability may not be used now, or None."""
    if not origin_trusted(origin):
        return (
            f"{title} only runs when the Studio owner starts this flow in Unsloth Studio, not through an API key, "
            "MCP or another account"
        )
    if cap == "http" and not detail:
        return f"{title} needs a full http:// or https:// URL"
    if not is_approved(approved, cap, detail):
        what = _WHAT.get(cap, cap).format(detail = detail)
        return f"{title} {what}, which you have not approved for this flow yet. Open the flow and approve it."
    return None
