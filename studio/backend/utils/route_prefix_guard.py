# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Let the router skip route-table entries whose literal path prefix rules the request out.

The router tries every entry in order until one matches. With ~800 routes, a request for a path
registered late (``/api/health``, ``/api/liveness``, every ``/assets`` file, the SPA, any 404) costs
~0.9 ms of regex and bookkeeping on the event loop before its handler runs.

An entry's path regex is anchored at the start of the literal text before its first ``{``, so a route
path that does not start with that text cannot match it at all: ``matches()`` returns ``Match.NONE``.
The guard returns exactly that without running the regex. Anything it cannot prove is path-anchored
(Host routes, a ``matches`` override, a ``root_path`` deployment) is left on the original code path.
Order, precedence, 405 partial matches and the OpenAPI schema are untouched.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from starlette.routing import BaseRoute, Match, Mount, Route, WebSocketRoute

try:
    from fastapi.routing import APIRoute, APIWebSocketRoute
except Exception:  # pragma: no cover - fastapi is a hard dependency of the backend
    APIRoute = APIWebSocketRoute = None

# Route classes whose stock ``matches`` is a start-anchored regex over the route path.
_PATH_ROUTE_TYPES = tuple(
    cls for cls in (Route, WebSocketRoute, Mount, APIRoute, APIWebSocketRoute) if cls is not None
)
_STOCK_MATCHES = frozenset(cls.matches for cls in _PATH_ROUTE_TYPES)
_GUARD_MARK = "_unsloth_prefix_guard"


def _literal_prefix(path: Any) -> Optional[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    literal = path.split("{", 1)[0]
    # "/" (or less) rules nothing out, so a guard would only add work.
    return literal if len(literal) > 1 else None


def _stock_path_route(route: Any) -> bool:
    return isinstance(route, _PATH_ROUTE_TYPES) and type(route).matches in _STOCK_MATCHES


def _included_router_parts(entry: Any):
    """(prefix, original_router) for FastAPI's lazily-included router entry (exact class), else None."""
    cls = type(entry)
    if cls.__name__ != "_IncludedRouter" or cls.__module__ != "fastapi.routing":
        return None
    prefix = getattr(getattr(entry, "include_context", None), "prefix", None)
    original = getattr(entry, "original_router", None)
    if not isinstance(prefix, str) or original is None:
        return None
    return prefix, original


def _subtree_is_path_anchored(router: Any, seen: set) -> bool:
    """Every route an included router can reach is matched on its (prefixed) path."""
    if id(router) in seen:
        return True
    seen.add(id(router))
    for route in getattr(router, "routes", None) or ():
        nested = _included_router_parts(route)
        if nested is not None:
            if not _subtree_is_path_anchored(nested[1], seen):
                return False
        elif not _stock_path_route(route):
            return False
    return True


def _leaf_ids(router: Any, out: set, seen: set) -> set:
    if id(router) in seen:
        return out
    seen.add(id(router))
    for route in getattr(router, "routes", None) or ():
        nested = _included_router_parts(route)
        if nested is not None:
            _leaf_ids(nested[1], out, seen)
        else:
            out.add(id(route))
    return out


def _guard_prefix(entry: Any, shared_leaf_ids: set) -> Optional[str]:
    included = _included_router_parts(entry)
    if included is not None:
        prefix, original = included
        if not _subtree_is_path_anchored(original, set()):
            return None
        return _literal_prefix(prefix)
    if not _stock_path_route(entry):
        return None
    # A route object also reachable through an included router is matched there against its prefixed
    # path, through this same bound method; its own path would be the wrong anchor.
    if id(entry) in shared_leaf_ids:
        return None
    return _literal_prefix(getattr(entry, "path", None))


def _guarded(matches: Callable, prefix: str) -> Callable:
    def guarded_matches(scope):
        # root_path deployments strip a prefix first (starlette get_route_path); keep the original path there.
        if not scope.get("root_path") and not scope["path"].startswith(prefix):
            return Match.NONE, {}
        return matches(scope)

    guarded_matches.__wrapped__ = matches
    setattr(guarded_matches, _GUARD_MARK, prefix)
    return guarded_matches


def install_route_prefix_guards(routes: list) -> int:
    """Guard each eligible top-level entry once; returns how many are guarded. Safe to call again."""
    shared: set = set()
    for entry in routes:
        included = _included_router_parts(entry)
        if included is not None:
            _leaf_ids(included[1], shared, set())
    count = 0
    for entry in routes:
        if not isinstance(entry, BaseRoute):
            continue
        try:
            own = vars(entry)
        except TypeError:
            continue
        if _GUARD_MARK in own:
            count += 1
            continue
        prefix = _guard_prefix(entry, shared)
        if prefix is None:
            continue
        try:
            entry.matches = _guarded(entry.matches, prefix)
            setattr(entry, _GUARD_MARK, prefix)
        except (AttributeError, TypeError):
            continue
        count += 1
    return count
