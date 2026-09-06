# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import every routes module and inventory its real APIRouters, including hidden routes.

Run from studio/backend: python -m tests.multi_account.inventory --output ../../artifacts/route_inventory.md
"""

import argparse
import importlib
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter

import routes


@dataclass(frozen = True)
class RouteCase:
    module: str
    path: str
    method: str
    parameters: tuple[str, ...]
    router: APIRouter

    @property
    def key(self) -> str:
        return f"{self.module}:{self.method}:{self.path}"

    @property
    def object_parameters(self) -> tuple[str, ...]:
        return tuple(name for name in self.parameters if looks_like_object_id(name))


def looks_like_object_id(name: str) -> bool:
    return bool(re.search(r"(^id$|_ids?$|_uuid$|_ref$)", name)) or name in {
        "filename",
        "session",
        "name",
        "ref",
        "slug",
    }


def walk_router(router, prefix: str = ""):
    """FastAPI <=0.128 flattens includes; >=0.141 stores lazy _IncludedRouter entries."""
    for route in router.routes:
        original = getattr(route, "original_router", None)
        if original is not None:
            context = route.include_context
            yield from walk_router(original, prefix + context.prefix)
        elif hasattr(route, "path") and hasattr(route, "endpoint"):
            yield prefix + route.path, route


def collect_routes() -> tuple[RouteCase, ...]:
    found = {}
    modules = [routes]
    modules.extend(
        importlib.import_module(info.name)
        for info in sorted(
            pkgutil.walk_packages(routes.__path__, prefix = "routes."), key = lambda info: info.name
        )
    )
    seen_routers = set()
    for module in modules:
        for router in vars(module).values():
            if not isinstance(router, APIRouter) or id(router) in seen_routers:
                continue
            seen_routers.add(id(router))
            for path, route in walk_router(router):
                origin = route.endpoint.__module__
                if not origin.startswith("routes."):
                    continue
                params = tuple(re.findall(r"\{([^}:]+)(?::[^}]+)?\}", path))
                for method in sorted(getattr(route, "methods", None) or {"WEBSOCKET"}):
                    case = RouteCase(origin, path, method, params, router)
                    found.setdefault(case.key, case)
    return tuple(found[key] for key in sorted(found))


# Deliberately import-time: pytest parametrization includes routes added by another worker.
ROUTES = collect_routes()
OBJECT_ROUTES = tuple(case for case in ROUTES if case.object_parameters)


# Provisional domain numbering until the integrator supplies the other worker prompts.
WORKERS = {
    "auth": "01",
    "chat_history": "02",
    "chat_generation_runs": "02",
    "prompts": "02",
    "profile_stats": "02",
    "providers": "03",
    "provider_credentials": "03",
    "openai_codex_auth": "03",
    "rag": "04",
    "datasets": "04",
    "youtube": "04",
    "training": "05",
    "training_history": "05",
    "export": "05",
    "data_recipe": "05",
    "inference": "06",
    "llama": "06",
    "llama_compat": "06",
    "video": "06",
    "whisper": "06",
    "research_runs": "07",
    "mcp_servers": "07",
    "preview": "08",
    "settings": "08",
    "models": "08",
    "training_vram": "05",
}


def worker_for(case: RouteCase) -> str:
    return WORKERS.get(case.module.split(".")[1], "10")


def render_inventory() -> str:
    from .factories import FACTORIES

    covered = sum(case.key in FACTORIES for case in OBJECT_ROUTES)
    lines = [
        "# Studio route isolation inventory",
        "",
        "Generated from imported APIRouters under studio/backend/routes. Paths are router-relative; "
        "mount aliases outside routes (main.py, hub, picker, MCP mounts) are outside this inventory.",
        "",
        f"{len(ROUTES)} route/method pairs; {len(OBJECT_ROUTES)} have object-like parameters; "
        f"{covered} have factories; {len(OBJECT_ROUTES) - covered} are uncovered.",
        "",
        "Each object route produces five cases: owner reading Alice's resource, Alice, Bob, "
        "unauthenticated, and a deactivated Alice with a previously issued JWT. "
        "Uncovered cases fail at factory lookup under a strict worker xfail; they do not send a request. "
        "A factory means exercised, not necessarily passing. See pytest outcomes for pending behavior.",
        "",
        "Worker numbers are provisional domain assignments; confirm them against integration ownership. "
        "Worker 10 owns adding the remaining factories; domain workers own the underlying behavior.",
        "",
        "| Module | Method | Router path | Object parameters | Factory / gap | Domain worker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in ROUTES:
        coverage = (
            FACTORIES[case.key].name
            if case.key in FACTORIES
            else ("**uncovered**" if case.object_parameters else "no object-like path parameter")
        )
        lines.append(
            f"| {case.module} | {case.method} | `{case.path}` | "
            f"{', '.join(case.object_parameters) or '-'} | {coverage} | {worker_for(case)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--output", type = Path, required = True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(render_inventory(), encoding = "utf-8")


if __name__ == "__main__":
    main()
