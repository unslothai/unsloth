# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render counts while one node is dragged across a large Recipe Studio graph.

A drag step moves one node, so it must not re-render the graph's other nodes. Counts come from a
React DevTools hook installed before the page loads, using DevTools' rule for which fibers
rendered. The recipe API is answered from the page, so no backend runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5218"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
PAIRS = 40  # a sampler and an LLM column each, plus the provider and model nodes
STEPS = 12
NODE_COMPONENTS = ("RecipeGraphNodeBase", "AuxNodeBase")

RENDER_COUNTER = r"""
(() => {
  window.__renders = null;
  const nameOf = (fiber) => {
    const type = fiber.type;
    if (!type) return null;
    if (fiber.tag === 0 || fiber.tag === 1 || fiber.tag === 15) return type.displayName || type.name || null;
    if (fiber.tag === 11) return type.displayName || type.render?.displayName || type.render?.name || null;
    if (fiber.tag === 14) return type.displayName || type.type?.displayName || type.type?.name || null;
    return null;
  };
  const walk = (fiber, counts) => {
    // Children that were not cloned this commit were not visited by React.
    if (fiber.alternate && fiber.alternate.child === fiber.child) return;
    for (let child = fiber.child; child; child = child.sibling) {
      const name = nameOf(child);
      // PerformedWork (1) is set only on fibers whose component function ran this commit.
      if (name && child.alternate && (child.flags & 1)) counts[name] = (counts[name] || 0) + 1;
      walk(child, counts);
    }
  };
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    supportsFiber: true,
    isDisabled: false,
    renderers: new Map(),
    inject(renderer) { const id = this.renderers.size + 1; this.renderers.set(id, renderer); return id; },
    checkDCE() {},
    onScheduleFiberRoot() {},
    onCommitFiberUnmount() {},
    onPostCommitFiberRoot() {},
    onCommitFiberRoot(_id, root) {
      if (window.__renders) walk(root.current, window.__renders);
    },
  };
  localStorage.setItem("unsloth_auth_token", "x.eyJleHAiOjk5OTk5OTk5OTl9.y");
  localStorage.setItem("unsloth_auth_refresh_token", "r");
})();
"""

AUTH_STATUS = {
    "initialized": True,
    "requires_password_change": False,
    "login_mode": "single",
    "full_access": True,
}


def info(message: str) -> None:
    print(f"[recipe-drag-renders] {message}", flush = True)


def big_recipe(pairs: int) -> dict:
    columns, nodes, edges = [], [], []
    nodes.append({"id": "vllm", "x": 0, "y": 0, "width": 400})
    nodes.append({"id": "m1", "x": 0, "y": 200, "width": 400})
    for i in range(pairs):
        sampler, llm = f"s{i}", f"l{i}"
        previous = f" {{{{ l{i - 1} }}}}" if i else ""
        columns.append(
            {
                "column_type": "sampler",
                "name": sampler,
                "drop": True,
                "sampler_type": "category",
                "params": {"values": ["a", "b", "c"]},
            }
        )
        columns.append(
            {
                "column_type": "llm-text",
                "name": llm,
                "drop": False,
                "model_alias": "m1",
                "prompt": f"Write about {{{{ {sampler} }}}}{previous}",
                "system_prompt": "",
                "with_trace": "none",
                "extract_reasoning_content": False,
            }
        )
        nodes.append({"id": sampler, "x": 500 + (i % 10) * 900, "y": (i // 10) * 500, "width": 400})
        nodes.append(
            {"id": llm, "x": 950 + (i % 10) * 900, "y": (i // 10) * 500 + 150, "width": 400}
        )
        edge = {"type": "canvas", "source_handle": "data-out", "target_handle": "data-in"}
        edges.append({"from": sampler, "to": llm, **edge})
        if i:
            edges.append({"from": f"l{i - 1}", "to": llm, **edge})
    payload = {
        "recipe": {
            "model_providers": [
                {
                    "name": "vllm",
                    "endpoint": "",
                    "provider_type": "openai",
                    "extra_headers": {},
                    "extra_body": {},
                }
            ],
            "mcp_providers": [],
            "model_configs": [
                {
                    "alias": "m1",
                    "model": "",
                    "provider": "vllm",
                    "inference_parameters": {"temperature": 0.7},
                }
            ],
            "tool_configs": [],
            "columns": columns,
            "processors": [],
        },
        "run": {"rows": 5, "preview": True, "output_formats": ["jsonl"]},
        "ui": {"nodes": nodes, "edges": edges, "layout_direction": "LR"},
    }
    return {"id": "big", "name": "Big recipe", "payload": payload, "createdAt": 1, "updatedAt": 2}


def handle_api(recipe: dict):
    answers = {
        "/api/auth/status": AUTH_STATUS,
        "/api/data-recipe/recipes": {"recipes": [recipe]},
        "/api/data-recipe/recipes/big": recipe,
        "/api/data-recipe/recipes/big/executions": {"executions": []},
    }

    def handler(route) -> None:
        path = urlsplit(route.request.url).path
        body = answers.get(path) if route.request.method == "GET" else None
        if body is None:
            route.fulfill(status = 404, content_type = "application/json", body = "{}")
        else:
            route.fulfill(status = 200, content_type = "application/json", body = json.dumps(body))

    return handler


def run(page, expected_nodes: int) -> dict[str, int]:
    # A cold dev server compiles the app on first sight and may reload it once while it
    # re-optimizes dependencies, so a first load that does not settle is retried.
    for attempt in range(3):
        try:
            page.goto(f"{BASE}/data-recipes/big", wait_until = "domcontentloaded", timeout = 120_000)
            page.wait_for_function(
                "(n) => document.querySelectorAll('.react-flow__node').length >= n",
                arg = expected_nodes,
                timeout = 120_000,
            )
            break
        except Exception as error:
            if attempt == 2:
                raise
            info(f"graph did not load ({type(error).__name__}), retrying")
    page.wait_for_timeout(1500)
    box = page.evaluate(
        """() => {
          for (const node of document.querySelectorAll('.react-flow__node')) {
            const r = node.getBoundingClientRect();
            if (r.x > 320 && r.x < 1100 && r.y > 120 && r.y < 700 && r.height < 80) {
              return { x: r.x, y: r.y, id: node.dataset.id, transform: node.style.transform };
            }
          }
          return null;
        }"""
    )
    assert box, "no draggable node in view"
    x, y = box["x"] + 60, box["y"] + 6
    page.evaluate("() => { window.__renders = {}; }")
    page.mouse.move(x, y)
    page.mouse.down()
    for step in range(STEPS):
        page.mouse.move(x + 6 * (step + 1), y + 3 * (step + 1))
        page.wait_for_timeout(30)
    page.mouse.up()
    page.wait_for_timeout(500)
    counts = page.evaluate(
        "() => { const c = window.__renders; window.__renders = null; return c; }"
    )
    moved = page.evaluate(
        '(id) => document.querySelector(`.react-flow__node[data-id="${id}"]`)?.style.transform',
        box["id"],
    )
    # Fails the run, not the bound, when the drag never moved the node.
    assert moved and moved != box["transform"], f"the drag did not move node {box['id']}"
    return counts


def main() -> int:
    recipe = big_recipe(PAIRS)
    expected_nodes = 2 + 2 * PAIRS
    proc = None
    try:
        if not _EXTERNAL:
            proc = start_vite(PORT)
            wait_for_smoke_page(f"{BASE}/data-recipes", "main.tsx", proc = proc, info = info)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless = True, args = chromium_launch_args())
            try:
                context = browser.new_context(viewport = {"width": 1440, "height": 900})
                context.add_init_script(RENDER_COUNTER)
                # Only the backend's /api root: the dev server serves source modules under .../api/ too.
                context.route(
                    lambda url: urlsplit(url).path.startswith("/api/"), handle_api(recipe)
                )
                page = context.new_page()
                errors: list[str] = []
                # Endpoints this page does not need answer 404, which the app reports this way.
                page.on(
                    "pageerror",
                    lambda error: None
                    if str(error).startswith("Request failed (404)")
                    else errors.append(str(error)),
                )
                counts = run(page, expected_nodes)
            finally:
                browser.close()
        nodes = sum(counts.get(name, 0) for name in NODE_COMPONENTS)
        info(json.dumps({"node renders": nodes, "nodes": expected_nodes, "steps": STEPS}))
        assert not errors, errors
        # The dragged node may render once per step; the other nodes must stay put.
        assert nodes <= 2 * STEPS + expected_nodes // 4, (
            f"nodes rendered {nodes} times over a {STEPS}-step drag of one node in a "
            f"{expected_nodes}-node graph: a drag step is re-rendering every node"
        )
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
