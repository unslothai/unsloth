# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render counts in the chat model picker's On Device tab with 600 models.

Each quant probe answer must re-render only the row it describes, and a search keystroke must not
re-render every row it leaves in place. Then the rows still work: arrow keys move focus between
them, a row's menu opens, and a click picks the model. Counts come from a React DevTools hook
installed before the page loads, using DevTools' rule for which fibers rendered (mounts are not
counted, only re-renders).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5237"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
LOCAL = 300
CACHED = 150
QUERY = "model-01"

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
})();
"""


def info(message: str) -> None:
    print(f"[model-picker-renders] {message}", flush = True)


def _local_models() -> dict:
    models = []
    for i in range(LOCAL):
        gguf = i % 2 == 0
        source = ["models_dir", "lmstudio", "ollama", "custom"][i % 4]
        models.append(
            {
                "id": f"/models/{source}/org{i % 17}/model-{i:04d}" + ("-GGUF" if gguf else ""),
                "display_name": f"Model {i:04d} {'Q4_K_M' if gguf else 'bf16'}",
                "path": f"/models/{source}/org{i % 17}/model-{i:04d}",
                "size_bytes": 1_000_000_000 + i,
                "model_format": "gguf" if gguf else "safetensors",
                "runtime": "llama_cpp" if gguf else "transformers",
                "capabilities": {"can_train": not gguf, "can_chat": True, "can_delete": True},
                "source": source,
                "model_id": f"org{i % 17}/model-{i:04d}",
                "updated_at": 1_760_000_000 + i,
                "pipeline_tag": "text-generation",
            }
        )
    return {
        "models_dir": "/models",
        "hf_cache_dir": "/hf",
        "lmstudio_dirs": ["/models/lmstudio"],
        "ollama_dirs": [],
        "models": models,
    }


def _cached(fmt: str) -> dict:
    rows = []
    for i in range(CACHED):
        repo = f"unsloth/cached-{fmt}-{i:04d}" + ("-GGUF" if fmt == "gguf" else "")
        rows.append(
            {
                "repo_id": repo,
                "load_id": repo,
                "model_format": fmt,
                "size_bytes": 2_000_000_000 + i,
                "cache_path": f"/hf/models--{repo.replace('/', '--')}",
                "last_modified": 1_760_000_000 + i,
                "pipeline_tag": "text-generation",
            }
        )
    return {"cached": rows, "scan_confirmed": True}


class MockApi:
    def __init__(self) -> None:
        self.bodies = {
            "/api/auth/status": {
                "initialized": True,
                "requires_password_change": False,
                "login_mode": "single",
                "full_access": True,
            },
            "/api/hub/local": _local_models(),
            "/api/models/local": _local_models(),
            "/api/hub/cached-gguf": _cached("gguf"),
            "/api/hub/cached-models": _cached("safetensors"),
            "/api/models/list": {"models": [], "default_models": []},
            "/api/models/loras": {"loras": [], "outputs_dir": "/out"},
            "/api/hub/hidden-models": {"hidden": []},
            "/api/chat/threads": {"threads": [], "has_more": False},
            "/api/chat/projects": {"projects": []},
            "/api/settings/pinned-models": {"pinned": []},
        }
        self.variant_probes = 0

    def handle(self, route) -> None:
        parsed = urlparse(route.request.url)
        path = parsed.path
        if path == "/api/hub/gguf-variants":
            # One downloaded quant per repo, so every GGUF row turns into a sole-quant row.
            self.variant_probes += 1
            repo = parse_qs(parsed.query).get("repo_id", ["x"])[0]
            return self._json(
                route,
                {
                    "repo_id": repo,
                    "variants": [
                        {
                            "filename": "m-Q4_K_M.gguf",
                            "quant": "Q4_K_M",
                            "size_bytes": 4_000_000_000,
                            "downloaded": True,
                            "cache_path": "/hf/x/m-Q4_K_M.gguf",
                        }
                    ],
                    "has_vision": False,
                    "default_variant": "Q4_K_M",
                    "dependencies_resolved": True,
                },
            )
        if path in self.bodies and route.request.method == "GET":
            return self._json(route, self.bodies[path])
        route.fulfill(status = 404, content_type = "application/json", body = "{}")

    @staticmethod
    def _json(route, payload: dict) -> None:
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(payload))


# 600 rows mount once, then each probe answer may re-render its own row a few times. Re-rendering
# every row per answer is tens of thousands.
SETTLE_BOUND = 3000
# Rows whose list position changes must re-render (their option id carries it): at most ~400 for
# this query. Re-rendering every remaining row per keystroke is several thousand.
TYPING_BOUND = 1500


def take_counts(page) -> dict:
    counts = page.evaluate(
        "() => { const c = window.__renders; window.__renders = null; return c; }"
    )
    # A reload mid-measure (vite re-optimizing dependencies) resets the counter to null.
    assert counts is not None, "the page reloaded while it was being measured"
    return counts


def wait_quiet(
    page,
    *,
    quiet_ms: int = 1500,
    limit_ms: int = 60_000,
) -> None:
    """Waits until no React commit has rendered a ModelRow for `quiet_ms`."""
    page.evaluate(
        """([quiet, limit]) => new Promise((resolve) => {
          const start = performance.now();
          let last = -1, since = performance.now();
          const tick = () => {
            const now = performance.now();
            const seen = window.__renders ? (window.__renders.ModelRow || 0) : 0;
            if (seen !== last) { last = seen; since = now; }
            if (now - since >= quiet || now - start >= limit) return resolve();
            setTimeout(tick, 100);
          };
          tick();
        })""",
        [quiet_ms, limit_ms],
    )


def reload(page) -> None:
    # Vite's own dependency reload can abort ours; the page still ends up loaded.
    try:
        page.reload(timeout = 120_000)
    except PlaywrightError as error:
        if "ERR_ABORTED" not in str(error):
            raise
        page.wait_for_load_state("load", timeout = 120_000)


def open_on_device(page) -> None:
    page.get_by_text("Select model").first.click()
    page.get_by_text("On Device").first.click()
    page.wait_for_function("() => /cached-gguf-0000/.test(document.body.innerText)", timeout = 60_000)


def main() -> int:
    proc = None
    try:
        if not _EXTERNAL:
            proc = start_vite(PORT)
            wait_for_smoke_page(f"{BASE}/chat", "/src/main.tsx", proc = proc, info = info)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless = True, args = chromium_launch_args())
            try:
                api = MockApi()
                context = browser.new_context(viewport = {"width": 1280, "height": 900})
                context.add_init_script(
                    "localStorage.setItem('unsloth_auth_token','x.eyJleHAiOjk5OTk5OTk5OTl9.y');"
                    "localStorage.setItem('unsloth_auth_refresh_token','r');"
                )
                context.add_init_script(RENDER_COUNTER)
                context.route(lambda url: urlparse(url).path.startswith("/api/"), api.handle)
                context.route(
                    "https://huggingface.co/**",
                    lambda route: route.fulfill(
                        status = 200, content_type = "application/json", body = "[]"
                    ),
                )
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))

                # Warm-up: one full pass lets the dev server finish optimizing dependencies, whose
                # reload would otherwise land in the middle of a measurement.
                page.goto(f"{BASE}/chat", timeout = 120_000)
                for attempt in range(4):
                    try:
                        page.get_by_text("Select model").first.wait_for(timeout = 30_000)
                        break
                    except Exception:
                        # A first load can lose a lazy section to that reload; load it again.
                        if attempt == 3:
                            raise
                        reload(page)
                open_on_device(page)
                page.wait_for_timeout(3000)
                reload(page)
                page.get_by_text("Select model").first.wait_for(timeout = 120_000)
                page.wait_for_timeout(2000)
                api.variant_probes = 0

                page.evaluate("() => { window.__renders = {}; }")
                open_on_device(page)
                wait_quiet(page)
                settle = take_counts(page)
                rows = page.evaluate(
                    "() => document.querySelectorAll('[data-model-picker-option]').length"
                )

                # The probes landed: GGUF rows carry their one quant as a chip. A loaded runner can
                # paint the last chips after the render counts went quiet, so wait for them first.
                chip_count = "() => (document.body.innerText.match(/\\bQ4_K_M\\b/g) || []).length"
                with contextlib.suppress(PlaywrightTimeoutError):
                    page.wait_for_function(f"({chip_count})() >= {CACHED}", timeout = 30_000)
                chips = page.evaluate(chip_count)
                search = page.get_by_placeholder("Search").first
                search.focus()
                page.evaluate("() => { window.__renders = {}; }")
                for char in QUERY:
                    page.keyboard.type(char)
                    page.wait_for_timeout(150)
                wait_quiet(page, quiet_ms = 1000)
                typing = take_counts(page)
                matches = page.evaluate(
                    "() => document.querySelectorAll('[data-model-picker-option]').length"
                )
                info(
                    json.dumps(
                        {
                            "rows": rows,
                            "variant_probes": api.variant_probes,
                            "settle_ModelRow": settle.get("ModelRow", 0),
                            "settle_ModelRowMenu": settle.get("ModelRowMenu", 0),
                            "typing_ModelRow": typing.get("ModelRow", 0),
                            "typing_ModelRowMenu": typing.get("ModelRowMenu", 0),
                            "matches": matches,
                            "quant_chips": chips,
                        }
                    )
                )
                # Fails the run, not the bound, when nothing was there to measure.
                assert rows >= 2 * LOCAL, f"only {rows} On Device rows rendered"
                assert (
                    api.variant_probes >= CACHED
                ), f"only {api.variant_probes} quant probes ran, so the settle measured little"
                assert chips >= CACHED, f"only {chips} GGUF rows show their sole quant"
                assert 0 < matches < rows, f"the search left {matches} of {rows} rows"

                # The rows still work: arrows move focus and the active marker, a menu opens.
                options = page.locator("[data-model-picker-option]")
                options.nth(0).focus()
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(200)
                second_id = options.nth(1).get_attribute("id")
                focused = page.evaluate("() => document.activeElement && document.activeElement.id")
                assert focused == second_id, f"ArrowDown focused {focused}, not {second_id}"
                assert options.nth(1).get_attribute("data-model-picker-active-option") == "true"
                page.keyboard.press("ArrowUp")
                page.wait_for_timeout(200)
                first_id = options.nth(0).get_attribute("id")
                focused = page.evaluate("() => document.activeElement && document.activeElement.id")
                assert focused == first_id, f"ArrowUp focused {focused}, not {first_id}"
                page.get_by_role("button", name = re.compile(r"^More options for ")).first.click()
                page.get_by_role("menuitem").first.wait_for(timeout = 10_000)
                page.keyboard.press("Escape")
                page.wait_for_timeout(300)
            finally:
                browser.close()
        # Endpoints this check does not mock answer 404, which the app reports and moves past.
        unexpected = [error for error in errors if "404" not in error]
        assert not unexpected, unexpected
        problems = []
        if settle.get("ModelRow", 0) > SETTLE_BOUND:
            problems.append(
                f"ModelRow re-rendered {settle['ModelRow']} times while {api.variant_probes} "
                f"quant probes settled (bound {SETTLE_BOUND}): each answer re-renders every row"
            )
        if typing.get("ModelRow", 0) > TYPING_BOUND:
            problems.append(
                f"ModelRow re-rendered {typing['ModelRow']} times while typing {QUERY!r} "
                f"(bound {TYPING_BOUND}): each keystroke re-renders every row"
            )
        assert not problems, "; ".join(problems)
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
