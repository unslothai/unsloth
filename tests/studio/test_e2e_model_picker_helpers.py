# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The E2E unload helper must reach the configured backend, and only when idle (#7453).

Two things the Playwright suite cannot check for itself, since running it needs a
real backend and a real GGUF:

* ``playwright.config.ts`` derives its backend URL from ``E2E_BACKEND_PORT``; the
  helper has to derive the same one, or an overridden port sends the status and
  unload calls to whatever happens to be on 8888.
* ``/api/inference/unload`` is called unforced, and the route refuses with 409
  while any conversation is still decoding (see
  ``studio/backend/tests/test_active_generations.py``). The bar test sends a
  message before unloading, so the helper must wait for the generation counter
  the gate itself reads to reach zero.

The helper is sliced verbatim into a node harness and driven against a recording
request double, so what is pinned here is the shipped source.
"""

from __future__ import annotations

import textwrap

from _node_harness import WORKDIR, read, require_node, run_harness, slice_between, source_path

HELPER = source_path("studio/frontend/e2e/helpers/model-picker.ts")

TEMP = WORKDIR / "temp" / "e2e_model_picker_helpers"

SOURCES = (HELPER,)

PRELUDE = """
// @ts-nocheck
// The only fixture the sliced helper reads through, besides process.env.
export const expect: any = (value: any) => ({
  toBeTruthy: () => {
    if (!value) throw new Error("expected a truthy value");
  },
});

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""


def _harness_source() -> str:
    """backendUrl, waitForGenerationsIdle and unloadInferenceModel, verbatim."""
    body = slice_between(read(HELPER), "const backendUrl =", "function modelSelectorTrigger(")
    return PRELUDE + body


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script)


# A request double that records every call, answers /status with one loaded model
# and drains its active-generation counter one poll at a time.
REQUEST_DOUBLE = """
    const calls: any[] = [];
    let activeCounts = [...__COUNTS__];
    const request: any = {
      get: async (url: string) => {
        if (url.includes("/active-generations")) {
          const active =
            activeCounts.length > 1 ? activeCounts.shift() : activeCounts[0];
          calls.push({ method: "GET", url, active });
          return { ok: () => true, json: async () => ({ count: active, active: [] }) };
        }
        calls.push({ method: "GET", url, active: activeCounts[0] });
        return {
          ok: () => true,
          json: async () => ({ model_identifier: "org/A-GGUF" }),
        };
      },
      post: async (url: string, _options: any) => {
        calls.push({ method: "POST", url, active: activeCounts[0] });
        return { ok: () => true };
      },
    };
"""


def _script(counts: str, env: str) -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        {env}
        process.env.E2E_ACCESS_TOKEN = "token-abc";
        const {{ unloadInferenceModel }} = await import("./harness.ts");
        {REQUEST_DOUBLE.replace("__COUNTS__", counts)}
        await unloadInferenceModel(request);
        console.log(JSON.stringify({{ calls }}));
        """
    )


def test_the_unload_helper_targets_the_configured_backend_port():
    """With E2E_BACKEND_PORT set and no explicit URL, every call must go there."""
    out = _run(
        _script(
            "[0]",
            'process.env.E2E_BACKEND_PORT = "9111";\n        '
            "delete process.env.E2E_BACKEND_URL;",
        )
    )
    calls = out.get("calls") or []
    assert calls, "the helper must talk to the backend"
    hosts = sorted({str(call.get("url", "")).split("/api/")[0] for call in calls})
    assert hosts == ["http://127.0.0.1:9111"], hosts


def test_an_explicit_backend_url_still_wins_over_the_port():
    out = _run(
        _script(
            "[0]",
            'process.env.E2E_BACKEND_PORT = "9111";\n        '
            'process.env.E2E_BACKEND_URL = "http://127.0.0.1:7000";',
        )
    )
    hosts = sorted(
        {str(call.get("url", "")).split("/api/")[0] for call in (out.get("calls") or [])}
    )
    assert hosts == ["http://127.0.0.1:7000"], hosts


def test_the_unload_waits_for_the_generation_counter_to_drain():
    """An unforced unload posted while a chat decodes comes back 409, and the helper
    asserts on response.ok(), so the bar test would fail before it ever reloads."""
    out = _run(
        _script(
            "[2, 2, 0]",
            'process.env.E2E_BACKEND_PORT = "8888";\n        '
            "delete process.env.E2E_BACKEND_URL;",
        )
    )
    calls = out.get("calls") or []
    posts = [call for call in calls if call.get("method") == "POST"]
    assert len(posts) == 1, calls
    assert str(posts[0].get("url", "")).endswith("/api/inference/unload")
    # The deciding value: how many chats were still decoding when the unload landed.
    assert posts[0].get("active") == 0, calls

    idle_polls = [call for call in calls if "/active-generations" in str(call.get("url", ""))]
    assert len(idle_polls) == 3, calls
    assert [call.get("active") for call in idle_polls] == [2, 2, 0], calls
