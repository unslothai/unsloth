# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The first-boot setup effect must spend the single-use token exactly once.

``src/main.tsx`` wraps the app in ``<StrictMode>``, and React runs one extra
setup/cleanup/setup cycle per effect in development. The setup token the page
carries is SINGLE USE, so an effect that exchanges it on every setup burns it on
the first run and gets a 401 on the second; because the first run's result is
discarded by its own cleanup, the component then holds an error and no session,
and the submit-time retry re-exchanges the same dead token. First-boot setup
becomes unusable under ``npm run dev`` and no reload can recover it, because the
fresh token the reload mints is double-exchanged in exactly the same way.

A ref claimed during setup covers the StrictMode replay, because a replay keeps
the same component instance. It does NOT cover a genuine remount: the form's own
"Back to login" link goes to /login, which bounces straight back while
must_change_password is set, and the new instance gets a fresh ref, re-exchanges
the same token out of the same HTML and lands on a 401 that only a full page load
clears. So the exchange is keyed by the token at module scope instead, which
covers the replay, the remount and a submit racing the in-flight request with one
mechanism.

Following tests/studio/_node_harness.py's approach: the real source is sliced
verbatim and run under node, and only the things it reads through are stubbed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUTH_FORM = REPO / "studio/frontend/src/features/auth/components/auth-form.tsx"

EFFECT_ANCHOR = "const token = window.__UNSLOTH_BOOTSTRAP__?.link_token;"


def _node_or_skip() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    probe = subprocess.run(
        [node, "--experimental-strip-types", "--version"],
        capture_output = True,
    )
    if probe.returncode != 0:
        pytest.skip("node does not support --experimental-strip-types")
    return node


def _slice_setup_effect(src: str) -> str:
    """The `useEffect(() => {...}, [])` that redeems the injected setup token."""
    anchor = src.find(EFFECT_ANCHOR)
    assert anchor != -1, (
        "the setup-token exchange effect is gone; first boot no longer redeems "
        f"the injected token (looked for {EFFECT_ANCHOR!r})"
    )
    start = src.rfind("useEffect(", 0, anchor)
    assert start != -1
    end = src.index("}, []);", anchor) + len("}, []);")
    return src[start:end]


def _slice_setup_exchange_cache(src: str) -> str:
    """The module-scope `setupExchanges` map and `startSetupExchange` helper.

    Sliced rather than stubbed: this IS the deduplication under test, so a
    harness that reimplemented it would pass while the shipped code regressed.
    """
    start = src.index("const setupExchanges = new Map")
    end = src.index("\n}", src.index("function startSetupExchange")) + len("\n}")
    sliced = src[start:end]
    # The harness runs plain JS (node -e, no --experimental-strip-types), so drop
    # the TypeScript annotations this slice uses: the Map generic and the one
    # function's parameter/return types.
    sliced = re.sub(r"new Map<[^>]*>+\(\)", "new Map()", sliced)
    sliced = re.sub(
        r"function startSetupExchange\([^)]*\)[^{]*\{",
        "function startSetupExchange(linkToken) {",
        sliced,
    )
    sliced = re.sub(r"\blet inFlight[^=]*=", "let inFlight =", sliced)
    return sliced


def _slice_refs(src: str) -> str:
    """Every `const <name> = useRef(<literal>);` the component declares."""
    return "\n".join(
        f"const {name} = useRef({init});"
        for name, init in re.findall(r"const (\w+) = useRef\(([^)]*)\);", src)
    )


def _run_effect_harness(*, strict_mode: bool, remount: bool = False) -> dict:
    node = _node_or_skip()
    src = AUTH_FORM.read_text(encoding = "utf-8")

    harness = textwrap.dedent(
        """
        // Minimal React shim: enough to run one component's effect the way React
        // does, including StrictMode's extra setup/cleanup/setup cycle.
        const useRef = (initial) => ({ current: initial });

        let exchangeCalls = 0;
        let tokenBurned = false;
        let setupSession = null;
        let setupError = null;
        const setSetupSession = (v) => { setupSession = v; };
        const setSetupError = (v) => { setupError = v; };
        const isLoginMode = false;

        globalThis.window = { __UNSLOTH_BOOTSTRAP__: { username: "unsloth", link_token: "T" } };

        // The real backend: /api/auth/link-exchange consumes the nonce, so the
        // second exchange of one token is a 401 no matter who sends it.
        async function exchangeSetupToken(linkToken) {
          exchangeCalls += 1;
          if (tokenBurned || linkToken !== "T") return { access: null, status: 401 };
          tokenBurned = true;
          return { access: "ACCESS", status: 200 };
        }

        __CACHE__

        // One "component instance" per call: its refs are created here, so a
        // guard living in the instance does NOT survive a remount, exactly as in
        // React. Declaring the refs once at module scope instead would let a
        // per-instance ref pass the remount test it is supposed to fail.
        function mount() {
          __REFS__
          return __EFFECT__;
        }

        const pending = [];
        const origVoid = (p) => pending.push(p);

        async function run() {
          const setup = mount();
          const cleanup1 = setup();
          if (__STRICT__) {
            // StrictMode replays setup/cleanup/setup on the SAME instance, so
            // the same refs are still in scope.
            if (typeof cleanup1 === "function") cleanup1();
            const cleanup2 = setup();
            void cleanup2;
          }
          if (__REMOUNT__) {
            // A route change away and back destroys the instance: fresh refs,
            // fresh state, same page and same HTML.
            if (typeof cleanup1 === "function") cleanup1();
            for (let i = 0; i < 20; i += 1) await Promise.resolve();
            setupSession = null;
            setupError = null;
            const remounted = mount();
            const cleanup3 = remounted();
            void cleanup3;
          }
          // Let every scheduled exchange settle.
          for (let i = 0; i < 50; i += 1) await Promise.resolve();
          await new Promise((r) => setTimeout(r, 25));
          console.log(JSON.stringify({
            exchangeCalls, setupSession, setupError,
          }));
        }
        void origVoid;
        run();
        """
    )

    effect_src = _slice_setup_effect(src)
    # `useEffect(<arrow>, [])` -> the arrow itself, so the harness can call it.
    arrow = effect_src[len("useEffect(") : effect_src.rindex(", []);")]

    script = (
        harness.replace("__CACHE__", _slice_setup_exchange_cache(src))
        .replace("__REFS__", _slice_refs(src))
        .replace("__EFFECT__", arrow)
        .replace("__STRICT__", "true" if strict_mode else "false")
        .replace("__REMOUNT__", "true" if remount else "false")
    )

    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_single_mount_exchanges_the_token_once():
    result = _run_effect_harness(strict_mode = False)
    assert result["exchangeCalls"] == 1
    assert result["setupSession"] == "ACCESS"
    assert result["setupError"] is None


def test_a_strictmode_replay_does_not_burn_the_single_use_token():
    """setup, cleanup, setup must still leave the component with a session.

    Without a ref claimed during setup this exchanges twice: the first call
    burns the token and has its result thrown away by the cleanup, the second
    gets a 401, and the operator is left on an error they cannot reload out of.
    """
    result = _run_effect_harness(strict_mode = True)
    assert (
        result["exchangeCalls"] == 1
    ), "the setup token was exchanged more than once across a StrictMode replay, which burns it"
    assert result["setupSession"] == "ACCESS"
    assert result["setupError"] is None


def test_a_remount_reuses_the_exchange_instead_of_respending_the_token():
    """The case a per-instance ref cannot cover.

    The setup form renders a "Back to login" link. /login bounces straight back
    while must_change_password is set, so the component is destroyed and rebuilt
    with the SAME token still sitting in the page's HTML. A guard living in the
    instance is gone by then; the exchange has to outlive it.
    """
    result = _run_effect_harness(strict_mode = False, remount = True)
    assert result["exchangeCalls"] == 1, (
        "the setup token was exchanged again after a remount, so a click on "
        "'Back to login' leaves first boot stuck on a 401 until a full reload"
    )
    assert result["setupSession"] == "ACCESS"
    assert result["setupError"] is None


def test_the_exchange_is_keyed_at_module_scope_not_in_the_component():
    """Source contract: the guard must outlive the component instance."""
    src = AUTH_FORM.read_text(encoding = "utf-8")
    effect = _slice_setup_effect(src)
    assert "startSetupExchange" in effect, (
        "the setup effect calls exchangeSetupToken directly again, so every "
        "mount spends the single-use token afresh"
    )
    cache = _slice_setup_exchange_cache(src)
    assert "new Map" in cache and "setupExchanges.set" in cache, (
        "startSetupExchange no longer memoises by token, so a remount or a "
        "racing submit exchanges twice"
    )
    # And the submit-time retry must join it rather than start its own.
    assert src.count("startSetupExchange(") >= 3, (
        "the submit-time retry still calls exchangeSetupToken directly, so a "
        "submit during the in-flight mount exchange burns the token"
    )
