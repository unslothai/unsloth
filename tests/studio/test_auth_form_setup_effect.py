# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The first-boot setup effect must survive a StrictMode effect replay.

``src/main.tsx`` wraps the app in ``<StrictMode>``, and React runs one extra
setup/cleanup/setup cycle per effect in development. The setup token the page
carries is SINGLE USE, so an effect that exchanges it on every setup burns it on
the first run and gets a 401 on the second; because the first run's result is
discarded by its own cleanup, the component then holds an error and no session,
and the submit-time retry re-exchanges the same dead token. First-boot setup
becomes unusable under ``npm run dev`` and no reload can recover it, because the
fresh token the reload mints is double-exchanged in exactly the same way.

The rest of this codebase already defends this exact case with a ref claimed
during setup rather than a local cancelled flag (new-project-dialog.tsx,
find-bar.tsx, pickers.tsx, model-config-page.tsx, prompt-storage-dialog.tsx).

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


def _slice_refs(src: str) -> str:
    """Every `const <name> = useRef(<literal>);` the component declares."""
    return "\n".join(
        f"const {name} = useRef({init});"
        for name, init in re.findall(r"const (\w+) = useRef\(([^)]*)\);", src)
    )


def _run_effect_harness(*, strict_mode: bool) -> dict:
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

        __REFS__

        const setup = __EFFECT__;

        const pending = [];
        const origVoid = (p) => pending.push(p);

        async function run() {
          const cleanup1 = setup();
          if (__STRICT__) {
            if (typeof cleanup1 === "function") cleanup1();
            const cleanup2 = setup();
            void cleanup2;
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
        harness.replace("__REFS__", _slice_refs(src))
        .replace("__EFFECT__", arrow)
        .replace("__STRICT__", "true" if strict_mode else "false")
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
    assert result["exchangeCalls"] == 1, (
        "the setup token was exchanged more than once across a StrictMode "
        "replay, which burns it; claim a ref during setup like the rest of the "
        "codebase does rather than relying on a local cancelled flag"
    )
    assert result["setupSession"] == "ACCESS"
    assert result["setupError"] is None


def test_the_effect_guards_with_a_ref_not_only_a_local_flag():
    """Source contract, so the guard cannot be removed without a failure here."""
    effect = _slice_setup_effect(AUTH_FORM.read_text(encoding = "utf-8"))
    refs = re.findall(r"const (\w+) = useRef\(", AUTH_FORM.read_text(encoding = "utf-8"))
    assert any(ref in effect for ref in refs), (
        "the setup-token effect guards only with a local variable, so a "
        "StrictMode replay re-runs it and burns the single-use token"
    )
