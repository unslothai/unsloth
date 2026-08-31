# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive the two coverage runs that a symbol bridge is built from.

The bridge needs the IDENTICAL fixture executed at the same small rungs against
two different builds. That is the only requirement, and it is a strict one: if
the two ladders differ in any way that changes how many times a function runs,
the count vectors describe different experiments and every match is spurious.
`build_bridge` cannot detect that, which is why the anchor check exists.

Everything here is expressed against a `RungRunner` callable rather than against
the session harness, so this module has no dependency on how a rung is actually
driven. The caller supplies something that puts the page into rung state; this
module owns only the coverage bracketing and the ordering guarantees.

WHY THE RUNGS ARE SMALL. The bridge is a dictionary lookup, not a measurement,
so it wants exactly enough dynamic range to make count vectors unique and not
one rung more. Small rungs also keep the dev build, which is slow, from taking
minutes. Two or three rungs spanning about a decade is plenty: three counts of
a few thousand each collide far less often than one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import CellFailure
from .symbols import FAILED, Bridge, build_bridge

# A `RungRunner` puts the page into the state for one rung and returns when the
# work for that rung is finished. It must be deterministic: the same rung must
# do the same amount of work every time, or the count vectors are noise.
RungRunner = Callable[[Any, str], None]

DEFAULT_BRIDGE_RUNGS: tuple[str, ...] = ("bridge_s", "bridge_m", "bridge_l")


@dataclass
class BridgeArm:
    """One build under test: how to reach it and what it is."""

    build: str  # "dev" or "prod"
    cdp: Any
    page: Any
    react_version: str = ""
    bundle_text: str = ""
    snapshots: list[Any] = field(default_factory = list)


def collect_arm(
    arm: BridgeArm,
    runner: RungRunner,
    rungs: Sequence[str] = DEFAULT_BRIDGE_RUNGS,
    *,
    detailed: bool = False,
) -> list[Any]:
    """Run the ladder once under precise coverage, one snapshot per rung.

    Coverage is started ONCE and snapshotted per rung, and each rung's counts are
    the difference against the previous snapshot. Restarting coverage between
    rungs would reset V8's counters and also re-run `DeoptimizeAll`, which
    changes what gets compiled and therefore what gets counted.
    """
    from ..instruments.coverage import PreciseCoverage

    cov = PreciseCoverage(arm.cdp, detailed = detailed)
    cov.start()
    try:
        snapshots: list[Any] = []
        for rung in rungs:
            cov.mark()
            runner(arm.page, rung)
            snapshots.append(cov.window())
        arm.snapshots = snapshots
        return snapshots
    finally:
        cov.stop()


def build(
    dev_arm: BridgeArm,
    prod_arm: BridgeArm,
    runner: RungRunner,
    *,
    rungs: Sequence[str] = DEFAULT_BRIDGE_RUNGS,
    anchor_names: Sequence[str],
    react_url_filter: str | None = "react-dom",
    anchor_url_filter: str | None = None,
    symbols_dir: str | None = None,
) -> Bridge:
    """Collect both arms and build the bridge, persisting it if asked.

    The two arms are collected in sequence rather than interleaved, because
    interleaving would mean two coverage sessions alive at once and V8's
    coverage mode is per isolate. Sequential collection is safe here precisely
    because nothing timed is being compared: only integers cross out of this
    function, and an integer does not drift between sessions.
    """
    if dev_arm.build != "dev" or prod_arm.build != "prod":
        raise CellFailure(
            "bridge_arms_mislabelled",
            f"expected a dev arm and a prod arm, got {dev_arm.build!r} and {prod_arm.build!r}",
        )
    collect_arm(dev_arm, runner, rungs)
    collect_arm(prod_arm, runner, rungs)

    bridge = build_bridge(
        dev_arm.snapshots,
        prod_arm.snapshots,
        rungs = rungs,
        react_version = prod_arm.react_version or dev_arm.react_version,
        bundle_source = prod_arm.bundle_text,
        anchor_names = anchor_names,
        react_url_filter = react_url_filter,
        anchor_url_filter = anchor_url_filter,
    )
    if symbols_dir and bridge.status != FAILED:
        bridge.save(symbols_dir)
    return bridge


def assert_profiling_build_loaded(page: Any) -> dict[str, Any]:
    """Prove the profiling build is actually what the page loaded.

    The check is that `<Profiler>`'s `onRender` FIRES NON-ZERO. `onRender` is
    gated on React's `__PROFILE__` compile-time flag, so it does not exist in
    the production build at all: `react-dom-client.production.js` contains zero
    occurrences of the string `actualDuration`. A React stage that reads exactly
    0.00 is therefore not a fast app, it is a broken instrument, and it must
    abort the run rather than pass as a clean result.

    The page side is expected to have installed a recorder that pushes
    `onRender` arguments into `window.__studiobench_profiler`, whose entries are
    `[id, phase, actualDuration, baseDuration, startTime, commitTime]` in React
    19's six-argument order.
    """
    entries = page.evaluate("window.__studiobench_profiler || null")
    if not entries:
        raise CellFailure(
            "profiling_build_not_loaded",
            "no <Profiler> onRender callbacks were recorded. Either the profiling alias "
            "did not take effect (react-dom/client was not rewritten to react-dom/profiling) "
            "or the recorder was not installed. Either way the React stage would read 0.00, "
            "which is a broken instrument and not a fast app.",
        )
    durations = [float(e[2]) for e in entries if isinstance(e, (list, tuple)) and len(e) >= 3]
    total = sum(durations)
    if total <= 0.0:
        raise CellFailure(
            "profiling_build_reads_zero",
            f"{len(entries)} onRender callbacks fired but actualDuration summed to {total}. "
            "A React stage reading exactly 0.00 is a broken instrument; aborting rather "
            "than reporting it as a clean result.",
        )
    return {
        "onRender_callbacks": len(entries),
        "actual_duration_total_ms": total,
        "phases": sorted({str(e[1]) for e in entries if len(e) >= 2}),
        "profiling_build_verified": True,
    }


PROFILER_RECORDER_JS = """
(() => {
  // Installed via add_init_script BEFORE the app boots. Records every
  // <Profiler> onRender call so the profiling alias can be verified by
  // evidence rather than by trusting the build config.
  window.__studiobench_profiler = [];
  window.__studiobenchOnRender = function (id, phase, actualDuration,
                                           baseDuration, startTime, commitTime) {
    window.__studiobench_profiler.push(
      [id, phase, actualDuration, baseDuration, startTime, commitTime]);
  };
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# Build provenance: is the thing under measurement the thing we think it is?
# ═══════════════════════════════════════════════════════════════════════════
#
# The worst output this layer can produce is a real number from the wrong
# bundle under the right label. A dev server inflates React's own work several
# fold and would MANUFACTURE the symptom we are hunting; a stale shipping dist
# left in the output directory would silently measure code without the
# profiling renderer. Both look like a clean run.
#
# Three independent checks, because each catches something the others cannot:
#
#   1. `/@vite/client` must not answer 200. That file exists only when a Vite
#      dev server is serving, and it is a fact about the SERVER.
#   2. React must report `bundleType: 0` in the same renderer entry as
#      `rendererPackageName: "react-dom"`. That is a fact about the RENDERER
#      THAT ACTUALLY LOADED, which is the only thing that matters and the only
#      one of the three that a build-config mistake cannot fake.
#   3. `__STUDIOBENCH_ATTRIBUTION_BUILD__` must be true. That is a fact about
#      WHICH dist is mounted, and it is what catches a stale shipping build
#      sitting in the directory passed to `--frontend`.
#
# DO NOT ADD A `jsxDEV` GREP. It false-positives: `hast-util-to-jsx-runtime`,
# which Streamdown pulls in, references the dev JSX runtime in shipping code, so
# a production bundle contains the string and the check condemns a correct
# build.
#
# `bundleType` is read through the DevTools global hook. React only injects into
# that hook if it already exists when the renderer initialises, so the stub must
# be installed with `add_init_script` BEFORE the first `goto`, exactly like the
# JWT seeding in `studio_test_kit.auth`. Installed after, the hook is empty and
# the check reads as "React is missing" on a perfectly good page.

DEVTOOLS_HOOK_STUB_JS = """
(() => {
  // Minimal __REACT_DEVTOOLS_GLOBAL_HOOK__. React calls `inject()` during
  // renderer init and hands over an object carrying `bundleType` and
  // `rendererPackageName`; everything else here exists only so React's
  // instrumentation calls do not throw.
  if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) return;
  const renderers = new Map();
  let uid = 0;
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    renderers,
    supportsFiber: true,
    inject(renderer) {
      const id = ++uid;
      renderers.set(id, renderer);
      (window.__studiobench_renderers = window.__studiobench_renderers || []).push({
        bundleType: renderer && renderer.bundleType,
        version: renderer && renderer.version,
        rendererPackageName: renderer && renderer.rendererPackageName,
      });
      return id;
    },
    onCommitFiberRoot() {},
    onCommitFiberUnmount() {},
    onPostCommitFiberRoot() {},
    onScheduleFiberRoot() {},
    checkDCE() {},
  };
})();
"""

# React's `bundleType`: 0 is a production build, 1 is development. Confirmed on
# a real pair: the dev server reported 1 and the profiling build reported 0.
#
# NOTE THAT 0 DOES NOT MEAN "profiling renderer is loaded". React's profiling
# build is a production build, so it also reports 0, and `bundleType` alone
# cannot tell the two apart. That is why `assert_profiling_build_loaded` is a
# separate check on `onRender` actually firing, and why neither one substitutes
# for the other.
BUNDLE_TYPE_PRODUCTION = 0
BUNDLE_TYPE_DEVELOPMENT = 1


def assert_production_bundle(page: Any, *, base_url: str | None = None) -> dict[str, Any]:
    """Refuse to measure a development build.

    Raises `CellFailure` rather than warning. A dev-server run must be REFUSED,
    not annotated, because the numbers it produces are wrong in the direction
    that confirms the hypothesis: React's development build does several times
    the work of the shipping one, so it would manufacture exactly the symptom
    being investigated.
    """
    renderers = page.evaluate("window.__studiobench_renderers || null")
    if not renderers:
        raise CellFailure(
            "no_react_renderer_seen",
            "the React DevTools hook recorded no renderer. Either React never initialised, "
            "or DEVTOOLS_HOOK_STUB_JS was installed after the first navigation instead of "
            "through add_init_script before it, in which case React had nothing to inject into.",
        )
    # Match on the SAME entry, not across entries. A page can host more than one
    # renderer (react-dom plus react-reconciler in a canvas library, say), and
    # checking `bundleType` from one against `rendererPackageName` from another
    # is how a development react-dom passes behind a production sibling.
    dom = [r for r in renderers if str(r.get("rendererPackageName") or "") == "react-dom"]
    if not dom:
        raise CellFailure(
            "no_react_dom_renderer",
            f"no renderer identified itself as react-dom; saw "
            f"{[r.get('rendererPackageName') for r in renderers]}",
        )
    dev = [r for r in dom if r.get("bundleType") == BUNDLE_TYPE_DEVELOPMENT]
    if dev:
        raise CellFailure(
            "development_bundle",
            f"react-dom reported bundleType {BUNDLE_TYPE_DEVELOPMENT} (development), version "
            f"{dev[0].get('version')}. A development build does several times the work of the "
            "shipping one and would manufacture the symptom under investigation. Refusing.",
        )
    out: dict[str, Any] = {
        "react_dom_bundle_type": BUNDLE_TYPE_PRODUCTION,
        "react_version": str(dom[0].get("version") or ""),
        "renderers_seen": len(renderers),
        "production_bundle_verified": True,
    }
    if base_url:
        out.update(assert_not_dev_server(page, base_url))
    return out


def assert_not_dev_server(page: Any, base_url: str) -> dict[str, Any]:
    """`/@vite/client` must not answer 200.

    Checked from inside the page so it goes through the same origin and the same
    server the app was actually loaded from, rather than from Python where a
    proxy or a different host could answer.
    """
    url = base_url.rstrip("/") + "/@vite/client"
    status = page.evaluate(
        """async (u) => {
             try {
               const r = await fetch(u, { method: "GET", cache: "no-store" });
               return r.status;
             } catch (e) { return -1; }
           }""",
        url,
    )
    if status == 200:
        raise CellFailure(
            "vite_dev_server",
            f"{url} answered 200, so this Unsloth is being served by a Vite dev server. "
            "Every timing from it is inflated and the run must be refused.",
        )
    return {"vite_client_status": int(status), "dev_server_ruled_out": True}


def assert_attribution_build(page: Any) -> dict[str, Any]:
    """Confirm the dist under measurement is the studiobench attribution build.

    Catches the staleness failure: a shipping dist left in the directory handed
    to `unsloth studio --frontend <dir>` produces a perfectly healthy Unsloth
    serving the WRONG bundle, with no profiling renderer and therefore a React
    stage that reads 0.00. `attribution/vite.studiobench.config.ts` defines
    `__STUDIOBENCH_ATTRIBUTION_BUILD__` for exactly this check.
    """
    marker = page.evaluate("globalThis.__STUDIOBENCH_ATTRIBUTION_BUILD__ === true")
    if not marker:
        raise CellFailure(
            "not_the_attribution_build",
            "__STUDIOBENCH_ATTRIBUTION_BUILD__ is not defined in the loaded bundle. The "
            "directory passed to `unsloth studio --frontend` is serving some other dist, most "
            "likely a stale shipping build. Rebuild with attribution/vite.studiobench.config.ts.",
        )
    return {"attribution_build_verified": True}


def verify_build_provenance(
    page: Any,
    base_url: str,
    *,
    require_attribution: bool = True,
) -> dict[str, Any]:
    """All the provenance gates at once, for an Unsloth that is up and rendering.

    Call this once per cell before any measurement. Every failure raises, and
    that is deliberate: each of these conditions produces numbers that look
    entirely reasonable and describe a different program.
    """
    out = assert_production_bundle(page, base_url = base_url)
    if require_attribution:
        out.update(assert_attribution_build(page))
    out.update(assert_profiling_build_loaded(page))
    return out
