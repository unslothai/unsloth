# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Both builds must run the identical fixture at the same rungs, or every bridge match is spurious."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import CellFailure
from .symbols import FAILED, Bridge, build_bridge

# Must be deterministic, or the count vectors are noise.
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
    """Starts coverage once and diffs snapshots per rung; restarting would reset V8 counters."""
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
    """Arms run in sequence, not interleaved, since V8 coverage is per isolate; only integers leave."""
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
    """Requires Profiler onRender to fire non-zero: production builds compile it out, so 0.00 is broken."""
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


# Build provenance: no `/@vite/client`, react-dom `bundleType: 0` on the same renderer entry, and
# `__STUDIOBENCH_ATTRIBUTION_BUILD__`. Do not grep for `jsxDEV`: shipping Streamdown contains it.
# Install this hook stub with `add_init_script` before the first `goto` or React never injects.

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

# 0 is production, but the profiling build also reports 0, hence the separate `onRender` check.
BUNDLE_TYPE_PRODUCTION = 0
BUNDLE_TYPE_DEVELOPMENT = 1


def assert_production_bundle(page: Any, *, base_url: str | None = None) -> dict[str, Any]:
    """Raises rather than warns: a dev build does several times the work and manufactures the symptom."""
    renderers = page.evaluate("window.__studiobench_renderers || null")
    if not renderers:
        raise CellFailure(
            "no_react_renderer_seen",
            "the React DevTools hook recorded no renderer. Either React never initialised, "
            "or DEVTOOLS_HOOK_STUB_JS was installed after the first navigation instead of "
            "through add_init_script before it, in which case React had nothing to inject into.",
        )
    # Same entry: a development react-dom could pass behind a production sibling renderer.
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
    """/@vite/client must not answer 200; checked in-page, so it hits the server the app loaded from."""
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
    """Catches a stale shipping dist: a healthy Unsloth serving the wrong bundle, no profiling renderer."""
    marker = page.evaluate("globalThis.__STUDIOBENCH_ATTRIBUTION_BUILD__ === true")
    if not marker:
        raise CellFailure(
            "not_the_attribution_build",
            "__STUDIOBENCH_ATTRIBUTION_BUILD__ is not defined in the loaded bundle. The "
            "directory passed to `unsloth studio --frontend` is serving some other dist, most "
            "likely a stale shipping build. Rebuild with "
            "tests/studio/studiobench/attribution/vite.studiobench.config.ts.",
        )
    return {"attribution_build_verified": True}


def verify_build_provenance(
    page: Any,
    base_url: str,
    *,
    require_attribution: bool = True,
) -> dict[str, Any]:
    """Every gate raises: the wrong build yields plausible numbers that describe a different program."""
    out = assert_production_bundle(page, base_url = base_url)
    if require_attribution:
        out.update(assert_attribution_build(page))
    out.update(assert_profiling_build_loaded(page))
    return out
