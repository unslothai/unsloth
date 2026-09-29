# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source contracts for generalized compare cancellation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "studio" / "frontend" / "src" / "features" / "chat"


def _read(name: str) -> str:
    return (CHAT / name).read_text(encoding = "utf-8")


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_stop_unloads_only_a_load_already_sent():
    composer = _read("shared-composer.tsx")
    stop = _between(composer, "  function stop() {", "\n  }\n")
    assert "compareRunsRef.current.cancelCurrent()" in stop
    assert "cancel_load_request_id: run.loadingModel.requestId" in stop
    load = _between(composer, "const resp = await loadModel(", "saveSpeculativeType")
    assert "signal: compareSignal" in load
    start_hook = _between(load, "onRequestStart: () => {", "},")
    assert "requestId: loadRequestId" in start_hook
    assert "load_request_id: loadRequestId" in load
    assert load.index("setLoadingModel(run, null)") > load.index("onRequestStart")


def test_stopped_compare_never_starts_the_next_step():
    composer = _read("shared-composer.tsx")
    sequence = _between(composer, "setComparing(true);", "} catch (err) {")
    steps = [
        "ensureModelLoaded(model1);",
        "handle1.startRun();",
        "ensureModelLoaded(model2);",
        "handle2.startRun();",
        "compareStepSucceededRef.current = true;",
    ]
    positions = [sequence.index(step) for step in steps]
    for before, after in zip(positions, positions[1:]):
        assert "throwIfCompareCancelled(compareSignal);" in sequence[before:after]


def test_cancelled_load_waits_for_unload_then_reconciles_checkpoint():
    composer = _read("shared-composer.tsx")
    catch = _between(composer, "} catch (err) {\n        compareStepSucceededRef", "} finally {")
    assert catch.index("await run.cleanup") < catch.index(
        "await resyncInferenceStatusAfterServerModelChange()"
    )
    assert 'toast.info("Compare stopped"' in catch
    finally_ = _between(composer, "      } finally {\n        compareRunsRef", "\n      }\n")
    assert "onComparingChange?.(false);" in finally_


def test_generalized_compare_does_not_relist_threads_mid_send():
    page = _read("chat-page.tsx")
    general = _between(page, "const GeneralCompareContent = memo(", "return (")
    assert "(anyRunning || comparing) && listedPairRef.current === pairId" in general
    assert "onComparingChange={setComparing}" in page


def test_auth_layer_carries_no_compare_lifecycle():
    auth = (ROOT / "studio" / "frontend" / "src" / "features" / "auth" / "api.ts").read_text(
        encoding = "utf-8"
    )
    assert "onRequestStart" not in auth
    assert "onAuthenticationRequired" not in auth
