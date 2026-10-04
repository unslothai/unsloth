# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat eject must show a toast on the first click and refuse a second (#10339)."""

from __future__ import annotations

import textwrap

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

HOOK = source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")
CONFIRM = source_path("studio/frontend/src/features/chat/utils/confirm-stop-running-chats.ts")
TEMP = WORKDIR / "temp" / "chat_eject_in_flight"

EJECT_HARNESS = """
// @ts-nocheck
export const world: any = {
  toasts: [] as { kind: string; title: string }[],
  unloads: 0,
  beginCalls: 0,
};

let leasePhase: string | null = null;
const chatModelLifecycleGate = {
  currentPhase() { return leasePhase; },
};
const params = { checkpoint: "org/model" };
function setModelsError(_message: string | null): void {}
function clearCheckpoint(): void {}
async function refresh(): Promise<void> {}
function isExternalModelId(_id: string): boolean { return false; }
function stopQueuedRuns(_decision: unknown, _scoped: boolean): void {}
function requestLocalPromptQueueStop(_ids?: unknown): void {}
async function unloadModel(_payload: unknown): Promise<void> {
  world.unloads += 1;
}

let confirmGate: Promise<void> | null = null;
let releaseConfirm: (() => void) | null = null;
export function hangNextConfirm(): void {
  confirmGate = new Promise((resolve) => {
    releaseConfirm = resolve;
  });
}
export function releaseHungConfirm(): void {
  releaseConfirm?.();
  releaseConfirm = null;
  confirmGate = null;
}

async function confirmStopRunningChatsIfNeeded(
  _action: string,
  _effect: string,
  _model?: string,
): Promise<any> {
  if (confirmGate) await confirmGate;
  return {
    proceed: true,
    forceCancelActive: false,
    promptQueueThreadIds: [],
    preStreamRunTokens: [],
  };
}

const toast = {
  info(title: string) {
    world.toasts.push({ kind: "info", title });
  },
  loading(title: string) {
    world.toasts.push({ kind: "loading", title });
    return "toast-1";
  },
  dismiss(_id: unknown) {
    world.toasts.push({ kind: "dismiss", title: "" });
  },
  promise(pending: Promise<unknown>) {
    return pending;
  },
};

const useChatRuntimeStore = {
  getState() {
    return {
      get modelLoading() { return leasePhase !== null; },
      loadingModelPick: null,
      loadedModels: [{ checkpoint: "org/model" }],
      beginModelLoading(phase: string) {
        world.beginCalls += 1;
        if (leasePhase !== null) return null;
        leasePhase = phase;
        return 1;
      },
      endModelLoading(_lease: unknown) {
        leasePhase = null;
      },
    };
  },
};

async function ejectModel(confirmed?: any): Promise<boolean> {
__EJECT_BODY__
}

export { ejectModel };
"""

CONFIRM_HARNESS = """
// @ts-nocheck
export const world: any = {
  hadSignal: false,
  disposed: 0,
  timeoutMs: 0,
};

const ACTIVE_GENERATIONS_TIMEOUT_MS = 8_000;

function disposableTimeoutSignal(ms: number) {
  world.timeoutMs = ms;
  const controller = new AbortController();
  return {
    signal: controller.signal,
    dispose() {
      world.disposed += 1;
    },
  };
}

async function getActiveGenerations(_model?: string, signal?: AbortSignal) {
  world.hadSignal = signal instanceof AbortSignal;
  return { count: 0, thread_ids: [], active: [] };
}

const useChatRuntimeStore = {
  getState() {
    return { runningByThreadId: {}, localRunByThreadId: {} };
  },
};
const usePromptQueueUI = {
  getState() {
    return { byThreadId: {} };
  },
};
function listLocalPreStreamRunReservations() { return []; }
function getLocalPromptQueueThreadIds() { return []; }
const useStopRunningChatsDialogStore = {
  getState() { return { requestConfirm: async () => true }; },
};
async function listStoredChatThreads() { return []; }

__CONFIRM__
"""


def _eject_body() -> str:
    return slice_between(
        read(HOOK),
        "    if (!params.checkpoint) {\n      return false;\n    }",
        "  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);",
    )


def _confirm_source() -> str:
    text = read(CONFIRM)
    start = "export async function confirmStopRunningChatsIfNeeded("
    return text[text.index(start) :]


def _run_eject(script: str) -> dict:
    require_node((HOOK,))
    return run_harness(
        TEMP,
        EJECT_HARNESS.replace("__EJECT_BODY__", _eject_body()),
        script,
        sources = (),
    )


def _run_confirm(script: str) -> dict:
    require_node((CONFIRM,))
    return run_harness(
        TEMP / "confirm",
        CONFIRM_HARNESS.replace("__CONFIRM__", _confirm_source()),
        script,
        sources = (),
    )


def test_the_first_eject_click_shows_a_loading_toast_before_confirm():
    out = _run_eject(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { ejectModel, hangNextConfirm, releaseHungConfirm, world } from "./harness.ts";
            hangNextConfirm();
            const first = ejectModel();
            await new Promise((resolve) => setTimeout(resolve, 20));
            const loading = world.toasts.filter((t) => t.kind === "loading").map((t) => t.title);
            const second = await ejectModel();
            releaseHungConfirm();
            const firstResult = await first;
            console.log(JSON.stringify({
              loading,
              second,
              firstResult,
              unloads: world.unloads,
              info: world.toasts.filter((t) => t.kind === "info").map((t) => t.title),
              beginCalls: world.beginCalls,
            }));
            """
        )
    )
    assert out["loading"] == [
        "Unloading model"
    ], "the first click must toast before confirmStopRunningChatsIfNeeded returns"
    assert out["second"] is False
    assert out["firstResult"] is True
    assert out["unloads"] == 1
    assert out["beginCalls"] == 1
    assert out["info"] == ["Wait for the model to finish unloading."]


def test_the_active_generations_snapshot_passes_a_timeout_signal():
    assert "ACTIVE_GENERATIONS_TIMEOUT_MS = 8_000" in read(CONFIRM)
    assert "getActiveGenerations(model, timeout.signal)" in read(CONFIRM)
    out = _run_confirm(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { confirmStopRunningChatsIfNeeded, world } from "./harness.ts";
            const decision = await confirmStopRunningChatsIfNeeded("Unloading the model", "unload");
            console.log(JSON.stringify({
              proceed: decision.proceed,
              hadSignal: world.hadSignal,
              disposed: world.disposed,
              timeoutMs: world.timeoutMs,
            }));
            """
        )
    )
    assert out["proceed"] is True
    assert out["hadSignal"] is True
    assert out["disposed"] == 1
    assert out["timeoutMs"] == 8000
