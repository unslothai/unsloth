# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A second chat eject while confirm is open must not toast as a load (#10339).

The red-circle path lives in ``useChatModelRuntime`` three times (chat, hub, hub
gear), so the in-flight flag is module-scoped. This replays the callback body
verbatim with confirm hanging, then fires a second eject.
"""

from __future__ import annotations

import textwrap

import pytest

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

HOOK = source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")
TEMP = WORKDIR / "temp" / "chat_eject_in_flight"

HARNESS = """
// @ts-nocheck
export const world: any = {
  toasts: [] as { title: string; description?: string }[],
  unloads: 0,
  beginCalls: 0,
  lease: { id: "lease-1" } as any,
};

let chatEjectInFlight = false;
const params = { checkpoint: "org/model" };
function setModelsError(_message: string | null): void {}
function clearCheckpoint(): void {}
async function refresh(): Promise<void> {}
function isExternalModelId(_id: string): boolean { return false; }
function cancelPreStreamRunReservations(_tokens: unknown): void {}
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
  info(title: string, options?: { description?: string }) {
    world.toasts.push({ title, description: options?.description });
  },
  promise(pending: Promise<unknown>) {
    return pending;
  },
};

const useChatRuntimeStore = {
  getState() {
    return {
      modelLoading: false,
      loadingModelPick: null,
      beginModelLoading() {
        world.beginCalls += 1;
        return world.lease;
      },
      endModelLoading(_lease: unknown) {},
    };
  },
};

async function ejectModel(): Promise<boolean> {
__EJECT_BODY__
}

export { ejectModel };
"""


def _eject_body() -> str:
    return slice_between(
        read(HOOK),
        "    if (!params.checkpoint) {\n      return false;\n    }",
        "  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);",
    )


def _run(script: str) -> dict:
    require_node((HOOK,))
    return run_harness(
        TEMP,
        HARNESS.replace("__EJECT_BODY__", _eject_body()),
        script,
        sources = (),
    )


def test_a_second_eject_while_confirm_is_open_does_not_unload_twice():
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { ejectModel, hangNextConfirm, releaseHungConfirm, world } from "./harness.ts";
            hangNextConfirm();
            const first = ejectModel();
            await new Promise((resolve) => setTimeout(resolve, 20));
            const second = await ejectModel();
            releaseHungConfirm();
            const firstResult = await first;
            console.log(JSON.stringify({
              second,
              firstResult,
              unloads: world.unloads,
              toasts: world.toasts.map((t) => t.title),
              beginCalls: world.beginCalls,
            }));
            """
        )
    )
    assert out["second"] is False
    assert out["firstResult"] is True
    assert out["unloads"] == 1
    assert out["beginCalls"] == 1
    assert out["toasts"] == ["Wait for the model to finish unloading."]


def test_a_null_lifecycle_lease_toasts_instead_of_returning_silently():
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { ejectModel, world } from "./harness.ts";
            world.lease = null;
            const result = await ejectModel();
            console.log(JSON.stringify({
              result,
              unloads: world.unloads,
              toasts: world.toasts.map((t) => t.title),
            }));
            """
        )
    )
    assert out["result"] is False
    assert out["unloads"] == 0
    assert out["toasts"] == ["Wait for the current model to finish loading."]
