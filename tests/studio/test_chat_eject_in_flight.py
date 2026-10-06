# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat eject toasts before the running-chats check and names a second click as an unload (#10339)."""

from __future__ import annotations

import textwrap

from _node_harness import WORKDIR, read, require_node, run_harness, slice_between, source_path

HOOK = source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")

HARNESS = """
// @ts-nocheck
export const world: any = { toasts: [], unloads: 0 };

let phase: string | null = null;
const chatModelLifecycleGate = { currentPhase: () => phase };
const params = { checkpoint: "org/model" };
function setModelsError(_message: string | null): void {}
function clearCheckpoint(): void {}
async function refresh(): Promise<void> {}
function isExternalModelId(_id: string): boolean { return false; }
function stopQueuedRuns(_decision: unknown, _scoped: boolean): void {}
function requestLocalPromptQueueStop(): void {}
async function unloadModel(_payload: unknown): Promise<void> { world.unloads += 1; }

// The running-chats check queued behind open streams: it answers only when released.
let release: () => void = () => {};
const checked = new Promise<void>((resolve) => { release = resolve; });
export function releaseCheck(): void { release(); }
async function confirmStopRunningChatsIfNeeded(): Promise<any> {
  await checked;
  return { proceed: true, forceCancelActive: false, promptQueueThreadIds: [], preStreamRunTokens: [] };
}

const toast = {
  info: (title: string) => world.toasts.push(["info", title]),
  loading: (title: string) => { world.toasts.push(["loading", title]); return "t1"; },
  dismiss: () => world.toasts.push(["dismiss", ""]),
  promise: (pending: Promise<unknown>, data: any) => world.toasts.push(["promise", data.id]),
};

const useChatRuntimeStore = {
  getState: () => ({
    get modelLoading() { return phase !== null; },
    loadingModelPick: null,
    loadedModels: [{ checkpoint: "org/model" }],
    beginModelLoading(next: string) {
      if (phase !== null) return null;
      phase = next;
      return 1;
    },
    endModelLoading() { phase = null; },
  }),
};

export async function ejectModel(confirmed?: any): Promise<boolean> {
__BODY__
}
"""


def test_eject_toasts_before_the_check_and_refuses_a_second_click_as_unloading():
    require_node((HOOK,))
    body = slice_between(
        read(HOOK),
        "    if (!params.checkpoint) {\n      return false;\n    }",
        "  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);",
    )
    out = run_harness(
        WORKDIR / "temp" / "chat_eject_in_flight",
        HARNESS.replace("__BODY__", body),
        textwrap.dedent(
            """
            // @ts-nocheck
            import { ejectModel, releaseCheck, world } from "./harness.ts";
            const first = ejectModel();
            await new Promise((resolve) => setTimeout(resolve, 20));
            const beforeCheck = [...world.toasts];
            const second = await ejectModel();
            releaseCheck();
            console.log(JSON.stringify({ beforeCheck, second, first: await first,
              toasts: world.toasts, unloads: world.unloads }));
            """
        ),
        sources = (),
    )
    assert out["beforeCheck"] == [["loading", "Unloading model"]]
    assert out["second"] is False
    assert ["info", "Wait for the model to finish unloading."] in out["toasts"]
    assert not any("A model is loading" in t for _, t in out["toasts"])
    assert out["first"] is True
    assert out["unloads"] == 1
    assert ["promise", "t1"] in out["toasts"], "the unload must reuse the click's toast"
