// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const MODEL = "unsloth/Qwen3-8B";

function statusFor(requested: number | null, effective = requested) {
  return {
    active_model: MODEL,
    model_identifier: MODEL,
    is_gguf: false,
    requested_parallel_slots: requested,
    parallel_slots: effective,
  } as never;
}

test("a non-GGUF status carries the width it reports into the rollback baseline", () => {
  useChatRuntimeStore.setState({
    modelLoading: false,
    loadedNParallel: null,
    nParallel: null,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });

  applyActiveModelStatusToStore(statusFor(8, 1), { previousCheckpoint: MODEL });

  assert.equal(
    useChatRuntimeStore.getState().loadedNParallel,
    8,
    "the width the load was invoked with is what a rollback must re-send",
  );
});

test("a status reporting no width clears the baseline it would otherwise re-send", () => {
  useChatRuntimeStore.setState({
    modelLoading: false,
    loadedNParallel: 8,
    nParallel: null,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });

  applyActiveModelStatusToStore(statusFor(null, null), { previousCheckpoint: MODEL });

  assert.equal(useChatRuntimeStore.getState().loadedNParallel, null);
});

test("a remembered non-GGUF width is adopted into the control, not just the baseline", async () => {
  // The baseline above only feeds a rollback. Without this the control stays blank while
  // the model runs on the remembered width, and the next Apply would save the blank over it.
  const { setResidentInitialConfig } = await import(
    "./helpers/store-stubs/model-picker.ts"
  );
  setResidentInitialConfig({ remembered: true, config: { nParallel: 2 } });
  useChatRuntimeStore.setState({
    modelLoading: false,
    loadedNParallel: null,
    nParallel: null,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: "unsloth/Other" },
  });

  applyActiveModelStatusToStore(statusFor(2), { previousCheckpoint: "unsloth/Other" });
  setResidentInitialConfig(null);

  assert.equal(useChatRuntimeStore.getState().nParallel, 2);
});

test("the baseline follows a width the server changed underneath this tab", () => {
  // An MLX width changes without a reload, so the running count can move while this tab
  // holds the same model. A baseline left at the first count seen would show no pending
  // difference and roll back to a width the server stopped running.
  useChatRuntimeStore.setState({
    modelLoading: false,
    loadedNParallel: 2,
    nParallel: 2,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });

  applyActiveModelStatusToStore(statusFor(4), { previousCheckpoint: MODEL });

  assert.equal(useChatRuntimeStore.getState().loadedNParallel, 4);
});
