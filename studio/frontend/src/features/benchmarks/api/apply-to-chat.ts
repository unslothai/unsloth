// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hands a winning row back to chat: its Run Settings take the row's values and the
// model reloads with them, so the next message runs at the measured speed.

import {
  applyActiveModelStatusToStore,
  getInferenceStatus,
  loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { type Variant, variantLoad } from "../lib/bench-math";
import { chatBaseLoad } from "./chat-base";

export async function applyVariantToChat(
  variant: Variant,
  modelPath?: string | null,
  ggufVariant?: string | null,
): Promise<void> {
  const status = await getInferenceStatus();
  const target = modelPath ?? status.active_model;
  if (!target) throw new Error("Load a model in chat first.");
  const same =
    target === status.active_model &&
    (ggufVariant === undefined || ggufVariant === status.gguf_variant);
  const base = same
    ? chatBaseLoad(status)
    : {
        ...chatBaseLoad({ ...status, active_model: target }),
        gguf_variant: ggufVariant ?? null,
      };
  const payload = variantLoad(base, variant);
  // The sheet's own fields move first, so a poll landing mid-load cannot revert them.
  useChatRuntimeStore.setState({
    speculativeType: payload.speculative_type ?? null,
    specDraftNMax: payload.spec_draft_n_max ?? null,
    loadedLlamaExtraArgs: payload.llama_extra_args ?? [],
  });
  await loadModel(payload, { runtime: "chat" });
  applyActiveModelStatusToStore(await getInferenceStatus());
}
