// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stubs the chat, images and video barrels and the embedder; /unload 409s like the backend while a chat
// is generating unless forced.

type UnloadPayload = { model_path: string; force_cancel_active?: boolean };

export const state = {
  active: null as string | null,
  loaded: [] as string[],
  servingCheckpoints: [] as string[],
  embeddingLoaded: false,
  generating: false,
  confirmAnswer: { proceed: true, forceCancelActive: false },
  confirmCalls: [] as { action: string; effect: string; model?: string }[],
  unloads: [] as UnloadPayload[],
  queueStops: [] as string[][],
};

export function resetStub({
  active = null as string | null,
  loaded = [] as string[],
  servingCheckpoints = [] as string[],
  embeddingLoaded = false,
  generating = false,
} = {}): void {
  state.active = active;
  state.loaded = [...loaded];
  state.servingCheckpoints = [...servingCheckpoints];
  state.embeddingLoaded = embeddingLoaded;
  state.generating = generating;
  state.confirmAnswer = { proceed: true, forceCancelActive: false };
  state.confirmCalls = [];
  state.unloads = [];
  state.queueStops = [];
}

export async function getInferenceStatus() {
  return {
    active_model: state.active,
    loaded: [...state.loaded],
    serving_checkpoints: [...state.servingCheckpoints],
  };
}

export function resolveInferenceCheckpointId(status: {
  active_model: string | null;
}): string | null {
  return status.active_model ?? null;
}

export function isExternalModelId() {
  return false;
}

export async function unloadModel(payload: UnloadPayload): Promise<void> {
  state.unloads.push(payload);
  if (state.generating && !payload.force_cancel_active) {
    throw new Error(
      "Unloading this model would stop 1 chat that is still generating. Stop them first, or retry with force_cancel_active.",
    );
  }
  state.generating = false;
  state.loaded = state.loaded.filter((name) => name !== payload.model_path);
  state.servingCheckpoints = state.servingCheckpoints.filter(
    (checkpoint) => checkpoint !== payload.model_path,
  );
  if (state.active === payload.model_path) state.active = null;
}

export async function confirmStopRunningChatsIfNeeded(
  action: string,
  effect: string,
  model?: string,
) {
  state.confirmCalls.push({ action, effect, model });
  const answer = state.generating
    ? state.confirmAnswer
    : { proceed: true, forceCancelActive: false };
  const threads = state.generating ? ["running-thread"] : [];
  return { ...answer, promptQueueThreadIds: threads, preStreamRunTokens: [] };
}

export function requestScopedLocalPromptQueueStop(threadIds: string[]) {
  state.queueStops.push(threadIds);
}

export const useChatRuntimeStore = {
  getState: () => ({ params: { checkpoint: null }, clearCheckpoint() {} }),
};

export async function getDiffusionStatus() {
  return { loaded: false };
}
export async function unloadDiffusionModel() {
  return { loaded: false };
}
export async function getVideoStatus() {
  return { loaded: false };
}
export async function unloadVideoModel() {
  return { loaded: false };
}

export async function unloadEmbeddingModel() {
  state.embeddingLoaded = false;
  return { backendLoaded: false };
}
export const useEmbeddingModelStore = {
  getState: () => ({
    applyResidency: async (request: () => Promise<unknown>) => {
      await request();
    },
  }),
};
