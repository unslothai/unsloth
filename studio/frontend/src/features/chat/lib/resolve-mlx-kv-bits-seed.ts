// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// MLX KV width + verdict fields from /api/inference/status. The editable width is a
// paired control/baseline like the other load params; the verdict strings are facts
// about the running server and always follow the baseline.

export interface MlxKvBitsSeedState {
  mlxKvBits: number | null;
  loadedMlxKvBitsRequested: number | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
}

export type MlxKvBitsSeed = Partial<MlxKvBitsSeedState>;

export function resolveMlxKvBitsSeed(options: {
  isMlx: boolean | undefined;
  mlxKvBitsDefined: boolean;
  incomingRequested: number | null;
  incomingReason: string | null;
  incomingTemplateReason: string | null;
  incomingNote: string | null;
  previous: MlxKvBitsSeedState;
  hydratingExistingModel: boolean;
  seedLoadParams: boolean;
}): MlxKvBitsSeed {
  const {
    isMlx,
    mlxKvBitsDefined,
    incomingRequested,
    incomingReason,
    incomingTemplateReason,
    incomingNote,
    previous,
    hydratingExistingModel,
    seedLoadParams,
  } = options;
  if (!seedLoadParams || !mlxKvBitsDefined) {
    return {};
  }
  if (isMlx !== true) {
    if (!hydratingExistingModel) {
      return {};
    }
    return {
      loadedMlxKvBitsRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    };
  }
  const verdict = {
    mlxKvQuantReason: incomingReason,
    chatTemplateOverrideReason: incomingTemplateReason,
    mlxKvQuantNote: incomingNote,
  };
  const unseeded =
    previous.loadedMlxKvBitsRequested === null &&
    previous.mlxKvBits === null &&
    previous.mlxKvQuantReason === null &&
    previous.chatTemplateOverrideReason === null;
  if (hydratingExistingModel || unseeded) {
    return {
      mlxKvBits: incomingRequested,
      loadedMlxKvBitsRequested: incomingRequested,
      ...verdict,
    };
  }
  if (incomingRequested === previous.loadedMlxKvBitsRequested) {
    // Width is steady, but verdict strings can move when another client reloads
    // with the same mlx_kv_bits_requested and a different chat-template override.
    return verdict;
  }
  const controlIsDirty =
    previous.mlxKvBits !== previous.loadedMlxKvBitsRequested;
  return {
    loadedMlxKvBitsRequested: incomingRequested,
    ...verdict,
    ...(controlIsDirty ? {} : { mlxKvBits: incomingRequested }),
  };
}
