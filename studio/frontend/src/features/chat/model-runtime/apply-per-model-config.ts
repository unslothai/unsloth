// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "../model-config/per-model-config";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { normalizeRuntimePerModelLoadConfig } from "./per-model-load-config";

export function applyPerModelConfigToRuntime(config: PerModelConfig): void {
  const normalized = normalizeRuntimePerModelLoadConfig(config);
  const state = useChatRuntimeStore.getState();
  useChatRuntimeStore.setState({
    kvCacheDtype: normalized.kvCacheDtype,
    speculativeType: normalized.speculativeType,
    specDraftNMax: normalized.specDraftNMax,
    customContextLength: normalized.customContextLength,
    chatTemplateOverride: normalized.chatTemplateOverride,
  });
  state.setParams({
    ...state.params,
    trustRemoteCode: normalized.trustRemoteCode,
  });
}
