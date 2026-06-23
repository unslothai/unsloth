// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeSpeculativeType,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import type { PerModelConfig } from "./per-model-config";

function cleanTemplate(value: string | null | undefined): string | null {
  return value?.trim() ? value : null;
}

export function applyPerModelConfigToRuntime(config: PerModelConfig): void {
  useChatRuntimeStore.setState({
    customContextLength: config.customContextLength ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(config.speculativeType),
    specDraftNMax: config.specDraftNMax ?? null,
    tensorParallel: config.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
  });
}
