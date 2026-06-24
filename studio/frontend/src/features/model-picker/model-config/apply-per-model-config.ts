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

export function currentRuntimePerModelConfig(): PerModelConfig {
  const s = useChatRuntimeStore.getState();
  return {
    customContextLength: s.customContextLength ?? null,
    kvCacheDtype: s.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(s.speculativeType),
    specDraftNMax: s.specDraftNMax ?? null,
    tensorParallel: s.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
  };
}

export function perModelConfigsEqual(
  a: PerModelConfig,
  b: PerModelConfig,
): boolean {
  return (
    (a.customContextLength ?? null) === (b.customContextLength ?? null) &&
    (a.kvCacheDtype ?? null) === (b.kvCacheDtype ?? null) &&
    normalizeSpeculativeType(a.speculativeType) ===
      normalizeSpeculativeType(b.speculativeType) &&
    (a.specDraftNMax ?? null) === (b.specDraftNMax ?? null) &&
    Boolean(a.tensorParallel) === Boolean(b.tensorParallel) &&
    cleanTemplate(a.chatTemplateOverride) ===
      cleanTemplate(b.chatTemplateOverride)
  );
}
