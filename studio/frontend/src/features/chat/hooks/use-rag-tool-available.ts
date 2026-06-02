// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/**
 * search_knowledge_base is a local tool, so retrieval only works once a local,
 * tool-capable model is loaded. Single source of truth for the RAG pill's
 * disabled gate and the Add Files bar's visibility so the two never disagree
 * (the bar must not show while the pill is inert).
 */
export function useRagToolAvailable(): boolean {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  return (
    modelLoaded && parseExternalModelId(checkpoint) === null && supportsTools
  );
}
