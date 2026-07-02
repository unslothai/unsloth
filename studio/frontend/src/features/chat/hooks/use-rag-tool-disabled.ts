// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Pre-select gate for the RAG toggle, mirroring Web search/Code/MCP: armable
// with no model; disabled only when a loaded model can't run
// search_knowledge_base. The send path checks supportsTools independently.
export function useRagToolDisabled(): boolean {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  return (
    modelLoaded && (parseExternalModelId(checkpoint) !== null || !supportsTools)
  );
}
