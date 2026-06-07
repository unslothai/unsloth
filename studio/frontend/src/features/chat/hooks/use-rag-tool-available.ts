// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Single source of truth for the RAG pill's disabled gate and the Add Files bar's
// visibility so the bar never shows while the pill is inert.
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
