// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  parseExternalModelId,
  providerModelSupportsStudioTools,
} from "../external-providers";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Pre-select gate for the RAG toggle, mirroring Web search / Code / MCP: armable with no model,
// disabled only when a loaded model cannot run search_knowledge_base. The send path checks
// supportsTools independently.
export function useRagToolDisabled(): boolean {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const externalSelection = parseExternalModelId(checkpoint);
  const providers = useExternalProvidersStore((s) => s.providers);
  const externalProvider = externalSelection
    ? providers.find((provider) => provider.id === externalSelection.providerId)
    : undefined;
  const externalWithoutStudioTools =
    externalSelection !== null &&
    providerModelSupportsStudioTools(
      externalProvider?.providerType,
      externalSelection.modelId,
    ) !== true;
  return modelLoaded && (externalWithoutStudioTools || !supportsTools);
}
