// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ExternalProviderConfig,
  parseExternalModelId,
  providerLocalToolsEnabled,
  supportsProviderLocalTools,
} from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";

export interface ExternalLocalToolsState {
  /** the external provider connection, or null. */
  provider: ExternalProviderConfig | null;
  /** whether the external provider connection supports running local tools. */
  supported: boolean;
  /** whether the user enabled local tools in settings. */
  enabled: boolean;
}

const NO_LOCAL_TOOLS: ExternalLocalToolsState = {
  provider: null,
  supported: false,
  enabled: false,
};

/**
 * check whether the selected external connection can run unsloth's local tools.
 * when `supported && !enabled`, search/code pills show an opt-in notice.
 */
export function useExternalLocalTools(): ExternalLocalToolsState {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const providers = useExternalProvidersStore((s) => s.providers);

  const selection = parseExternalModelId(checkpoint);
  if (!selection || !connectionsEnabled) return NO_LOCAL_TOOLS;
  const provider =
    providers.find((entry) => entry.id === selection.providerId) ?? null;
  if (!provider || !supportsProviderLocalTools(provider.providerType)) {
    return NO_LOCAL_TOOLS;
  }
  return {
    provider,
    supported: true,
    enabled: providerLocalToolsEnabled(provider),
  };
}
