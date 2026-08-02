// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatProvidersSettings } from "@/features/chat/chat-providers-dialog";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";

export function ConnectionsTab() {
  const providers = useExternalProvidersStore((s) => s.providers);
  const setProviders = useExternalProvidersStore((s) => s.setProviders);

  return (
    <ChatProvidersSettings
      providers={providers}
      onProvidersChange={setProviders}
    />
  );
}
