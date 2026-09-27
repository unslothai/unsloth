// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatProvidersSettings } from "@/features/chat/chat-providers-dialog";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function ConnectionsTab() {
  const providers = useExternalProvidersStore((s) => s.providers);
  const setProviders = useExternalProvidersStore((s) => s.setProviders);
  // Set when the picker's Connected group gear asked for one connection by name.
  const connectionRequested = useSettingsDialogStore(
    (s) => s.connectionRequested,
  );
  const consumeConnectionRequest = useSettingsDialogStore(
    (s) => s.consumeConnectionRequest,
  );

  return (
    <ChatProvidersSettings
      providers={providers}
      onProvidersChange={setProviders}
      openProviderId={connectionRequested}
      onOpenProviderConsumed={consumeConnectionRequest}
    />
  );
}
