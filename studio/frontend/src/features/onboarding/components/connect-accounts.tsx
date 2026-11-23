// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ApiProviderLogo } from "@/features/chat/api-provider-logo";
import {
  type ExternalProviderConfig,
  newConnectionRequest,
} from "@/features/chat/external-providers";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { type TranslationKey, useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { Group, Row } from "./onboarding-ui";

const ACCOUNTS: {
  type: string;
  label: TranslationKey;
  body: TranslationKey;
}[] = [
  {
    type: "openai_codex",
    label: "onboarding.connect.chatgpt",
    body: "onboarding.connect.chatgptBody",
  },
  {
    type: "anthropic",
    label: "onboarding.connect.claude",
    body: "onboarding.connect.claudeBody",
  },
];

function isConnected(provider: ExternalProviderConfig | undefined): boolean {
  if (!provider) return false;
  return provider.authKind === "chatgpt_oauth"
    ? provider.authStatus === "connected"
    : provider.hasApiKey === true;
}

export function ConnectAccounts() {
  const t = useT();
  const providers = useExternalProvidersStore((s) => s.providers);
  return (
    <div className="mt-10">
      <div className="mb-6">
        <div className="text-sm font-semibold">
          {t("onboarding.connect.title")}
        </div>
        <div className="text-muted-foreground mt-1 text-xs leading-relaxed">
          {t("onboarding.connect.body")}
        </div>
      </div>
      <Group>
        {ACCOUNTS.map((account) => {
          const provider = providers.find(
            (p) => (p.backendProviderType ?? p.providerType) === account.type,
          );
          const connected = isConnected(provider);
          return (
            <Row
              key={account.type}
              lead={
                <span className="bg-muted flex size-8 shrink-0 items-center justify-center rounded-full">
                  <ApiProviderLogo
                    providerType={account.type}
                    className="size-4"
                  />
                </span>
              }
              label={t(account.label)}
              description={t(account.body)}
              trailing={
                connected ? (
                  <span className="text-muted-foreground flex items-center gap-1.5 px-2 text-xs font-medium">
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      className="size-3.5"
                      strokeWidth={2.5}
                    />
                    {t("onboarding.connect.connected")}
                  </span>
                ) : (
                  <Button
                    variant="muted"
                    size="sm"
                    className="h-7 rounded-full px-3.5 text-xs font-medium"
                    onClick={() =>
                      useSettingsDialogStore
                        .getState()
                        .openConnectionSettings(
                          provider?.id ?? newConnectionRequest(account.type),
                        )
                    }
                  >
                    {t("onboarding.connect.connect")}
                  </Button>
                )
              }
            />
          );
        })}
      </Group>
    </div>
  );
}
