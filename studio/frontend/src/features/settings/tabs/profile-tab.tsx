// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MemorySettingsPanel } from "@/features/chat";
import { ProfilePersonalizationPanel } from "@/features/profile";
import { useT } from "@/i18n";

export function ProfileTab() {
  const t = useT();

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1
          data-settings-label={t("settings.profile.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.profile.title")}
        </h1>
        <p
          data-settings-label={t("settings.profile.description")}
          className="text-xs text-muted-foreground"
        >
          {t("settings.profile.description")}
        </p>
      </header>

      <ProfilePersonalizationPanel />

      <MemorySettingsPanel />
    </div>
  );
}
