// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ProfilePersonalizationPanel } from "@/features/profile";
import { useI18n } from "@/features/i18n";

export function ProfileTab() {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.profile.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.profile.subtitle")}
        </p>
      </header>

      <ProfilePersonalizationPanel />
    </div>
  );
}
