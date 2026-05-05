// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { ThemeSegmented } from "../components/theme-segmented";

export function AppearanceTab() {
  const { pinned, setPinned } = useSidebarPin();
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">Appearance</h1>
        <p className="text-xs text-muted-foreground">
          How Unsloth Studio looks on this device.
        </p>
      </header>

      <SettingsSection title="Theme">
        <SettingsRow
          label="Color scheme"
          description="Choose light, dark, or follow your system."
        >
          <ThemeSegmented />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Layout">
        <SettingsRow
          label="Pin sidebar by default"
          description="Keep the sidebar expanded instead of collapsing to icons."
        >
          <Switch checked={pinned} onCheckedChange={setPinned} />
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
