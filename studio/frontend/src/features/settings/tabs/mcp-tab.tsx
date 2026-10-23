// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { McpAccessSection } from "../components/mcp-access-section";
import { McpToolsSection } from "../components/mcp-tools-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function McpTab() {
  const t = useT();
  const newApiKey = useSettingsDialogStore((s) => s.newApiKey);

  return (
    <div className="settings-page">
      {/* data-settings-label lets indexed settings search scroll to these. */}
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.mcp.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.mcp.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.mcp.description")}
        </p>
      </header>

      <div className="flex flex-col gap-2">
        <McpAccessSection apiKey={newApiKey} />
        <p className="text-xs leading-snug text-muted-foreground">
          {t("settings.mcp.keyHint")}{" "}
          <button
            type="button"
            className="rounded-sm font-medium text-foreground underline underline-offset-2 transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            onClick={() =>
              useSettingsDialogStore.getState().setActiveTab("api-keys")
            }
          >
            {t("settings.mcp.createKey")}
          </button>
        </p>
      </div>

      <McpToolsSection />
    </div>
  );
}
