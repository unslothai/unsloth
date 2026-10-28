// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { loadMcpAccess } from "../api/mcp-access";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/** Agent access status with a link to its settings in Settings > Agents. Owner-only callers. */
export function UnslothStudioMcpRow(): ReactElement {
  const t = useT();
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    let live = true;
    loadMcpAccess().then(
      (next) => live && setEnabled(next.enabled),
      (err) =>
        console.warn("Couldn't load the Unsloth MCP status.", err),
    );
    return () => {
      live = false;
    };
  }, []);

  const status =
    enabled === null
      ? "\u00a0"
      : enabled
        ? t("settings.mcp.status.on")
        : t("settings.mcp.status.off");

  return (
    <SettingsSection title={t("settings.apiKeys.mcp.title")}>
      <SettingsRow
        label={t("settings.mcp.apiKeysRow.label")}
        description={status}
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            useSettingsDialogStore
              .getState()
              .openDialog("agents", { scrollTarget: "agents-mcp" })
          }
        >
          {t("settings.mcp.apiKeysRow.open")}
          <HugeiconsIcon
            icon={ChevronRightStandardIcon}
            className="ml-1 size-3.5"
          />
        </Button>
      </SettingsRow>
    </SettingsSection>
  );
}
