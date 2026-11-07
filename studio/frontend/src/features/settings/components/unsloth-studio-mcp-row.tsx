// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { loadMcpAccess } from "../api/mcp-access";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";

/** Agent access status with a link to its settings at the end of Settings > Agents. Owner-only callers. */
export function UnslothStudioMcpRow(): ReactElement {
  const t = useT();
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    let live = true;
    loadMcpAccess().then(
      (next) => live && setEnabled(next.enabled),
      // The status stays blank; the link still works.
      (err) => console.warn("Couldn't load the Unsloth MCP status.", err),
    );
    return () => {
      live = false;
    };
  }, []);

  const status =
    enabled === null
      ? " "
      : enabled
        ? t("settings.mcp.status.on")
        : t("settings.mcp.status.off");

  // Same shape as the Keyless API access link beside it.
  return (
    <SettingsRow
      icon={
        <HugeiconsIcon
          icon={McpServerIcon}
          strokeWidth={1.75}
          className="size-[calc(18px*var(--ui-space-scale,1))]"
        />
      }
      label={t("settings.apiKeys.mcp.title")}
      description={status}
    >
      <button
        type="button"
        onClick={() =>
          useSettingsDialogStore
            .getState()
            .openDialog("agents", { scrollTarget: "agents-mcp" })
        }
        className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        {t("settings.mcp.apiKeysRow.open")}
        <HugeiconsIcon icon={ChevronRightStandardIcon} className="size-3" />
      </button>
    </SettingsRow>
  );
}
