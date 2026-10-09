// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { translate, useT } from "@/i18n";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import {
  type McpAccessSettings,
  loadMcpAccess,
  updateMcpAccess,
} from "../api/mcp-access";
import { SettingsRow } from "./settings-row";

const ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP";

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

export function McpAccessSection(): ReactElement | null {
  const t = useT();
  const [settings, setSettings] = useState<McpAccessSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadMcpAccess().then(
      (next) => live && setSettings(next),
      (err) =>
        live &&
        setError(
          errorMessage(err) ?? translate("settings.apiKeys.mcp.loadError"),
        ),
    );
    return () => {
      live = false;
    };
  }, []);

  const apply = async (enabled: boolean) => {
    setBusy(true);
    setError(null);
    try {
      setSettings(await updateMcpAccess(enabled));
    } catch (err) {
      // The server may hold a different value than the switch now shows.
      try {
        setSettings(await loadMcpAccess());
      } catch (refreshError) {
        console.warn(
          "Couldn't refresh agent access settings after a rejected change.",
          refreshError,
        );
      }
      setError(errorMessage(err) ?? t("settings.apiKeys.mcp.saveError"));
    } finally {
      setBusy(false);
    }
  };

  const header = (
    <>
      <div className="flex items-start gap-3 bg-muted/30 p-4">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
          <HugeiconsIcon
            icon={McpServerIcon}
            className="size-4 text-foreground"
          />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <h2 className="settings-heading text-base font-semibold font-heading">
            {t("settings.apiKeys.mcp.title")}
          </h2>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.apiKeys.mcp.description")}
          </p>
        </div>
      </div>

      {error ? (
        <p className="border-t border-border/60 px-4 py-2.5 text-xs leading-snug text-destructive">
          {error}
        </p>
      ) : null}
    </>
  );

  if (!settings) {
    return error ? (
      <section
        data-settings-label={t("settings.apiKeys.mcp.title")}
        className="overflow-hidden rounded-lg border border-border/70"
      >
        {header}
      </section>
    ) : null;
  }

  const { enabled, forcedByEnv } = settings;

  return (
    <section
      data-settings-label={t("settings.apiKeys.mcp.title")}
      className="overflow-hidden rounded-lg border border-border/70"
    >
      {header}

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label={t("settings.apiKeys.mcp.enable")}
          description={
            forcedByEnv
              ? t("settings.apiKeys.mcp.lockedByEnv", { name: ENV_FORCE })
              : t("settings.apiKeys.mcp.enableDescription")
          }
        >
          <Switch
            checked={enabled}
            disabled={busy || forcedByEnv}
            onCheckedChange={(on) => void apply(on)}
            aria-label={t("settings.apiKeys.mcp.enable")}
          />
        </SettingsRow>
      </div>
    </section>
  );
}
