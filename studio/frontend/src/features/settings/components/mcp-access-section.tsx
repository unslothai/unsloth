// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import { translate, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import {
  type McpAccessSettings,
  loadMcpAccess,
  updateMcpAccess,
} from "../api/mcp-access";
import {
  type ExampleOs,
  useSettingsPanelPrefsStore,
} from "../stores/settings-panel-prefs-store";
import { AgentIcon, CommandBlock } from "./agent-command-block";
import { SUPPORTED_AGENTS, detailsFor } from "./coding-agent-list";
import {
  MCP_API_KEY_ENV,
  MCP_SHELL_AGENT_IDS,
  buildMcpSnippet,
} from "./mcp-agent-snippet";
import { SettingsRow } from "./settings-row";
import { readUseTunnelPref } from "./usage-examples";

const ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP";
const OS_OPTIONS = [
  { os: "unix", labelKey: "settings.apiKeys.osUnix" },
  { os: "windows", labelKey: "settings.apiKeys.osWindows" },
] as const;

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

export function McpAccessSection(): ReactElement | null {
  const t = useT();
  const [settings, setSettings] = useState<McpAccessSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const storedAgent = useSettingsPanelPrefsStore((s) => s.mcpAgent);
  const storedOs = useSettingsPanelPrefsStore((s) => s.mcpOs);
  const setStoredAgent = useSettingsPanelPrefsStore((s) => s.setMcpAgent);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setMcpOs);

  const agent =
    storedAgent && SUPPORTED_AGENTS.some((a) => a.id === storedAgent)
      ? storedAgent
      : SUPPORTED_AGENTS[0].id;
  // an explicit pick wins, since the agent may run on another machine.
  const os: ExampleOs =
    storedOs ?? (deviceType === "windows" ? "windows" : "unix");

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
  // The same address the API usage examples show, so both point at one server.
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    readUseTunnelPref() && cloudflareUrl
      ? cloudflareUrl
      : (serverUrl ?? origin);
  const snippet = enabled
    ? (buildMcpSnippet(agent, base, os) ??
      buildMcpSnippet(agent, settings.url, os))
    : null;
  const agentDetails = detailsFor(agent);

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

        {snippet ? (
          <>
            <SettingsRow label={t("settings.apiKeys.mcp.agent")}>
              <Select value={agent} onValueChange={setStoredAgent}>
                <SelectTrigger
                  className="w-48"
                  aria-label={t("settings.apiKeys.mcp.agent")}
                >
                  <SelectValue>
                    <span className="flex min-w-0 items-center gap-2">
                      <AgentIcon
                        logo={agentDetails.logo}
                        icon={agentDetails.icon}
                        darkIcon={agentDetails.darkIcon}
                        color={agentDetails.color}
                        mark={agentDetails.mark}
                      />
                      <span className="truncate">{agentDetails.name}</span>
                    </span>
                  </SelectValue>
                </SelectTrigger>
                <SelectContent align="end">
                  {SUPPORTED_AGENTS.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      <span className="flex min-w-0 items-center gap-2">
                        <AgentIcon
                          logo={option.logo}
                          icon={option.icon}
                          darkIcon={option.darkIcon}
                          color={option.color}
                          mark={option.mark}
                        />
                        <span className="truncate">{option.name}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </SettingsRow>

            <div className="flex min-w-0 flex-col gap-2 pb-3">
              {MCP_SHELL_AGENT_IDS.has(agent) ? (
                <fieldset className="flex min-w-0">
                  <legend className="sr-only">
                    {t("settings.agents.commandShell")}
                  </legend>
                  <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
                    {OS_OPTIONS.map((option) => (
                      <button
                        key={option.os}
                        type="button"
                        onClick={() => setStoredOs(option.os)}
                        aria-pressed={os === option.os}
                        className={cn(
                          "inline-flex h-8 items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                          os === option.os
                            ? "hub-tab-toggle-pill text-foreground"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {t(option.labelKey)}
                      </button>
                    ))}
                  </div>
                </fieldset>
              ) : null}

              <CommandBlock command={snippet.text} />

              {snippet.configPath ? (
                <p className="text-xs leading-snug text-muted-foreground">
                  {t("settings.apiKeys.mcp.configFileHint", {
                    path: snippet.configPath,
                  })}
                </p>
              ) : null}
              {snippet.readsKeyEnv ? (
                <p className="text-xs leading-snug text-muted-foreground">
                  {t("settings.apiKeys.mcp.exportKeyHint", {
                    name: MCP_API_KEY_ENV,
                  })}
                </p>
              ) : null}
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}
