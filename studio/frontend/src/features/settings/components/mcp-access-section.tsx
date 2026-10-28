// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { McpServerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { usePlatformStore } from "@/config/env";
import { translate, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import {
  type ReactElement,
  useEffect,
  useId,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  type McpAccessSettings,
  loadMcpAccess,
  updateMcpAccess,
  updateMcpTools,
} from "../api/mcp-access";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
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
  hasMcpSnippet,
} from "./mcp-agent-snippet";
import { McpToolsDialog } from "./mcp-tools-dialog";
import { SettingsRow } from "./settings-row";
import { readUseTunnelPref, subscribeUseTunnelPref } from "./tunnel-preference";

const ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP";
const MCP_AGENTS = SUPPORTED_AGENTS.filter((agent) => hasMcpSnippet(agent.id));
const OS_OPTIONS = [
  { os: "unix", labelKey: "settings.apiKeys.osUnix" },
  { os: "windows", labelKey: "settings.apiKeys.osWindows" },
] as const;

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

export function McpAccessSection({
  apiKey = null,
}: {
  /** newly created key inserted into snippets instead of the environment variable */
  apiKey?: string | null;
}): ReactElement | null {
  const t = useT();
  const pillId = useId();
  const [settings, setSettings] = useState<McpAccessSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolsSaving, setToolsSaving] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const sectionRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore(
    (s) => s.consumeScrollTarget,
  );
  const loaded = settings !== null || error !== null;
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const storedAgent = useSettingsPanelPrefsStore((s) => s.mcpAgent);
  const storedOs = useSettingsPanelPrefsStore((s) => s.mcpOs);
  const setStoredAgent = useSettingsPanelPrefsStore((s) => s.setMcpAgent);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setMcpOs);

  const agent =
    storedAgent && MCP_AGENTS.some((a) => a.id === storedAgent)
      ? storedAgent
      : MCP_AGENTS[0].id;
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
      // resync because a rejected update may leave the server value unknown.
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

  // Settings > API links here; wait for the first load, since the card renders nothing before it.
  useEffect(() => {
    if (scrollTarget !== "agents-mcp" || !loaded) return;
    const frame = window.requestAnimationFrame(() => {
      sectionRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      consumeScrollTarget("agents-mcp");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget, loaded]);

  const saveTools = async (states: Record<string, boolean>) => {
    setToolsSaving(true);
    setToolsError(null);
    try {
      setSettings(await updateMcpTools(states));
    } catch (err) {
      // resync because a rejected update may leave the server value unknown.
      try {
        setSettings(await loadMcpAccess());
      } catch (refreshError) {
        console.warn(
          "Couldn't refresh the MCP tools after a rejected change.",
          refreshError,
        );
      }
      setToolsError(errorMessage(err) ?? t("settings.mcp.tools.saveError"));
    } finally {
      setToolsSaving(false);
    }
  };

  // subscribe to same-tab tunnel changes from the API examples.
  const useTunnel = useSyncExternalStore(
    subscribeUseTunnelPref,
    readUseTunnelPref,
    () => true,
  );

  if (!(settings || error)) {
    return null;
  }

  // match the server address shown by the API examples.
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl
      ? cloudflareUrl
      : isTauri
        ? (serverUrl ?? origin)
        : origin;
  const snippet = settings?.enabled
    ? (buildMcpSnippet(agent, base, os, apiKey) ??
      buildMcpSnippet(agent, settings.url, os, apiKey))
    : null;
  const agentDetails = detailsFor(agent);

  return (
    <div
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.mcp.title")}
      className="flex min-w-0 flex-col"
    >
      {error ? (
        <p className="pt-2 text-xs leading-snug text-destructive">{error}</p>
      ) : null}

      {settings ? (
        <>
          <SettingsRow
            icon={
              <HugeiconsIcon
                icon={McpServerIcon}
                strokeWidth={1.75}
                className="size-[calc(18px*var(--ui-space-scale,1))]"
              />
            }
            label={t("settings.apiKeys.mcp.title")}
            hint={t("settings.apiKeys.mcp.enableDescription")}
            description={
              settings.forcedByEnv
                ? t("settings.apiKeys.mcp.lockedByEnv", { name: ENV_FORCE })
                : t("settings.apiKeys.mcp.description")
            }
          >
            <Switch
              checked={settings.enabled}
              disabled={busy || settings.forcedByEnv}
              onCheckedChange={(on) => void apply(on)}
              aria-label={t("settings.apiKeys.mcp.enable")}
            />
          </SettingsRow>

          {/* null on an older backend that sends no tool list. */}
          {settings.tools ? (
            <SettingsRow
              label={t("settings.mcp.tools.title")}
              description={t("settings.mcp.tools.count", {
                on: settings.tools.filter((tool) => tool.enabled).length,
                total: settings.tools.length,
              })}
            >
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setToolsError(null);
                  setToolsOpen(true);
                }}
              >
                {t("settings.mcp.tools.choose")}
              </Button>
            </SettingsRow>
          ) : null}

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
                        <AgentIcon {...agentDetails} />
                        <span className="truncate">{agentDetails.name}</span>
                      </span>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent align="end">
                    {MCP_AGENTS.map((option) => (
                      <SelectItem key={option.id} value={option.id}>
                        <span className="flex min-w-0 items-center gap-2">
                          <AgentIcon {...option} />
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
                            "relative inline-flex h-8 items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                            os === option.os
                              ? "text-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {os === option.os && (
                            <SegmentedPill group={`${pillId}-shell`} />
                          )}
                          {t(option.labelKey)}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                ) : null}

                {/* exclude commands containing a newly created key from reload snapshots. */}
                <div
                  className="min-w-0"
                  data-reload-snapshot-sensitive={apiKey ? "" : undefined}
                >
                  <CommandBlock command={snippet.text} />
                </div>

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
        </>
      ) : null}

      {settings?.tools ? (
        <McpToolsDialog
          open={toolsOpen}
          onOpenChange={setToolsOpen}
          tools={settings.tools}
          saving={toolsSaving}
          error={toolsError}
          onSave={(states) => void saveTools(states)}
        />
      ) : null}
    </div>
  );
}
