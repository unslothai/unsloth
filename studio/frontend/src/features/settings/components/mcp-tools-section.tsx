// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { translate, useT } from "@/i18n";
import { AiSpeechIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import {
  AiBrain01Icon,
  Album02Icon,
  Cancel01Icon,
  ChartLineData01Icon,
  CheckmarkSquare02Icon,
  ChefHatIcon,
  DashboardSpeed01Icon,
  Database01Icon,
  Download01Icon,
  FlimSlateIcon,
  HeadphonesIcon,
  Image03Icon,
  LayersIcon,
  McpServerIcon,
  Message01Icon,
  Mic01Icon,
  PackageIcon,
  PlayCircleIcon,
  SlidersHorizontalIcon,
  SpeechToTextIcon,
  StopIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import {
  type McpAccessSettings,
  type McpToolSetting,
  loadMcpAccess,
  updateMcpTool,
} from "../api/mcp-access";
import { mcpToolLabelKey } from "./mcp-tool-labels";
import { SettingsSection } from "./settings-section";

const TOOL_ICONS: Readonly<Record<string, IconSvgElement>> = {
  studio_status: DashboardSpeed01Icon,
  list_models: AiBrain01Icon,
  load_model: PlayCircleIcon,
  unload_model: StopIcon,
  plan_load: SlidersHorizontalIcon,
  download_model: Download01Icon,
  chat: Message01Icon,
  embed: LayersIcon,
  system_one: CheckmarkSquare02Icon,
  generate_image: Image03Icon,
  generate_audio: AiSpeechIcon,
  transcribe: SpeechToTextIcon,
  analyze_audio: HeadphonesIcon,
  generate_video: FlimSlateIcon,
  run_recipe: ChefHatIcon,
  get_job: TaskDone01Icon,
  datasets: Database01Icon,
  start_training: TestTubeOutlineIcon,
  list_training_runs: ChartLineData01Icon,
  export_model: PackageIcon,
  list_media: Album02Icon,
  voices: Mic01Icon,
  cancel: Cancel01Icon,
};

function toolIcon(name: string): IconSvgElement {
  return Object.hasOwn(TOOL_ICONS, name) ? TOOL_ICONS[name] : McpServerIcon;
}

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

function ToolRow({
  tool,
  disabled,
  onToggle,
}: {
  tool: McpToolSetting;
  disabled: boolean;
  onToggle: (name: string, on: boolean) => void;
}): ReactElement {
  const t = useT();
  const labelKey = mcpToolLabelKey(tool.name);
  const label = labelKey ? t(labelKey) : tool.name;
  return (
    <div
      data-mcp-tool={tool.name}
      className="flex min-h-9 items-center gap-2.5 rounded-lg px-2 py-1"
    >
      <span className="flex size-4 shrink-0 items-center justify-center text-foreground/80 [&_svg]:size-4">
        <HugeiconsIcon icon={toolIcon(tool.name)} strokeWidth={1.75} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="text-ui-13 text-foreground break-words">{label}</span>
        {labelKey ? (
          <span className="text-ui-11 text-muted-foreground font-mono break-all">
            {tool.name}
          </span>
        ) : null}
      </span>
      <Switch
        className="ml-auto"
        aria-label={label}
        checked={tool.enabled}
        disabled={disabled}
        onCheckedChange={(on) => onToggle(tool.name, on)}
      />
    </div>
  );
}

/** Per-tool switches for /mcp/. Hidden on a backend that sends no tool list. */
export function McpToolsSection(): ReactElement | null {
  const t = useT();
  const [settings, setSettings] = useState<McpAccessSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadMcpAccess().then(
      (next) => live && setSettings(next),
      (err) =>
        live &&
        setError(
          errorMessage(err) ?? translate("settings.mcp.tools.loadError"),
        ),
    );
    return () => {
      live = false;
    };
  }, []);

  const toggle = async (name: string, on: boolean) => {
    if (!settings?.tools) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      setSettings(await updateMcpTool(name, on));
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
      setError(errorMessage(err) ?? t("settings.mcp.tools.saveError"));
    } finally {
      setSaving(false);
    }
  };

  if (!(settings || error)) {
    return null;
  }
  if (settings && settings.tools === null) {
    return null;
  }
  const tools = settings?.tools ?? [];

  return (
    <SettingsSection
      title={t("settings.mcp.tools.title")}
      description={t("settings.mcp.tools.description")}
    >
      {error ? (
        <p className="pb-2 text-xs leading-snug text-destructive">{error}</p>
      ) : null}
      {tools.length > 0 ? (
        <div
          data-testid="mcp-tools"
          className="mt-3 flex flex-col rounded-xl border border-border/70 p-1.5"
        >
          {tools.map((tool) => (
            <ToolRow
              key={tool.name}
              tool={tool}
              disabled={saving}
              onToggle={(toolName, on) => void toggle(toolName, on)}
            />
          ))}
        </div>
      ) : null}
    </SettingsSection>
  );
}
