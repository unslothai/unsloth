// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
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
import type { ReactElement } from "react";
import type { McpToolSetting } from "../api/mcp-access";
import { groupMcpTools, mcpToolLabelKey } from "./mcp-tool-labels";

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

function ToolRow({
  tool,
  disabled,
  onToggle,
}: {
  tool: McpToolSetting;
  disabled: boolean;
  onToggle: (on: boolean) => void;
}): ReactElement {
  const t = useT();
  const labelKey = mcpToolLabelKey(tool.name);
  const label = labelKey ? t(labelKey) : tool.name;
  // The whole row is the switch's label, so clicking the name toggles it too.
  return (
    <label
      data-mcp-tool={tool.name}
      className="flex min-h-9 cursor-pointer items-center gap-2.5 rounded-lg px-2 py-1 has-disabled:cursor-default"
    >
      <span className="flex size-4 shrink-0 items-center justify-center text-foreground/80 [&_svg]:size-4">
        <HugeiconsIcon icon={toolIcon(tool.name)} strokeWidth={1.75} />
      </span>
      {/* The raw tool name is for debugging, so it waits behind a hover. */}
      {labelKey ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <span className="min-w-0 flex-1 truncate text-ui-13 text-foreground">
              {label}
            </span>
          </TooltipTrigger>
          <TooltipContent className="font-mono text-ui-11">
            {tool.name}
          </TooltipContent>
        </Tooltip>
      ) : (
        <span className="min-w-0 flex-1 truncate font-mono text-ui-13 text-foreground">
          {label}
        </span>
      )}
      <Switch
        size="sm"
        aria-label={label}
        checked={tool.enabled}
        disabled={disabled}
        onCheckedChange={onToggle}
      />
    </label>
  );
}

/** Grouped on/off switches for the tools served at /mcp/. Saves each change at once. */
export function McpToolsDialog({
  open,
  onOpenChange,
  tools,
  saving,
  error,
  onSave,
  footnote,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tools: readonly McpToolSetting[];
  saving: boolean;
  error: string | null;
  /** Receives the tools to change, by name; the server merges them into its list. */
  onSave: (states: Record<string, boolean>) => void;
  /** muted line under the groups */
  footnote?: string | null;
}): ReactElement {
  const t = useT();
  const groups = groupMcpTools(tools);
  const anyOff = tools.some((tool) => !tool.enabled);

  // Send only the tools that change, so a toggle elsewhere at the same time still lands.
  const setEnabled = (names: readonly string[], on: boolean) => {
    onSave(Object.fromEntries(names.map((name) => [name, on])));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t("settings.mcp.tools.dialogTitle")}</DialogTitle>
          <DialogDescription>
            {t("settings.mcp.tools.description")}
          </DialogDescription>
        </DialogHeader>

        <div data-testid="mcp-tools" className="flex flex-col gap-3">
          {error ? (
            <p className="text-xs leading-snug text-destructive">{error}</p>
          ) : null}
          {groups.map((group) => {
            const on = group.tools.filter((tool) => tool.enabled).length;
            const label = t(group.labelKey);
            return (
              <section
                key={group.id}
                data-mcp-tool-group={group.id}
                className="flex flex-col"
              >
                <div className="flex items-center gap-2.5 px-3 pb-1">
                  <h3 className="min-w-0 flex-1 text-ui-12 font-medium text-muted-foreground">
                    {label}
                    <span className="ml-1.5 tabular-nums">
                      {on}/{group.tools.length}
                    </span>
                  </h3>
                  {/* On only when every tool in the group is on; flipping it sets them all. */}
                  <Switch
                    size="sm"
                    aria-label={t("settings.mcp.tools.groupToggle", {
                      group: label,
                    })}
                    checked={on === group.tools.length}
                    disabled={saving}
                    onCheckedChange={(next) =>
                      setEnabled(
                        group.tools.map((tool) => tool.name),
                        next,
                      )
                    }
                  />
                </div>
                <div className="flex flex-col rounded-xl border border-border/70 p-1">
                  {group.tools.map((tool) => (
                    <ToolRow
                      key={tool.name}
                      tool={tool}
                      disabled={saving}
                      onToggle={(next) => setEnabled([tool.name], next)}
                    />
                  ))}
                </div>
              </section>
            );
          })}
          {footnote ? (
            <p className="px-3 text-xs leading-snug text-muted-foreground">
              {footnote}
            </p>
          ) : null}
        </div>

        <DialogFooter>
          {/* Disabled rather than hidden, so the footer doesn't shift while toggling. */}
          <Button
            variant="outline"
            disabled={saving || !anyOff}
            onClick={() =>
              setEnabled(
                tools.filter((tool) => !tool.enabled).map((tool) => tool.name),
                true,
              )
            }
          >
            {t("settings.mcp.tools.turnAllOn")}
          </Button>
          <Button onClick={() => onOpenChange(false)}>
            {t("common.done")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
