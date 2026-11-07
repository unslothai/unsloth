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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { AiSpeechIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
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
import {
  type ReactElement,
  useEffect,
  useState,
} from "react";
import type { McpToolSetting } from "../api/mcp-access";
import {
  groupMcpTools,
  mcpToolDescriptionKey,
  mcpToolLabelKey,
} from "./mcp-tool-labels";

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

/** One row for many tools: the whole row is the button, and its right side says what a click does. */
function GroupToggle({
  tools,
  label,
  prominent = false,
  onSet,
}: {
  tools: readonly McpToolSetting[];
  label: string;
  prominent?: boolean;
  onSet: (names: readonly string[], on: boolean) => void;
}): ReactElement {
  const t = useT();
  const on = tools.filter((tool) => tool.enabled).length;
  const allOn = on === tools.length;
  return (
    <button
      type="button"
      // Some or none on turns every tool on; only an all-on group turns off.
      onClick={() =>
        onSet(
          tools
            .filter((tool) => tool.enabled === allOn)
            .map((tool) => tool.name),
          !allOn,
        )
      }
      className={cn(
        "group/toggle flex shrink-0 cursor-pointer items-center gap-2 rounded-full pe-3 text-start outline-none transition-colors duration-150 focus-visible:ring-3 focus-visible:ring-ring/40 motion-reduce:transition-none",
        prominent
          ? "min-h-11 gap-2.5 bg-wash-4 ps-4 hover:bg-wash-6"
          : "min-h-8 ps-3 hover:bg-wash-4",
      )}
    >
      <span
        className={cn(
          "text-ui-14 text-foreground",
          prominent ? "font-medium" : "font-semibold",
        )}
      >
        {label}
      </span>
      <span className="text-ui-12 tabular-nums text-muted-foreground">
        {t("settings.mcp.tools.count", { on, total: tools.length })}
      </span>
      <span className="ms-auto text-ui-13 text-muted-foreground transition-colors duration-150 group-hover/toggle:text-foreground motion-reduce:transition-none">
        {t(
          allOn
            ? "settings.mcp.tools.selectNone"
            : "settings.mcp.tools.selectAll",
        )}
      </span>
    </button>
  );
}

function ToolChip({
  tool,
  onToggle,
}: {
  tool: McpToolSetting;
  onToggle: (on: boolean) => void;
}): ReactElement {
  const t = useT();
  const labelKey = mcpToolLabelKey(tool.name);
  const descriptionKey = mcpToolDescriptionKey(tool.name);
  const label = labelKey ? t(labelKey) : tool.name;
  const on = tool.enabled;
  // The tick, not the color, carries the state.
  const chip = (
    <button
      type="button"
      role="checkbox"
      aria-checked={on}
      data-mcp-tool={tool.name}
      onClick={() => onToggle(!on)}
      className={cn(
        "relative flex min-h-9 w-full min-w-0 cursor-pointer select-none items-center gap-2.5 rounded-full py-1.5 ps-3.5 pe-8 text-start outline-none transition-colors duration-150 focus-visible:ring-3 focus-visible:ring-ring/40 motion-reduce:transition-none",
        on
          ? "bg-wash-9 text-foreground dark:bg-wash-14"
          : "bg-wash-4 text-muted-foreground hover:bg-wash-6 hover:text-foreground",
      )}
    >
      <span className="flex size-4 shrink-0 items-center justify-center [&_svg]:size-4">
        <HugeiconsIcon icon={toolIcon(tool.name)} strokeWidth={1.75} />
      </span>
      <span
        className={cn(
          "min-w-0 flex-1 truncate text-ui-13",
          labelKey ? null : "font-mono",
        )}
      >
        {label}
      </span>
      {on ? (
        <HugeiconsIcon
          icon={Tick02Icon}
          strokeWidth={2}
          aria-hidden={true}
          className="pointer-events-none absolute end-2.5 size-4"
        />
      ) : null}
    </button>
  );
  // A tool this build doesn't know already shows its raw name and has nothing to add.
  if (!descriptionKey) {
    return chip;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{chip}</TooltipTrigger>
      <TooltipContent className="flex max-w-64 flex-col gap-0.5 text-start">
        <span>{t(descriptionKey)}</span>
        <span className="font-mono text-ui-11 opacity-60">{tool.name}</span>
      </TooltipContent>
    </Tooltip>
  );
}

function useMoreBelow(): [(el: HTMLDivElement | null) => void, boolean] {
  // A callback ref: the dialog content mounts after this component's first effect.
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [more, setMore] = useState(false);
  useEffect(() => {
    if (!el) {
      return;
    }
    const update = () =>
      setMore(el.scrollTop + el.clientHeight < el.scrollHeight - 1);
    update();
    el.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [el]);
  return [setEl, more];
}

/** Grouped tool chips for the tools served at /mcp/. Saves each change at once. */
export function McpToolsDialog({
  open,
  onOpenChange,
  tools,
  error,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tools: readonly McpToolSetting[];
  error: string | null;
  /** Receives the tools to change, by name; the server merges them into its list. */
  onSave: (states: Record<string, boolean>) => void;
}): ReactElement {
  const t = useT();
  const groups = groupMcpTools(tools);
  const [listRef, moreBelow] = useMoreBelow();

  // Send only the tools that change, so a toggle elsewhere at the same time still lands.
  const setEnabled = (names: readonly string[], on: boolean) => {
    if (names.length > 0) {
      onSave(Object.fromEntries(names.map((name) => [name, on])));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Header and footer stay put; only the tool list scrolls, and only on short windows. */}
      <DialogContent
        className="flex flex-col gap-5 overflow-hidden outline-none sm:max-w-2xl"
        // Focus the dialog itself: Enter on "All tools" would flip every tool, and a focused
        // chip would pop its tooltip. Tab still lands on the first control.
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          (event.currentTarget as HTMLElement).focus();
        }}
      >
        <DialogHeader className="pe-8">
          <DialogTitle>{t("settings.mcp.tools.dialogTitle")}</DialogTitle>
          <DialogDescription>
            {t("settings.mcp.tools.description")}
          </DialogDescription>
        </DialogHeader>

        {error ? (
          <p className="text-xs leading-snug text-destructive">{error}</p>
        ) : null}

        <div
          ref={listRef}
          data-testid="mcp-tools"
          className="@container -mx-7 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto scroll-rounded px-7"
        >
          <GroupToggle
            tools={tools}
            label={t("settings.mcp.tools.allTools")}
            prominent={true}
            onSet={setEnabled}
          />
          {groups.map((group) => (
              <section
                key={group.id}
                data-mcp-tool-group={group.id}
                aria-label={t(group.labelKey)}
                className="flex flex-col gap-1.5"
              >
                <GroupToggle
                  tools={group.tools}
                  label={t(group.labelKey)}
                  onSet={setEnabled}
                />
                <div className="grid grid-cols-1 gap-1.5 @sm:grid-cols-2 @xl:grid-cols-3">
                  {group.tools.map((tool) => (
                    <ToolChip
                      key={tool.name}
                      tool={tool}
                      onToggle={(next) => setEnabled([tool.name], next)}
                    />
                  ))}
                </div>
              </section>
          ))}
        </div>

        {/* A rule shows only while more of the list is below, so nothing hides behind the footer. */}
        <DialogFooter
          className={cn(
            "-mx-7 -mt-5 flex-row justify-end border-t px-7 pt-5",
            moreBelow ? "border-border" : "border-transparent",
          )}
        >
          <Button variant="dark" onClick={() => onOpenChange(false)}>
            {t("common.done")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
