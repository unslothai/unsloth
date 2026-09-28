// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useId, useRef, useState } from "react";
import { MenuDismissGuard } from "@/lib/menu-dismiss-guard";
import { ChevronDownIcon, RotateCcwIcon } from "lucide-react";
import { BulbIcon } from "@/lib/bulb-icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  effortLabel,
  thinkingPresentation,
  type ThinkingCapabilities,
} from "../lib/thinking-presentation";
import type { ReasoningEffortLevel } from "../model-catalog";

export interface ThinkingControlProps {
  caps: ThinkingCapabilities;
  effort: ReasoningEffortLevel;
  enabled: boolean;
  disabled?: boolean;
  side?: "top" | "bottom";
  preserve?: boolean;
  onPreserveChange?: (enabled: boolean) => void;
  onEnabledChange: (enabled: boolean) => void;
  onEffortChange: (effort: ReasoningEffortLevel) => void;
  onReset: () => void;
  header?: ReactNode;
  footer?: ReactNode;
}

export function ThinkingControl({
  caps,
  effort,
  enabled,
  disabled,
  side = "top",
  preserve,
  onPreserveChange,
  onEnabledChange,
  onEffortChange,
  onReset,
  header,
  footer,
}: ThinkingControlProps) {
  const id = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const view = thinkingPresentation(caps);
  if (view.kind === "unsupported" && !onPreserveChange && !footer && !header)
    return null;
  const active =
    caps.supportsReasoning &&
    (!view.canDisable || (enabled && effort !== "none"));
  const selected = view.levels.includes(effort) ? effort : view.levels[0];
  const stateLabel =
    view.kind === "unknown"
      ? "Auto"
      : !active
        ? "Off"
        : view.kind === "fixed"
          ? `${effortLabel(selected)} · Fixed`
          : view.kind === "adjustable"
            ? effortLabel(selected)
            : view.kind === "always-on"
              ? "Always on"
              : "On";
  const label =
    view.kind === "unsupported"
      ? onPreserveChange
        ? "Preserve thinking"
        : "Model rates"
      : `Thinking · ${stateLabel}`;
  return (
    <Popover modal={false} open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          ref={triggerRef}
          type="button"
          disabled={disabled}
          className="unsloth-thinking-pill"
          data-pill-label={label}
          data-active={active || preserve ? "true" : "false"}
          aria-label={label}
        >
          <BulbIcon className="size-4 shrink-0" />
          <span className="unsloth-thinking-label">{label}</span>
          <ChevronDownIcon
            className="unsloth-thinking-caret size-[calc(15px*var(--ui-space-scale,1))] shrink-0"
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align="end"
        sideOffset={8}
        className="w-80 gap-4 rounded-2xl border border-border/60 p-4 shadow-lg"
        aria-label="Thinking settings"
      >
        {open ? <MenuDismissGuard triggerRef={triggerRef} /> : null}
        <div className="flex items-center justify-between gap-3">
          {header ?? (
            <span className="text-xs font-medium text-muted-foreground">
              Thinking effort
            </span>
          )}
          {caps.supportsReasoning && (
            <button
              type="button"
              onClick={onReset}
              disabled={disabled}
              className="flex size-9 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
              aria-label="Reset to model default"
              title="Reset to model default"
            >
              <RotateCcwIcon className="size-4" />
            </button>
          )}
        </div>
        {view.kind === "adjustable" ? (
          <>
            <Select
              value={selected}
              onValueChange={(value) =>
                onEffortChange(value as ReasoningEffortLevel)
              }
              disabled={disabled}
            >
              <SelectTrigger
                aria-label="Thinking effort"
                className="mx-auto w-auto min-w-32 border-0 bg-transparent text-lg shadow-none"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {view.levels.map((level) => (
                  <SelectItem key={level} value={level}>
                    {effortLabel(level)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="px-2 py-2">
              <Slider
                min={0}
                max={view.levels.length - 1}
                step={1}
                value={[Math.max(0, view.levels.indexOf(selected))]}
                onValueChange={([index]) => onEffortChange(view.levels[index])}
                disabled={disabled}
                aria-label="Thinking effort"
                thumbValueText={(index) => effortLabel(view.levels[index])}
                className="min-h-9 [&_[data-slot=slider-track]]:h-7 [&_[data-slot=slider-thumb]]:size-9"
              />
              <div
                className="mt-1 flex justify-between px-1"
                aria-hidden="true"
              >
                {view.levels.map((level) => (
                  <span
                    key={level}
                    className="size-1 rounded-full bg-muted-foreground/50"
                  />
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="text-center text-lg font-medium">
            {view.kind === "fixed"
              ? `${effortLabel(selected)} · Fixed`
              : view.kind === "always-on"
                ? "Always on"
                : view.kind === "unknown"
                  ? "Provider default"
                  : view.kind === "toggle"
                    ? active
                      ? "On"
                      : "Off"
                    : "Model settings"}
          </div>
        )}
        <p
          id={`${id}-description`}
          className="text-center text-xs leading-relaxed text-muted-foreground"
        >
          {view.description}
        </p>
        {view.canDisable && (
          <label className="flex min-h-9 items-center justify-between gap-3 text-sm">
            Thinking
            <Switch
              checked={active}
              onCheckedChange={onEnabledChange}
              disabled={disabled}
              aria-describedby={`${id}-description`}
              aria-label="Enable thinking"
            />
          </label>
        )}
        {onPreserveChange && (
          <label className="flex min-h-9 items-center justify-between gap-3 text-sm">
            Preserve thinking
            <Switch
              checked={preserve}
              onCheckedChange={onPreserveChange}
              disabled={disabled}
              aria-label="Preserve thinking"
            />
          </label>
        )}
        {footer}
      </PopoverContent>
    </Popover>
  );
}
