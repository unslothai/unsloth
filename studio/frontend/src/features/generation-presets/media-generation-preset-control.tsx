// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { Bookmark, Check, ChevronDown, Trash2 } from "lucide-react";
import { Fragment, useState } from "react";
import { DEFAULT_PRESET_NAME } from "./preset-policy";

interface MediaGenerationPresetControlProps {
  kind: "image" | "video";
  presets: readonly { name: string }[];
  activePreset: string;
  ready: boolean;
  hasUnsavedChanges: boolean;
  onSelect: (name: string) => void;
  onSave: (name: string) => Promise<string | null>;
  onDelete: () => Promise<boolean>;
}

export function MediaGenerationPresetControl({
  kind,
  presets,
  activePreset,
  ready,
  hasUnsavedChanges,
  onSelect,
  onSave,
  onDelete,
}: MediaGenerationPresetControlProps) {
  const [open, setOpen] = useState(false);
  const [draftName, setDraftName] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const name = draftName ?? activePreset;
  const trimmed = name.trim();
  const matching = presets.find((preset) => preset.name === trimmed);
  const activeIsCustom =
    activePreset !== DEFAULT_PRESET_NAME &&
    presets.some((preset) => preset.name === activePreset);
  const canSave =
    ready &&
    !saving &&
    trimmed.length > 0 &&
    (trimmed !== activePreset ||
      hasUnsavedChanges ||
      !matching ||
      trimmed === DEFAULT_PRESET_NAME);
  let saveLabel = "Saved";
  if (trimmed === DEFAULT_PRESET_NAME) {
    saveLabel = "Save copy";
  } else if (!matching) {
    saveLabel = "Save";
  } else if (matching.name !== activePreset) {
    saveLabel = "Overwrite";
  } else if (hasUnsavedChanges) {
    saveLabel = "Update";
  }
  const changeOpen = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (!nextOpen) setDraftName(null);
  };

  const save = async () => {
    setSaving(true);
    try {
      const saved = await onSave(name);
      if (saved) {
        changeOpen(false);
      }
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    setSaving(true);
    try {
      if (await onDelete()) {
        changeOpen(false);
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <Popover open={open} onOpenChange={changeOpen}>
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!ready || saving}
          aria-label={`Manage ${kind} generation presets`}
          className={cn(
            "relative size-8 shrink-0 gap-1.5 rounded-full border-border/60 bg-background/70 p-0 text-xs font-medium shadow-none backdrop-blur-sm hover:bg-muted/70 sm:h-8 sm:w-auto sm:max-w-40 sm:px-2.5",
            open && "border-border bg-muted/70",
          )}
          data-testid={`${kind}-generation-presets`}
        >
          <Bookmark className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="hidden min-w-0 truncate sm:block">
            {activePreset}
          </span>
          {hasUnsavedChanges ? (
            <span
              className="absolute right-1 top-1 size-1.5 shrink-0 rounded-full bg-amber-500 ring-2 ring-background sm:static sm:ring-0"
              aria-label="Modified"
            />
          ) : null}
          <ChevronDown
            className={cn(
              "hidden size-3.5 shrink-0 text-muted-foreground transition-transform sm:block",
              open && "rotate-180",
            )}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        align="end"
        sideOffset={8}
        collisionPadding={12}
        className="max-h-[var(--radix-popover-content-available-height)] w-[min(320px,calc(100vw-24px))] gap-0 overflow-x-hidden overflow-y-auto overscroll-contain rounded-xl border-border/70 p-0 shadow-xl"
      >
        <div className="shrink-0 border-b border-border/60 px-4 py-3.5">
          <div className="flex items-center justify-between gap-3">
            <p className="font-heading text-sm font-medium">
              Generation presets
            </p>
            {hasUnsavedChanges ? (
              <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-ui-10 font-medium text-amber-700 dark:text-amber-300">
                Modified
              </span>
            ) : null}
          </div>
        </div>

        <div className="max-h-48 shrink-0 overflow-y-auto p-2">
          {presets.map((preset, index) => {
            const isActive = preset.name === activePreset;

            return (
              <Fragment key={preset.name}>
                {index === 1 ? (
                  <div className="mx-2 my-1 h-px bg-border/60" />
                ) : null}
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => {
                    onSelect(preset.name);
                    changeOpen(false);
                  }}
                  className={cn(
                    "flex h-9 w-full items-center gap-2 rounded-lg px-2.5 text-left text-xs transition-colors hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                    isActive && "bg-muted font-medium",
                  )}
                >
                  <span className="min-w-0 flex-1 truncate">{preset.name}</span>
                  {preset.name === DEFAULT_PRESET_NAME ? (
                    <span className="text-ui-10 font-normal text-muted-foreground">
                      Built-in
                    </span>
                  ) : null}
                  {isActive ? (
                    <Check className="size-3.5 shrink-0 text-primary" />
                  ) : null}
                </button>
              </Fragment>
            );
          })}
        </div>

        <div className="shrink-0 border-t border-border/60 bg-muted/20 p-3">
          <label
            htmlFor={`${kind}-generation-preset-name`}
            className="mb-2 block text-xs font-medium"
          >
            Save current settings
          </label>
          <div className="flex gap-2">
            <Input
              id={`${kind}-generation-preset-name`}
              value={name}
              disabled={saving}
              onChange={(event) => setDraftName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && canSave) {
                  event.preventDefault();
                  save().catch(() => undefined);
                }
              }}
              placeholder="Preset name"
              maxLength={80}
              autoComplete="off"
              className="h-9 min-w-0 rounded-lg bg-background"
            />
            <Button
              type="button"
              size="sm"
              disabled={!canSave}
              onClick={() => save().catch(() => undefined)}
              className="h-9 shrink-0 rounded-lg px-3"
            >
              {saveLabel}
            </Button>
          </div>
          {activeIsCustom ? (
            <button
              type="button"
              disabled={saving}
              onClick={() => remove().catch(() => undefined)}
              className="mt-3 flex h-8 w-full items-center justify-center gap-1.5 rounded-lg text-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <Trash2 className="size-3.5" />
              Delete preset
            </button>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}
