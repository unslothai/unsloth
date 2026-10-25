// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import type { PerModelConfig } from "../model-config/per-model-config";
import {
  SAVED_RUN_SETTINGS_NAME_MAX,
  SAVED_RUN_SETTINGS_PER_MODEL_MAX,
  deleteSavedRunSettings,
  findSavedRunSettings,
  normalizeSavedRunSettingsName,
  saveRunSettings,
  useSavedRunSettings,
} from "../model-config/saved-run-settings";

interface SavedRunSettingsRowProps {
  modelId: string;
  disabled?: boolean;
  /** What Save stores: the panel's settings with any typed numbers committed. */
  currentConfig: () => PerModelConfig;
  onApply: (config: PerModelConfig) => void;
}

/** Laid out like the sidebar's Preset row, so picking, saving and deleting read the same. */
export function SavedRunSettingsRow({
  modelId,
  disabled = false,
  currentConfig,
  onApply,
}: SavedRunSettingsRowProps) {
  const saved = useSavedRunSettings(modelId);
  const [name, setName] = useState("");
  const match = findSavedRunSettings(saved, name);
  const cleanName = normalizeSavedRunSettingsName(name);
  const full = !match && saved.length >= SAVED_RUN_SETTINGS_PER_MODEL_MAX;

  const save = () => {
    if (!cleanName || disabled) {
      return;
    }
    if (full) {
      toast.error(
        `Up to ${SAVED_RUN_SETTINGS_PER_MODEL_MAX} saved settings per model. Delete one first.`,
      );
      return;
    }
    if (!saveRunSettings(modelId, cleanName, currentConfig())) {
      toast.error("Couldn't save these settings.");
      return;
    }
    setName(cleanName);
    toast.success(match ? `Updated "${cleanName}".` : `Saved "${cleanName}".`);
  };

  const remove = () => {
    if (!match || disabled) {
      return;
    }
    if (!deleteSavedRunSettings(modelId, match.name)) {
      toast.error("Couldn't delete these settings.");
      return;
    }
    setName("");
    toast.success(`Deleted "${match.name}".`);
  };

  return (
    <div className="flex flex-col gap-2">
      <span className="text-ui-13 text-foreground">Saved settings</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true} disabled={saved.length === 0}>
          <div
            className="w-full min-w-0 outline-none focus-visible:outline-none"
            aria-label="Open saved settings"
          >
            <InputGroup className="panel-input-group">
              <InputGroupInput
                value={name}
                onChange={(event) => setName(event.target.value)}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    save();
                  }
                  event.stopPropagation();
                }}
                placeholder={
                  saved.length ? "Pick or name settings" : "Name these settings"
                }
                maxLength={SAVED_RUN_SETTINGS_NAME_MAX}
                autoComplete="off"
                disabled={disabled}
                className="!h-9 min-h-0 min-w-0 self-stretch !pl-3.5 !pr-2 py-0 text-ui-13 font-medium leading-9 md:text-ui-13"
                aria-label="Saved settings name"
              />
              {saved.length > 0 && (
                <InputGroupAddon
                  align="inline-end"
                  className="min-h-0 shrink-0 cursor-pointer gap-0 self-stretch border-0 py-0 pl-0 !pr-1"
                >
                  <span
                    className="pointer-events-none inline-flex !h-7 !w-7 shrink-0 items-center justify-center self-center rounded-full text-muted-foreground"
                    aria-hidden="true"
                  >
                    <HugeiconsIcon
                      icon={ChevronDownStandardIcon}
                      className="size-3.5"
                      strokeWidth={2}
                    />
                  </span>
                </InputGroupAddon>
              )}
            </InputGroup>
          </div>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          sideOffset={4}
          className="menu-soft-surface max-h-[min(18rem,var(--radix-dropdown-menu-content-available-height))] rounded-lg border-0 p-1 ring-0"
        >
          {saved.map((entry) => (
            <DropdownMenuItem
              key={entry.name}
              disabled={disabled}
              onSelect={() => {
                setName(entry.name);
                onApply(entry.config);
              }}
              className="flex min-h-9 items-center px-3 py-0 text-ui-13 font-medium"
            >
              {entry.name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled || !cleanName}
          onClick={save}
        >
          {match ? "Update" : "Save"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-destructive"
          disabled={disabled || !match}
          onClick={remove}
        >
          Delete
        </Button>
      </div>
    </div>
  );
}
