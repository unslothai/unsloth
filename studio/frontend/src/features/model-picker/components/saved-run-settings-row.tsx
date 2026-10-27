// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Edit03Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type MouseEvent,
  type PointerEvent,
  useEffect,
  useState,
} from "react";
import {
  SAVED_RUN_SETTINGS_NAME_MAX,
  type SavedRunSettings,
  normalizeSavedRunSettingsName,
} from "../model-config/saved-run-settings";
import { Tick02Icon } from "@/lib/tick-icon";

/** ``target`` is the set being renamed. */
export type RunSettingsNaming =
  | { mode: "new"; name: string }
  | { mode: "rename"; name: string; target: string };

interface RunSettingsPickerProps {
  modelName: string;
  sets: readonly SavedRunSettings[];
  /** The set the model loads with, or null for the default (recommended) settings. */
  selected: string | null;
  pinned?: string | null;
  /** Default with edits: the name they will be saved under. */
  customName: string | null;
  /** Null on Default settings, which cannot be renamed. */
  onEditName: (() => void) | null;
  /** The narrow sidebar: a smaller pencil, and the name box's buttons below it. */
  compact?: boolean;
  naming: RunSettingsNaming | null;
  disabled?: boolean;
  onPick: (name: string | null) => void;
  onNamingChange: (naming: RunSettingsNaming | null) => void;
  onNamingSubmit: () => void;
  onDelete: (name: string) => void;
}

const ITEM = "flex min-h-9 items-center gap-2 px-3 py-0 text-ui-13 font-medium";
// The default bg-border/50 vanishes on the dark menu surface.
const SEPARATOR =
  "mx-3 bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:bg-[rgb(255_255_255_/_calc(0.16*var(--contrast-wash-gain,1)))]";
const SECTION = "pb-1 pt-2 text-ui-11 font-medium";
const ROW_ICON =
  "flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

function Check({ on }: { on: boolean }) {
  return (
    <HugeiconsIcon
      icon={Tick02Icon}
      strokeWidth={2}
      className={cn("size-3.5 shrink-0", !on && "invisible")}
    />
  );
}

// Clicking an icon must not also pick its row: Radix selects on pointerup when it missed the pointerdown.
const keepRow = (event: MouseEvent | PointerEvent) => {
  event.preventDefault();
  event.stopPropagation();
};

export function RunSettingsPicker({
  modelName,
  sets,
  selected,
  pinned = null,
  customName,
  onEditName,
  compact = false,
  naming,
  disabled = false,
  onPick,
  onNamingChange,
  onNamingSubmit,
  onDelete,
}: RunSettingsPickerProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const isNaming = naming !== null;
  useEffect(() => {
    if (!isNaming) {
      return;
    }
    // Window capture runs before Radix's document listener, which then skips the prevented Escape.
    const keepPanel = (event: KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        event.target instanceof Element &&
        event.target.closest("[data-run-settings-name]")
      ) {
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", keepPanel, true);
    return () => window.removeEventListener("keydown", keepPanel, true);
  }, [isNaming]);
  if (naming) {
    return (
      // The sidebar is too narrow for a readable name beside the buttons, so they go below it.
      <div
        className={cn(
          "flex min-w-0 gap-2",
          compact ? "flex-col" : "items-center",
        )}
      >
        <Input
          autoFocus={true}
          data-run-settings-name=""
          value={naming.name}
          onFocus={(event) => event.currentTarget.select()}
          onChange={(event) =>
            onNamingChange({ ...naming, name: event.target.value })
          }
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onNamingSubmit();
            } else if (event.key === "Escape") {
              event.preventDefault();
              event.stopPropagation();
              onNamingChange(null);
            }
          }}
          maxLength={SAVED_RUN_SETTINGS_NAME_MAX}
          autoComplete="off"
          aria-label={
            naming.mode === "new"
              ? "Name for the new settings"
              : `New name for ${naming.target}`
          }
          className={cn(
            "h-9 min-w-0 rounded-full text-ui-13",
            compact ? "w-full" : "flex-1",
          )}
        />
        <div className="flex shrink-0 items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={normalizeSavedRunSettingsName(naming.name) === ""}
            onClick={onNamingSubmit}
          >
            Save
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
            onClick={() => onNamingChange(null)}
          >
            Cancel
          </Button>
        </div>
      </div>
    );
  }
  // Closed first: the rename box replaces the menu, and the delete confirmation must not open under it.
  const rename = (name: string) => {
    setMenuOpen(false);
    onNamingChange({ mode: "rename", name, target: name });
  };
  const remove = (name: string) => {
    setMenuOpen(false);
    onDelete(name);
  };
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
        <DropdownMenuTrigger asChild={true} disabled={disabled}>
          <button
            type="button"
            aria-label="Run settings for this model"
            className="flex h-9 w-full min-w-0 items-center justify-between gap-1.5 rounded-full border border-[color-mix(in_oklab,var(--foreground)_calc(15%*var(--contrast-edge-gain,1)),transparent)] bg-background px-3.5 text-left text-ui-13 font-medium text-foreground outline-none transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
          >
            <span className="min-w-0 truncate">
              {selected ?? customName ?? "Default settings"}
            </span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className="size-4 shrink-0 text-muted-foreground"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          sideOffset={4}
          className="menu-soft-surface max-h-[min(20rem,var(--radix-dropdown-menu-content-available-height))] w-max min-w-[var(--radix-dropdown-menu-trigger-width)] max-w-[min(36rem,calc(100vw-2rem))] rounded-lg border-0 p-1 ring-0"
        >
          <DropdownMenuLabel className={SECTION}>Default</DropdownMenuLabel>
          <DropdownMenuItem className={ITEM} onSelect={() => onPick(null)}>
            <Check on={selected === null && customName === null} />
            <span className="min-w-0 truncate">
              Default settings for {modelName}
            </span>
          </DropdownMenuItem>
          {sets.length > 0 && (
            <>
              <DropdownMenuSeparator className={SEPARATOR} />
              <DropdownMenuLabel className={SECTION}>
                Custom settings
              </DropdownMenuLabel>
            </>
          )}
          {sets.map((entry) => (
            <DropdownMenuItem
              key={entry.name}
              className={cn(ITEM, "pr-1.5")}
              onSelect={() => onPick(entry.name)}
              aria-keyshortcuts="F2 Delete"
              onKeyDown={(event) => {
                if (event.key === "F2") {
                  event.preventDefault();
                  rename(entry.name);
                } else if (event.key === "Delete") {
                  event.preventDefault();
                  remove(entry.name);
                }
              }}
            >
              <Check on={entry.name === selected} />
              <span className="min-w-0 flex-1 truncate">{entry.name}</span>
              {entry.name === pinned && (
                <span className="shrink-0 text-ui-11 font-normal text-muted-foreground">
                  Always
                </span>
              )}
              <button
                type="button"
                aria-label={`Rename ${entry.name}`}
                title="Rename (F2)"
                tabIndex={-1}
                className={ROW_ICON}
                onPointerDown={keepRow}
                onPointerUp={keepRow}
                onClick={(event) => {
                  keepRow(event);
                  rename(entry.name);
                }}
              >
                <HugeiconsIcon
                  icon={Edit03Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </button>
              <button
                type="button"
                aria-label={`Delete ${entry.name}`}
                title="Delete (Del)"
                tabIndex={-1}
                className={ROW_ICON}
                onPointerDown={keepRow}
                onPointerUp={keepRow}
                onClick={(event) => {
                  keepRow(event);
                  remove(entry.name);
                }}
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </button>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
      {onEditName && (
        <button
          type="button"
          aria-label="Rename this setting"
          title="Rename"
          disabled={disabled}
          onClick={onEditName}
          className={cn(ROW_ICON, "size-9 rounded-full disabled:opacity-50")}
        >
          <HugeiconsIcon
            icon={Edit03Icon}
            strokeWidth={1.75}
            className={compact ? "size-3.5" : "size-4"}
          />
        </button>
      )}
    </div>
  );
}
