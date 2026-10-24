// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Search01Icon,
  Tick02Icon,
  UnfoldMoreIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import type { DesktopAppModel } from "../api/claude-desktop";
import { FOLLOW_LOADED } from "./desktop-app-models";

type Props = {
  id: string;
  routeName: string;
  value: string;
  models: DesktopAppModel[];
  loadedModel: string | null;
  loading: boolean;
  disabled?: boolean;
  placeholder?: string;
  onOpen: () => void;
  onChange: (model: string) => void;
};

/** One searchable model picker, laid out like Ollama's ClaudeModelPicker: a button showing the
 *  pick, and a popover with "Find model..." and a list with a check on the current pick. Its first
 *  option follows whatever Studio has loaded. */
export function DesktopAppModelPicker({
  id,
  routeName,
  value,
  models,
  loadedModel,
  loading,
  disabled,
  placeholder,
  onOpen,
  onChange,
}: Props) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const follow = loadedModel
    ? t("settings.agents.desktopApps.models.followLoadedCurrent", {
        model: loadedModel,
      })
    : t("settings.agents.desktopApps.models.followLoadedNone");
  const label = value === FOLLOW_LOADED ? follow : value;
  const needle = query.trim().toLowerCase();
  const options = [
    { id: FOLLOW_LOADED, name: follow, loaded: false, missing: false },
    ...models.map((model) => ({
      id: model.id,
      name: model.id,
      loaded: model.loaded,
      missing: false,
    })),
  ].filter((option) => option.name.toLowerCase().includes(needle));
  // A saved pick that is no longer downloaded still shows, so the user sees what a slot is on.
  if (
    value &&
    value !== FOLLOW_LOADED &&
    !models.some((model) => model.id === value) &&
    value.toLowerCase().includes(needle)
  ) {
    options.splice(1, 0, {
      id: value,
      name: value,
      loaded: false,
      missing: true,
    });
  }

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          setQuery("");
          onOpen();
        }
      }}
    >
      <PopoverTrigger asChild={true}>
        <button
          id={id}
          type="button"
          aria-label={t("settings.agents.desktopApps.models.pickerLabel", {
            route: routeName,
          })}
          aria-haspopup="listbox"
          disabled={disabled}
          data-value={value}
          className="flex min-h-8 w-full min-w-0 items-center gap-2 rounded-lg bg-muted/50 px-2.5 py-1 text-left text-ui-12 text-foreground outline-none ring-1 ring-inset ring-border hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span
            className={cn(
              "min-w-0 flex-1 truncate",
              value ? "" : "text-muted-foreground",
            )}
          >
            {label ||
              placeholder ||
              t("settings.agents.desktopApps.models.select")}
          </span>
          <HugeiconsIcon
            icon={UnfoldMoreIcon}
            className="size-3.5 shrink-0 text-muted-foreground"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={6}
        className="w-(--radix-popover-trigger-width) min-w-64 gap-0 overflow-hidden p-0"
      >
        <div className="flex flex-none items-center gap-2 border-b border-border/60 px-3 py-2">
          <HugeiconsIcon
            icon={Search01Icon}
            className="size-3.5 shrink-0 text-muted-foreground"
          />
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("settings.agents.desktopApps.models.find")}
            aria-label={t("settings.agents.desktopApps.models.findLabel", {
              route: routeName,
            })}
            autoCorrect="off"
            autoComplete="off"
            className="min-w-0 flex-1 border-none bg-transparent py-0.5 text-ui-12 outline-none"
          />
        </div>
        {loading ? (
          <p
            role="status"
            className="px-3 py-2 text-ui-12 text-muted-foreground"
          >
            {t("settings.agents.desktopApps.models.loading")}
          </p>
        ) : null}
        <div role="listbox" className="max-h-72 min-h-0 overflow-y-auto py-1">
          {options.map((option) => {
            const selected = option.id === value;
            return (
              <button
                key={option.id}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  onChange(option.id);
                  setOpen(false);
                }}
                className="flex w-full cursor-pointer items-start gap-2 px-3 py-1.5 text-left text-ui-12 hover:bg-muted focus:bg-muted focus:outline-none"
              >
                <span className="mt-0.5 size-3.5 shrink-0">
                  {selected ? (
                    <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
                  ) : null}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{option.name}</span>
                  {option.missing ? (
                    <span className="mt-0.5 block truncate text-ui-10 text-muted-foreground">
                      {t("settings.agents.desktopApps.models.notDownloaded")}
                    </span>
                  ) : null}
                </span>
                {option.loaded ? (
                  <span className="shrink-0 rounded-full bg-emerald-500/10 px-1.5 text-ui-10 text-emerald-700 dark:text-emerald-400">
                    {t("settings.agents.desktopApps.models.loadedTag")}
                  </span>
                ) : null}
              </button>
            );
          })}
          {options.length === 0 ? (
            <p className="px-3 py-2 text-ui-12 text-muted-foreground">
              {t("settings.agents.desktopApps.models.empty")}
            </p>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}
