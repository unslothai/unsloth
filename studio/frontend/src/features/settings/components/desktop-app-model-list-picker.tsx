// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Search01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import type { DesktopAppModel } from "../api/claude-desktop";
import { toggleModel } from "./codex-desktop-state";

type Props = {
  id: string;
  value: string[];
  models: DesktopAppModel[];
  max: number;
  loadedModel: string | null;
  loading: boolean;
  disabled?: boolean;
  onOpen: () => void;
  onChange: (models: string[]) => void;
};

/** A list of Studio models, laid out like Ollama's ChatGPT model settings: chips with a remove
 *  button, and an "add" popover with "Find model...", keyboard navigation and a check on every
 *  listed model, holding up to ``max``. Empty follows whatever Studio has loaded. */
export function DesktopAppModelListPicker({
  id,
  value,
  models,
  max,
  loadedModel,
  loading,
  disabled,
  onOpen,
  onChange,
}: Props) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlighted, setHighlighted] = useState(-1);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const needle = query.trim().toLowerCase();
  const known = new Set(models.map((model) => model.id));
  // Listed models that are no longer downloaded still show, so they can be taken off the list.
  const options = [
    ...value
      .filter((model) => !known.has(model))
      .map((model) => ({ id: model, loaded: false, missing: true })),
    ...models.map((model) => ({
      id: model.id,
      loaded: model.loaded,
      missing: false,
    })),
  ].filter((option) => option.id.toLowerCase().includes(needle));
  const full = value.length >= max;
  const isDisabled = (option: { id: string }) =>
    full && !value.includes(option.id);
  const follow = loadedModel
    ? t("settings.agents.desktopApps.models.followLoadedCurrent", {
        model: loadedModel,
      })
    : t("settings.agents.desktopApps.models.followLoadedNone");

  useEffect(() => {
    if (highlighted >= 0) {
      optionRefs.current[highlighted]?.scrollIntoView({ block: "nearest" });
    }
  }, [highlighted]);

  const move = (direction: -1 | 1) => {
    setHighlighted((current) => {
      if (options.length === 0) {
        return -1;
      }
      const start = current < 0 ? (direction === 1 ? -1 : 0) : current;
      for (let step = 1; step <= options.length; step += 1) {
        const next =
          (start + direction * step + options.length) % options.length;
        if (!isDisabled(options[next])) {
          return next;
        }
      }
      return -1;
    });
  };
  const toggle = (model: string) => onChange(toggleModel(value, model, max));

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
      <PopoverAnchor asChild={true}>
        <div
          data-testid={`${id}-list`}
          className={cn(
            "relative flex min-h-8 w-full min-w-0 flex-wrap items-center gap-1.5 rounded-lg bg-muted/50 p-1 ring-1 ring-inset ring-border hover:bg-muted",
            disabled ? "opacity-50" : "",
          )}
        >
          <PopoverTrigger asChild={true}>
            <button
              id={id}
              type="button"
              aria-label={t("settings.agents.desktopApps.codex.addModel")}
              aria-haspopup="listbox"
              disabled={disabled}
              className="absolute inset-0 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed"
            />
          </PopoverTrigger>
          <div className="pointer-events-none relative flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
            {value.map((model, index) => (
              <span
                key={model}
                data-model={model}
                className={cn(
                  "inline-flex max-w-full items-stretch overflow-hidden rounded-md text-ui-12",
                  index === 0
                    ? "bg-accent text-accent-foreground"
                    : "bg-background text-foreground ring-1 ring-inset ring-border",
                )}
              >
                <span className="min-w-0 truncate py-0.5 pr-1 pl-2">
                  {model}
                </span>
                <button
                  type="button"
                  aria-label={t(
                    "settings.agents.desktopApps.codex.removeModel",
                    {
                      model,
                    },
                  )}
                  disabled={disabled}
                  onClick={(event) => {
                    event.stopPropagation();
                    toggle(model);
                  }}
                  className="pointer-events-auto inline-flex shrink-0 items-center px-1 hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed"
                >
                  <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                </button>
              </span>
            ))}
            {value.length === 0 ? (
              <span className="truncate px-1.5 py-0.5 text-ui-12 text-muted-foreground">
                {follow}
              </span>
            ) : null}
          </div>
        </div>
      </PopoverAnchor>
      <PopoverContent
        align="start"
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
            onChange={(event) => {
              setQuery(event.target.value);
              setHighlighted(-1);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                move(1);
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                move(-1);
              } else if (
                event.key === "Enter" &&
                highlighted >= 0 &&
                highlighted < options.length
              ) {
                event.preventDefault();
                toggle(options[highlighted].id);
              }
            }}
            placeholder={t("settings.agents.desktopApps.models.find")}
            aria-label={t("settings.agents.desktopApps.models.findLabel", {
              route: t("settings.agents.desktopApps.codex.name"),
            })}
            role="combobox"
            aria-expanded={true}
            aria-controls={`${id}-options`}
            aria-activedescendant={
              highlighted >= 0 ? `${id}-option-${highlighted}` : undefined
            }
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
        {full ? (
          <p className="border-b border-border/60 px-3 py-2 text-ui-11 text-muted-foreground">
            {t("settings.agents.desktopApps.codex.maxReached", {
              count: String(max),
            })}
          </p>
        ) : null}
        <div
          id={`${id}-options`}
          role="listbox"
          aria-multiselectable={true}
          className="max-h-72 min-h-0 overflow-y-auto py-1"
        >
          {options.map((option, index) => {
            const checked = value.includes(option.id);
            return (
              <button
                key={option.id}
                id={`${id}-option-${index}`}
                ref={(element) => {
                  optionRefs.current[index] = element;
                }}
                type="button"
                role="option"
                aria-selected={checked}
                disabled={isDisabled(option)}
                onClick={() => toggle(option.id)}
                onMouseEnter={() => setHighlighted(index)}
                className={cn(
                  "flex w-full cursor-pointer items-start gap-2 px-3 py-1.5 text-left text-ui-12 hover:bg-muted focus:bg-muted focus:outline-none disabled:cursor-not-allowed disabled:opacity-45",
                  highlighted === index ? "bg-muted" : "",
                )}
              >
                <span className="mt-0.5 size-3.5 shrink-0">
                  {checked ? (
                    <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
                  ) : null}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{option.id}</span>
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
