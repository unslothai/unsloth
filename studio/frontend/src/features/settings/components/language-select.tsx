// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import {
  AUTO_LOCALE,
  LOCALES,
  isLocalePreference,
  isSupportedLocale,
  setLocale,
  useLocale,
  useLocaleCatalogFailed,
  useLocalePreference,
  usePendingLocalePreference,
  useT,
} from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";

const SORTED_LOCALES = Object.entries(LOCALES).sort(([, a], [, b]) =>
  a.label.localeCompare(b.label, "en"),
);

export function LanguageSelect() {
  const t = useT();
  const locale = useLocale();
  const preference = useLocalePreference();
  const pendingPreference = usePendingLocalePreference();
  const catalogFailed = useLocaleCatalogFailed();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlighted, setHighlighted] = useState("");
  const [scrolled, setScrolled] = useState(false);
  // After a catalog failure check no entry, so both the failed preference and the fallback stay pickable.
  const checked = pendingPreference ?? (catalogFailed ? "" : preference);
  const autoLabel = t("settings.appearance.language.autoDetect");
  const shownLabel = catalogFailed
    ? LOCALES[locale].nativeLabel
    : checked === AUTO_LOCALE
      ? autoLabel
      : isSupportedLocale(checked)
        ? LOCALES[checked].nativeLabel
        : "";

  const close = (next: boolean) => {
    setOpen(next);
    if (next) {
      setHighlighted(checked || locale);
    } else {
      setQuery("");
      setScrolled(false);
    }
  };
  // Not cmdk's filter: its sort throws on rows inside the scroll div.
  const needle = query.trim().toLowerCase();
  const matches = (...names: string[]) =>
    !needle || names.some((name) => name.toLowerCase().includes(needle));
  const showAuto = matches(AUTO_LOCALE, autoLabel);
  const shownLocales = SORTED_LOCALES.filter(([value, metadata]) =>
    matches(value, metadata.nativeLabel, metadata.label),
  );
  const search = (next: string) => {
    setQuery(next);
    const term = next.trim().toLowerCase();
    const first = SORTED_LOCALES.find(([value, metadata]) =>
      [value, metadata.nativeLabel, metadata.label].some((name) =>
        name.toLowerCase().includes(term),
      ),
    );
    setHighlighted(
      term && !autoLabel.toLowerCase().includes(term)
        ? (first?.[0] ?? "")
        : AUTO_LOCALE,
    );
  };
  const pick = (value: string) => {
    close(false);
    if (value !== checked && isLocalePreference(value)) {
      setLocale(value);
    }
  };

  return (
    // Non-modal gets no wheel scrolling inside the Settings dialog.
    <Popover open={open} onOpenChange={close} modal={true}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("settings.appearance.language.label")}
          aria-expanded={open}
          aria-busy={pendingPreference !== null}
          className="flex h-8 w-48 cursor-pointer items-center justify-between gap-1.5 rounded-full border border-border bg-background px-3.5 text-sm outline-none transition-colors hover:bg-accent/50 focus-visible:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:focus-visible:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]"
        >
          <span data-language-label={true} className="min-w-0 truncate">
            {shownLabel}
          </span>
          {pendingPreference !== null ? (
            <Spinner className="size-3.5 shrink-0" />
          ) : (
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className="size-4 shrink-0 text-muted-foreground"
            />
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={4}
        className="w-72 gap-0 rounded-lg p-1"
      >
        <Command
          value={highlighted}
          onValueChange={setHighlighted}
          shouldFilter={false}
          className="rounded-none bg-transparent p-0"
        >
          <CommandInput
            value={query}
            onValueChange={search}
            aria-label={t("settings.appearance.language.search")}
            placeholder={t("settings.appearance.language.search")}
          />
          <CommandList className="max-h-none overflow-visible">
            <CommandEmpty>
              {t("settings.appearance.language.noResults")}
            </CommandEmpty>
            {showAuto ? (
              <CommandItem
                value={AUTO_LOCALE}
                keywords={[autoLabel]}
                data-checked={checked === AUTO_LOCALE}
                onSelect={pick}
                className="cursor-pointer"
              >
                {autoLabel}
              </CommandItem>
            ) : null}
            <div
              onScroll={(event) =>
                setScrolled(event.currentTarget.scrollTop > 0)
              }
              className={cn(
                "hover-scrollbar max-h-64 overflow-y-auto overscroll-contain",
                scrolled &&
                  "[mask-image:linear-gradient(to_bottom,transparent_0,#000_14px)]",
              )}
            >
              {shownLocales.map(([value, metadata]) => (
                <CommandItem
                  key={value}
                  value={value}
                  keywords={[metadata.nativeLabel, metadata.label]}
                  data-checked={checked === value}
                  onSelect={pick}
                  className="cursor-pointer"
                >
                  <span className="min-w-0 truncate">
                    {metadata.nativeLabel}
                  </span>
                  {metadata.label !== metadata.nativeLabel ? (
                    <span className="min-w-0 truncate text-xs text-muted-foreground">
                      {metadata.label}
                    </span>
                  ) : null}
                </CommandItem>
              ))}
            </div>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
