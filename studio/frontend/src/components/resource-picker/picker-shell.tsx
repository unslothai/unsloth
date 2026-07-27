// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  FolderSearchIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
  useId,
} from "react";
import { PICKER_TAB, type PickerTab, pickerTabId } from "./picker-tab-state";
import { PickerTabToggle } from "./picker-tab-toggle";

function OfflineHubState({
  noun,
  onSwitchDevice,
}: {
  noun: string;
  onSwitchDevice: () => void;
}) {
  const t = useT();
  return (
    <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
      <HugeiconsIcon
        icon={FolderSearchIcon}
        strokeWidth={1.5}
        className="size-5 text-muted-foreground/70"
      />
      <p className="text-ui-12p5 font-medium text-foreground">
        {t("picker.offlineTitle")}
      </p>
      <p className="text-ui-11 leading-snug text-muted-foreground">
        {t("picker.offlineBody", { noun })}
      </p>
      <button
        type="button"
        onClick={onSwitchDevice}
        className="hub-action-btn mt-1 h-7 px-3 text-ui-11p5"
      >
        {t("picker.offlineSwitchDevice")}
      </button>
    </div>
  );
}

function pickerOptions(container: HTMLElement): HTMLButtonElement[] {
  return Array.from(
    container.querySelectorAll<HTMLButtonElement>(
      '[data-picker-option="true"]:not(:disabled)',
    ),
  );
}

function optionMatchesQuery(option: HTMLButtonElement, query: string): boolean {
  const serialized = option.dataset.pickerValues;
  if (!serialized) {
    return false;
  }
  try {
    const values: unknown = JSON.parse(serialized);
    return Array.isArray(values) && values.includes(query);
  } catch {
    return false;
  }
}

function nextPickerNavigationTarget(
  container: HTMLElement,
  target: HTMLElement,
  key: "ArrowDown" | "ArrowUp",
): HTMLElement | undefined {
  const isSearch = target.matches('[data-picker-search="true"]');
  const isOption = target.matches('[data-picker-option="true"]');
  if (!(isSearch || isOption)) {
    return undefined;
  }
  const options = pickerOptions(container);
  if (options.length === 0) {
    return undefined;
  }
  if (isSearch) {
    return key === "ArrowDown" ? options[0] : options.at(-1);
  }
  const index = options.indexOf(target as HTMLButtonElement);
  if (key === "ArrowDown") {
    return options[Math.min(index + 1, options.length - 1)];
  }
  if (index <= 0) {
    return (
      container.querySelector<HTMLInputElement>(
        '[data-picker-search="true"]',
      ) ?? undefined
    );
  }
  return options[index - 1];
}

export function PickerShell({
  activeQuery,
  contentClassName,
  deviceContent,
  deviceQuery,
  hubContent,
  hubQuery,
  isHubLoading,
  noun,
  offlineNoun = noun,
  onOpenChange,
  onQueryChange,
  onTabChange,
  onUseThis,
  online,
  open,
  placeholder,
  scrollRef,
  showUseThis,
  tab,
  trigger,
  useThisLabel,
}: {
  activeQuery: string;
  contentClassName?: string;
  deviceContent: ReactNode;
  deviceQuery: string;
  hubContent: ReactNode;
  hubQuery: string;
  isHubLoading: boolean;
  noun: string;
  offlineNoun?: string;
  onOpenChange: (open: boolean) => void;
  onQueryChange: (value: string) => void;
  onTabChange: (tab: PickerTab) => void;
  onUseThis: () => void;
  online: boolean;
  open: boolean;
  placeholder: { hub: string; device: string };
  scrollRef: RefObject<HTMLDivElement | null>;
  showUseThis: boolean;
  tab: PickerTab;
  trigger: ReactNode;
  useThisLabel: string;
}) {
  const t = useT();
  const idBase = useId();
  const panelId = `${idBase}-panel`;
  const activeTabId = pickerTabId(idBase, tab);
  const tabs = [
    { value: PICKER_TAB.device, label: t("picker.onDevice") },
    { value: PICKER_TAB.hub, label: t("picker.huggingFace") },
  ] as const;

  function handlePickerNavigation(event: KeyboardEvent<HTMLDivElement>) {
    const key = event.key;
    if (key !== "ArrowDown" && key !== "ArrowUp") {
      return;
    }
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const next = nextPickerNavigationTarget(event.currentTarget, target, key);
    if (!next) {
      return;
    }
    event.preventDefault();
    next.focus();
    next.scrollIntoView({ block: "nearest" });
  }

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild={true}>{trigger}</PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        collisionPadding={16}
        onKeyDown={handlePickerNavigation}
        className={cn(
          "w-[min(420px,calc(100vw-2rem))] rounded-2xl p-4",
          contentClassName,
        )}
      >
        <PickerTabToggle
          tab={tab}
          options={tabs}
          onTabChange={onTabChange}
          idBase={idBase}
          panelId={panelId}
        />
        <div
          id={panelId}
          role="tabpanel"
          aria-labelledby={activeTabId}
          className="mt-2.5 flex flex-col gap-2"
        >
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={1.75}
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              data-picker-search="true"
              autoFocus={true}
              value={tab === PICKER_TAB.hub ? hubQuery : deviceQuery}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== "Enter") {
                  return;
                }
                e.preventDefault();
                if (showUseThis) {
                  onUseThis();
                  return;
                }
                const exactMatch = scrollRef.current
                  ? pickerOptions(scrollRef.current).find((option) =>
                      optionMatchesQuery(option, activeQuery),
                    )
                  : undefined;
                exactMatch?.click();
              }}
              placeholder={
                tab === PICKER_TAB.hub ? placeholder.hub : placeholder.device
              }
              aria-label={t("picker.searchAriaLabel", { noun })}
              className="field-soft h-9 rounded-full pl-9 text-ui-12p5"
            />
            {tab === PICKER_TAB.hub && isHubLoading && (
              <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            )}
          </div>

          <div
            ref={scrollRef}
            className="max-h-[320px] overflow-y-auto overscroll-contain rounded-[10px] [scrollbar-width:thin]"
          >
            {showUseThis && (
              <button
                type="button"
                data-picker-option="true"
                data-picker-values={JSON.stringify([activeQuery])}
                onClick={onUseThis}
                className="mb-1 flex w-full items-center gap-2 rounded-[8px] border border-dashed border-primary/30 bg-primary/[0.04] px-2.5 py-2 text-left text-ui-12p5 transition-colors hover:bg-primary/[0.08]"
              >
                <HugeiconsIcon
                  icon={
                    tab === PICKER_TAB.hub ? Search01Icon : FolderSearchIcon
                  }
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0 text-primary"
                />
                <span className="flex min-w-0 flex-1 flex-col leading-tight">
                  <span className="truncate font-medium text-foreground">
                    {activeQuery}
                  </span>
                  <span className="text-ui-10p5 text-muted-foreground/80">
                    {useThisLabel}
                  </span>
                </span>
                <HugeiconsIcon
                  icon={ArrowRight01Icon}
                  strokeWidth={1.5}
                  className="size-3.5 shrink-0 text-muted-foreground/70"
                />
              </button>
            )}
            {tab === PICKER_TAB.device ? (
              deviceContent
            ) : online ? (
              hubContent
            ) : (
              <OfflineHubState
                noun={offlineNoun}
                onSwitchDevice={() => onTabChange(PICKER_TAB.device)}
              />
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
