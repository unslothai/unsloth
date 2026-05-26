// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  FolderSearchIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, type RefObject, useId } from "react";
import {
  PICKER_TAB,
  PICKER_TABS,
  type PickerTab,
  PickerTabToggle,
  pickerTabId,
} from "./picker-tab-toggle";

function OfflineHubState({
  noun,
  onSwitchDevice,
}: {
  noun: string;
  onSwitchDevice: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
      <HugeiconsIcon
        icon={FolderSearchIcon}
        strokeWidth={1.5}
        className="size-5 text-muted-foreground/70"
      />
      <p className="text-[12.5px] font-medium text-foreground">
        You're offline
      </p>
      <p className="text-[11px] leading-snug text-muted-foreground">
        Switch to Device to use cached or local {noun}.
      </p>
      <button
        type="button"
        onClick={onSwitchDevice}
        className="hub-action-btn mt-1 h-7 px-3 text-[11.5px]"
      >
        Device
      </button>
    </div>
  );
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
  const idBase = useId();
  const panelId = `${idBase}-panel`;
  const activeTabId = pickerTabId(idBase, tab);
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild={true}>{trigger}</PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        collisionPadding={16}
        className={cn(
          "w-[min(420px,calc(100vw-2rem))] rounded-2xl p-4",
          contentClassName,
        )}
      >
        <PickerTabToggle
          tab={tab}
          options={PICKER_TABS}
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
              autoFocus={true}
              value={tab === PICKER_TAB.HUB ? hubQuery : deviceQuery}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                e.preventDefault();
                if (showUseThis) onUseThis();
              }}
              placeholder={
                tab === PICKER_TAB.HUB ? placeholder.hub : placeholder.device
              }
              aria-label={`Search ${noun}`}
              className="field-soft h-9 rounded-full pl-9 text-[12.5px]"
            />
            {tab === PICKER_TAB.HUB && isHubLoading && (
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
                onClick={onUseThis}
                className="mb-1 flex w-full items-center gap-2 rounded-[8px] border border-dashed border-primary/30 bg-primary/[0.04] px-2.5 py-2 text-left text-[12.5px] transition-colors hover:bg-primary/[0.08]"
              >
                <HugeiconsIcon
                  icon={tab === PICKER_TAB.HUB ? Search01Icon : FolderSearchIcon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0 text-primary"
                />
                <span className="flex min-w-0 flex-1 flex-col leading-tight">
                  <span className="truncate font-medium text-foreground">
                    {activeQuery}
                  </span>
                  <span className="text-[10.5px] text-muted-foreground/80">
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
            {tab === PICKER_TAB.DEVICE ? (
              deviceContent
            ) : !online ? (
              <OfflineHubState
                noun={offlineNoun}
                onSwitchDevice={() => onTabChange(PICKER_TAB.DEVICE)}
              />
            ) : (
              hubContent
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
