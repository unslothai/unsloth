// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export type TrainFamilyOption = {
  name: string;
  label: string;
  base_repos: string[];
};

// Training never runs on a GGUF, so only trainable bases are listed.
export function TrainBaseSelector({
  families,
  familyName,
  base,
  onSelect,
}: {
  families: TrainFamilyOption[];
  familyName: string;
  base: string;
  onSelect: (family: string, repo: string) => void;
}) {
  const family = families.find((f) => f.name === familyName);
  const label = base ? base.split("/").pop() || base : "Select base model";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        {/* Matches Create's picker, so the name stays put across tabs. */}
        <button
          type="button"
          aria-label="Training base model"
          className="unsloth-model-selector-trigger flex h-[calc(34px*var(--ui-space-scale,1))] min-w-0 max-w-full items-center gap-1 overflow-hidden rounded-[10px] pl-3 pr-1 transition-colors hover:bg-nav-surface-hover @[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"
        >
          <span className="flex min-w-0 flex-1 items-baseline">
            <span className="min-w-0 shrink-0 truncate whitespace-nowrap font-heading text-ui-14 font-medium leading-tight text-black dark:text-foreground @[68rem]:text-ui-16">
              {label}
            </span>
            {family && (
              <span className="min-w-0 shrink-[1000] truncate text-xs leading-tight text-muted-foreground before:inline-block before:w-2 before:content-['']">
                {family.label}
              </span>
            )}
          </span>
          <span className="-ml-1 flex size-4 shrink-0 items-center justify-center">
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-3.5 text-muted-foreground"
            />
          </span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-[min(calc(420px*var(--ui-space-scale,1)),var(--radix-dropdown-menu-content-available-height))] w-[calc(340px*var(--ui-space-scale,1))] overflow-y-auto">
        {families.map((f) => (
          <DropdownMenuGroup key={f.name}>
            <DropdownMenuLabel className="text-ui-11 text-muted-foreground">
              {f.label}
            </DropdownMenuLabel>
            {f.base_repos.map((repo) => {
              const selected = f.name === familyName && repo === base;
              return (
                <DropdownMenuItem
                  key={repo}
                  onSelect={() => onSelect(f.name, repo)}
                  className="gap-2"
                >
                  <span className="min-w-0 flex-1 truncate">{repo}</span>
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    className={cn("size-4 shrink-0", !selected && "opacity-0")}
                  />
                </DropdownMenuItem>
              );
            })}
          </DropdownMenuGroup>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
