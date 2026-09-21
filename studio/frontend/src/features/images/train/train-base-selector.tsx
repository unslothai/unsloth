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

// The Train tab's base picker, in the top bar where Create shows the generation model. Training
// never runs on a GGUF, so this lists only the trainable bases the panel's own selects offer,
// and picking one drives both.
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
  // Before /info answers there is nothing to pick from; the label still reads sensibly. The owner
  // prefix is dropped as the model selector does, family as the description.
  const label = base ? base.split("/").pop() || base : "Select base model";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label="Training base model"
          className="unsloth-model-selector-trigger flex h-[34px] min-w-0 max-w-[320px] items-center gap-2 rounded-full pl-4 pr-2 text-sm transition-colors hover:bg-accent"
        >
          <span className="min-w-0 truncate font-medium">{label}</span>
          {family && (
            <span className="shrink-0 text-xs leading-none text-muted-foreground">
              {family.label}
            </span>
          )}
          <span className="-ml-1 flex size-4 shrink-0 items-center justify-center">
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-3.5 text-muted-foreground"
            />
          </span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-[420px] w-[340px] overflow-y-auto">
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
