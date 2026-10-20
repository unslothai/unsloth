// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import {
  AUDIO_WORKFLOWS,
  type AudioWorkflowId,
  isAudioWorkflowId,
} from "../workflows";

export function WorkflowTitleMenu({
  workflow,
  onSelect,
}: {
  workflow: AudioWorkflowId;
  onSelect: (workflow: AudioWorkflowId) => void;
}) {
  const current =
    AUDIO_WORKFLOWS.find((tab) => tab.id === workflow) ?? AUDIO_WORKFLOWS[0];
  return (
    <h2 className="font-heading text-xl font-medium leading-none text-foreground">
      <DropdownMenu>
        <DropdownMenuTrigger
          data-tour="audio-mode"
          aria-label={`${current.heading}, switch Audio page`}
          className="-mx-2 -my-1 flex max-w-full items-center gap-2 rounded-full px-2 py-1 outline-none transition-colors duration-150 hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:bg-accent"
        >
          <HugeiconsIcon
            icon={current.icon}
            className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0"
          />
          <span className="truncate">{current.heading}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.5}
            className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
          />
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          sideOffset={6}
          className="w-[min(calc(320px*var(--ui-space-scale,1)),calc(100vw-32px))]"
        >
          <DropdownMenuRadioGroup
            value={current.id}
            onValueChange={(value) => {
              if (isAudioWorkflowId(value)) onSelect(value);
            }}
          >
            {AUDIO_WORKFLOWS.map((tab) => (
              <DropdownMenuRadioItem
                key={tab.id}
                value={tab.id}
                textValue={tab.label}
                className="items-start"
              >
                <HugeiconsIcon
                  icon={tab.icon}
                  className="mt-0.5 size-[calc(16px*var(--ui-space-scale,1))]"
                />
                <span className="grid min-w-0 gap-0.5">
                  <span className="text-ui-13 font-medium text-foreground">
                    {tab.label}
                  </span>
                  <span className="text-ui-11p5 leading-snug text-muted-foreground">
                    {tab.hint}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </h2>
  );
}
