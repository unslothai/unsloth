// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type {
  LinkedInstance,
  LinkedInstanceInfo,
  LinkedInstanceStatus,
} from "@/features/settings/api/linked-instances";
import { gpuPool } from "@/features/settings/components/linked-instance-format";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { CloudServerIcon, ComputerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";

function gpuLine(info?: LinkedInstanceInfo): string | null {
  if (!info?.gpus?.length) return info?.online ? "CPU only" : null;
  const pool = gpuPool(info.gpus);
  return pool.total
    ? `${pool.label} · ${Math.round(pool.total)} GB`
    : pool.label;
}

function Row({
  icon,
  title,
  detail,
  online,
  selected,
  onClick,
}: {
  icon: typeof ComputerIcon;
  title: string;
  detail: string | null;
  online?: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      onClick={onClick}
      className="relative flex w-full min-w-0 items-center gap-2.5 rounded-[10px] py-1.5 pr-8 pl-2 text-left outline-none transition-colors hover:bg-accent focus-visible:bg-accent"
    >
      <span className="relative flex size-7 shrink-0 items-center justify-center rounded-full bg-muted">
        <HugeiconsIcon icon={icon} className="size-3.5 text-foreground" />
        {online !== undefined ? (
          <span
            aria-hidden="true"
            className={cn(
              "absolute -right-0.5 -bottom-0.5 size-2.5 rounded-full ring-2 ring-popover",
              online ? "bg-emerald-500" : "bg-muted-foreground/50",
            )}
          />
        ) : null}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="break-all text-xs font-medium text-foreground">
          {title}
        </span>
        {detail ? (
          <span className="text-ui-11 text-muted-foreground">{detail}</span>
        ) : null}
      </span>
      {selected ? (
        <HugeiconsIcon
          icon={Tick02Icon}
          strokeWidth={2}
          className="pointer-events-none absolute right-2 size-4"
        />
      ) : null}
    </button>
  );
}

/** The half pill beside Recommended / On Device: which machine the picker lists and loads on. */
export function MachineSwitch({
  instances,
  statuses,
  infos,
  value,
  onValueChange,
  className,
}: {
  instances: LinkedInstance[];
  statuses: Record<string, LinkedInstanceStatus>;
  infos: Record<string, LinkedInstanceInfo>;
  value: LinkedInstance | null;
  onValueChange: (id: string | null) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const pick = (id: string | null) => {
    onValueChange(id);
    setOpen(false);
  };
  const online = value ? statuses[value.id]?.online : undefined;
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label="Choose which machine to list models for"
          aria-haspopup="listbox"
          aria-expanded={open}
          title={value ? `Models on @${value.name}` : "Models on this machine"}
          className={cn(
            "hub-menu-trigger hub-tab-toggle relative inline-flex h-(--picker-control-h) shrink-0 items-center gap-1.5 rounded-full pr-2.5 pl-2 text-ui-12p5 text-muted-foreground transition-colors hover:text-foreground",
            value && "text-foreground",
            className,
          )}
        >
          <span className="relative flex">
            <HugeiconsIcon
              icon={value ? CloudServerIcon : ComputerIcon}
              className="size-3.5 shrink-0"
            />
            {online !== undefined ? (
              <span
                aria-hidden="true"
                className={cn(
                  "absolute -right-1 -bottom-0.5 size-1.5 rounded-full",
                  online ? "bg-emerald-500" : "bg-muted-foreground/60",
                )}
              />
            ) : null}
          </span>
          {value ? (
            <span className="max-w-[9rem] truncate">@{value.name}</span>
          ) : null}
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className="size-3 shrink-0"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        side="bottom"
        sideOffset={8}
        collisionPadding={12}
        className="hub-menu-instant menu-soft-surface w-[min(18rem,calc(100vw-1rem))] rounded-[14px] p-1 ring-0"
      >
        <div
          role="listbox"
          aria-label="Machine"
          className="flex flex-col gap-0.5"
        >
          <div className="px-2 pt-1.5 pb-1 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
            Run models on
          </div>
          <Row
            icon={ComputerIcon}
            title="This machine"
            detail="Local GPU and downloads"
            selected={value === null}
            onClick={() => pick(null)}
          />
          {instances.map((instance) => (
            <Row
              key={instance.id}
              icon={CloudServerIcon}
              title={`@${instance.name}`}
              detail={
                gpuLine(infos[instance.id]) ??
                (statuses[instance.id]?.online === false ? "Offline" : null)
              }
              online={statuses[instance.id]?.online}
              selected={value?.id === instance.id}
              onClick={() => pick(instance.id)}
            />
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
