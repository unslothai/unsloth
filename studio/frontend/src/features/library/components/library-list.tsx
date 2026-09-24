// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import type { LibraryFolder, LibraryItem } from "../api";
import { modelLabel } from "../file-kind";
import { formatRelativeTime, pluralize } from "../format";
import { type LibraryTarget, useLibraryActions } from "../actions-context";
import { LibraryActionsMenu } from "./library-actions";
import { ItemTile } from "./library-cards";

function targetKey(target: LibraryTarget): string {
  return target.kind === "item" ? `item:${target.item.id}` : `folder:${target.folder.id}`;
}

// Row content lines up with the tab labels (px-4); the checkbox hangs in the margin to its left.
const ROW_INSET = "pl-4 pr-3";

/** The checkbox in the page margin: shown on hover, or on every row once anything is selected. */
function GutterCheckbox({
  checked,
  visible,
  group,
  onCheckedChange,
  label,
}: {
  checked: boolean | "indeterminate";
  visible: boolean;
  group: "row" | "head";
  onCheckedChange: () => void;
  label: string;
}) {
  return (
    <div className="absolute right-full top-1/2 mr-3 flex -translate-y-1/2">
      <Checkbox
        checked={checked}
        onCheckedChange={onCheckedChange}
        aria-label={label}
        className={cn(
          "rounded-full opacity-0 transition-opacity focus-visible:opacity-100",
          group === "row"
            ? "group-hover/library-row:opacity-100"
            : "group-hover/library-head:opacity-100",
          visible && "opacity-100",
        )}
      />
    </div>
  );
}

function Row({
  target,
  selected,
  selecting,
  onSelectedChange,
  onOpen,
  tile,
  name,
  activity,
}: {
  target: LibraryTarget;
  selected: boolean;
  selecting: boolean;
  onSelectedChange: (selected: boolean) => void;
  onOpen: () => void;
  tile: ReactNode;
  name: ReactNode;
  activity: string;
}) {
  return (
    <div
      className={cn(
        "group/library-row relative flex items-center gap-4 rounded-xl transition-colors hover:bg-muted/60",
        ROW_INSET,
        selected && "bg-muted/60",
      )}
    >
      <GutterCheckbox
        checked={selected}
        visible={selecting}
        group="row"
        onCheckedChange={() => onSelectedChange(!selected)}
        label="Select"
      />
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 items-center gap-4 py-2 text-left outline-none"
      >
        {tile}
        <span className="flex min-w-0 items-center gap-2 text-[14px] text-foreground">{name}</span>
        <span className="ml-auto hidden w-44 shrink-0 text-[13px] text-muted-foreground sm:block">
          {activity}
        </span>
      </button>
      <LibraryActionsMenu target={target} variant="row" />
    </div>
  );
}

export function LibraryList({
  folders,
  items,
  counts,
  selection,
  onSelectionChange,
}: {
  folders: LibraryFolder[];
  items: LibraryItem[];
  counts: Map<string, number>;
  selection: Set<string>;
  onSelectionChange: (next: Set<string>) => void;
}) {
  const actions = useLibraryActions();
  const targets: LibraryTarget[] = [
    ...folders.map((folder) => ({ kind: "folder" as const, folder })),
    ...items.map((item) => ({ kind: "item" as const, item })),
  ];
  const allSelected = targets.length > 0 && targets.every((t) => selection.has(targetKey(t)));
  const selecting = selection.size > 0;

  const setSelected = (target: LibraryTarget, selected: boolean) => {
    const next = new Set(selection);
    if (selected) next.add(targetKey(target));
    else next.delete(targetKey(target));
    onSelectionChange(next);
  };

  return (
    <div>
      <div
        className={cn(
          "group/library-head relative flex items-center gap-4 border-b border-border/60 pb-2 text-[13px] text-muted-foreground",
          ROW_INSET,
        )}
      >
        <GutterCheckbox
          checked={allSelected ? true : selecting ? "indeterminate" : false}
          visible={selecting}
          group="head"
          onCheckedChange={() =>
            onSelectionChange(allSelected ? new Set() : new Set(targets.map(targetKey)))
          }
          label="Select all"
        />
        <span className="flex-1">Name</span>
        <span className="hidden w-44 shrink-0 sm:block">Last activity</span>
        <span className="w-8 shrink-0" />
      </div>
      <div className="mt-1 flex flex-col">
        {folders.map((folder) => (
          <Row
            key={folder.id}
            target={{ kind: "folder", folder }}
            selected={selection.has(`folder:${folder.id}`)}
            selecting={selecting}
            onSelectedChange={(selected) => setSelected({ kind: "folder", folder }, selected)}
            onOpen={() => actions.openFolder(folder.id)}
            tile={
              <div className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-border/60">
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.5} className="size-5" />
              </div>
            }
            name={
              <>
                <span className="truncate">{folder.name}</span>
                <span className="shrink-0 text-muted-foreground text-sm">
                  {pluralize(counts.get(folder.id) ?? 0, "item")}
                </span>
              </>
            }
            activity={`Modified ${formatRelativeTime(folder.updatedAt)}`}
          />
        ))}
        {items.map((item) => (
          <Row
            key={item.id}
            target={{ kind: "item", item }}
            selected={selection.has(`item:${item.id}`)}
            selecting={selecting}
            onSelectedChange={(selected) => setSelected({ kind: "item", item }, selected)}
            onOpen={() => actions.openItem(item)}
            tile={<ItemTile item={item} />}
            name={
              <>
                <span className="truncate">{item.name}</span>
                {item.model && (
                  <span className="shrink-0 text-muted-foreground text-sm">{modelLabel(item)}</span>
                )}
                {item.favorite && (
                  <HugeiconsIcon
                    icon={StarPointedIcon}
                    aria-label="Favorite"
                    strokeWidth={1.75}
                    className="size-3.5 shrink-0 text-muted-foreground [&_path]:fill-current"
                  />
                )}
              </>
            }
            activity={`Modified ${formatRelativeTime(item.updatedAt)}`}
          />
        ))}
      </div>
    </div>
  );
}
