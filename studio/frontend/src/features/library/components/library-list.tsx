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
import { formatCardTime, formatSize, pluralize } from "../format";
import type { LibrarySortKey, LibrarySortState } from "../settings-store";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { type LibraryTarget, useLibraryActions } from "../actions-context";
import { LibraryActionsMenu } from "./library-actions";
import { ItemTile } from "./library-cards";

function targetKey(target: LibraryTarget): string {
  return target.kind === "item" ? `item:${target.item.id}` : `folder:${target.folder.id}`;
}

// Row content lines up with the tab labels (px-4); the checkbox hangs in the margin to its left.
const ROW_INSET = "pl-4 pr-3";

const MODIFIED_COLUMN = "w-36 shrink-0";
const SIZE_COLUMN = "w-24 shrink-0";

function SortHeader({
  column,
  label,
  sort,
  onSortChange,
  className,
}: {
  column: LibrarySortKey;
  label: string;
  sort: LibrarySortState;
  onSortChange: (key: LibrarySortKey) => void;
  className?: string;
}) {
  const active = sort.key === column;
  const Arrow = sort.desc ? ArrowDownIcon : ArrowUpIcon;
  return (
    <button
      type="button"
      onClick={() => onSortChange(column)}
      aria-sort={active ? (sort.desc ? "descending" : "ascending") : undefined}
      className={cn(
        "flex items-center gap-1 text-left transition-colors hover:text-foreground",
        active && "text-foreground",
        className,
      )}
    >
      {label}
      {active && <Arrow className="size-3.5" strokeWidth={2} />}
    </button>
  );
}

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
          "opacity-0 transition-opacity focus-visible:opacity-100",
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
  modified,
  size,
}: {
  target: LibraryTarget;
  selected: boolean;
  selecting: boolean;
  onSelectedChange: (selected: boolean) => void;
  onOpen: () => void;
  tile: ReactNode;
  name: ReactNode;
  modified: number;
  size: number | null;
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
        <span className={cn(MODIFIED_COLUMN, "ml-auto hidden text-[13px] text-muted-foreground sm:block")}>
          {formatCardTime(modified)}
        </span>
        <span className={cn(SIZE_COLUMN, "hidden text-[13px] text-muted-foreground sm:block")}>
          {formatSize(size)}
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
  sort,
  onSortChange,
}: {
  folders: LibraryFolder[];
  items: LibraryItem[];
  counts: Map<string, number>;
  selection: Set<string>;
  onSelectionChange: (next: Set<string>) => void;
  sort: LibrarySortState;
  onSortChange: (key: LibrarySortKey) => void;
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
        <span className="flex-1">
          <SortHeader column="name" label="Name" sort={sort} onSortChange={onSortChange} />
        </span>
        <SortHeader
          column="modified"
          label="Modified"
          sort={sort}
          onSortChange={onSortChange}
          className={cn(MODIFIED_COLUMN, "hidden sm:flex")}
        />
        <SortHeader
          column="size"
          label="Size"
          sort={sort}
          onSortChange={onSortChange}
          className={cn(SIZE_COLUMN, "hidden sm:flex")}
        />
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
            modified={folder.updatedAt}
            size={null}
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
            modified={item.updatedAt}
            size={item.sizeBytes}
          />
        ))}
      </div>
    </div>
  );
}
