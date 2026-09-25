// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import type { LibraryFolder, LibraryItem } from "../api";
import { modelLabelKey } from "../file-kind";
import { formatActivityTime, formatCardTime, formatItemCount, formatSize } from "../format";
import type { LibrarySortKey, LibrarySortState } from "../settings-store";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { type LibraryTarget, useLibraryActions } from "../actions-context";
import { LibraryActionsMenu } from "./library-actions";
import { ItemTile } from "./library-cards";

function targetKey(target: LibraryTarget): string {
  return target.kind === "item" ? `item:${target.item.id}` : `folder:${target.folder.id}`;
}

// Row content lines up with the tab labels (px-4); the checkbox hangs in the margin to its left.
const ROW_INSET = "pl-4 pr-6";

const CELL = "hidden text-[13px] text-muted-foreground sm:block";
const ACTIVITY_COLUMN = "w-48 shrink-0";
const MODIFIED_COLUMN = "w-40 shrink-0";
const SIZE_COLUMN = "w-28 shrink-0";
// Suggested shows one relative Last activity column instead of Modified and Size.
const ACTIVITY_COLUMNS: [LibrarySortKey, TranslationKey, string][] = [
  ["modified", "library.list.lastActivity", ACTIVITY_COLUMN],
];
const COLUMNS: [LibrarySortKey, TranslationKey, string][] = [
  ["modified", "library.list.modifiedColumn", MODIFIED_COLUMN],
  ["size", "library.preview.size", SIZE_COLUMN],
];

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
  checked: boolean;
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
          // A shade darker than the input border in light mode, which vanished on white.
          "rounded-full border-neutral-300 opacity-0 transition-opacity focus-visible:opacity-100 dark:border-input",
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
  opened,
  size,
  activity,
}: {
  target: LibraryTarget;
  selected: boolean;
  selecting: boolean;
  onSelectedChange: (selected: boolean) => void;
  onOpen: () => void;
  tile: ReactNode;
  name: ReactNode;
  modified: number;
  opened?: number | null;
  size?: number | null;
  activity: boolean;
}) {
  const t = useT();
  const locale = useLocale();
  return (
    <div
      className={cn(
        "group/library-row relative flex items-center gap-4 rounded-[14px] transition-colors hover:bg-muted dark:hover:bg-muted/60",
        ROW_INSET,
        selected && "bg-muted dark:bg-muted/60",
      )}
    >
      <GutterCheckbox
        checked={selected}
        visible={selecting}
        group="row"
        onCheckedChange={() => onSelectedChange(!selected)}
        label={t("library.selectItem", {
          name: target.kind === "item" ? target.item.name : target.folder.name,
        })}
      />
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 items-center gap-4 rounded-lg py-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {tile}
        <span className="flex min-w-0 items-center gap-2 text-[14px] text-foreground">{name}</span>
        {activity ? (
          <span className={cn(ACTIVITY_COLUMN, CELL, "ml-auto")}>
            {(opened ?? 0) > modified
              ? t("library.list.opened", { time: formatActivityTime(opened!, locale, t) })
              : t("library.list.modified", { time: formatActivityTime(modified, locale, t) })}
          </span>
        ) : (
          <>
            <span className={cn(MODIFIED_COLUMN, CELL, "ml-auto")}>{formatCardTime(modified, locale)}</span>
            <span className={cn(SIZE_COLUMN, CELL)}>{formatSize(size ?? null, locale, t)}</span>
          </>
        )}
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
  activity,
}: {
  folders: LibraryFolder[];
  items: LibraryItem[];
  counts: Map<string, number>;
  selection: Set<string>;
  onSelectionChange: (next: Set<string>) => void;
  sort: LibrarySortState;
  onSortChange: (key: LibrarySortKey) => void;
  activity: boolean;
}) {
  const t = useT();
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
      {/* The padding sits outside the row, so the checkbox centers on the column titles. */}
      <div className="pb-2">
        <div
          className={cn(
            "group/library-head relative flex items-center gap-4 text-[13px] text-muted-foreground",
            ROW_INSET,
          )}
        >
          {/* Ticked only once everything is: a partial tick read as every row being selected. */}
          <GutterCheckbox
            checked={allSelected}
            visible={selecting}
            group="head"
            onCheckedChange={() =>
              onSelectionChange(allSelected ? new Set() : new Set(targets.map(targetKey)))
            }
            label={t("library.list.selectAll")}
          />
          <span className="flex-1">
            <SortHeader column="name" label={t("library.list.name")} sort={sort} onSortChange={onSortChange} />
          </span>
          {(activity ? ACTIVITY_COLUMNS : COLUMNS).map(([column, label, width]) => (
            <SortHeader
              key={column}
              column={column}
              label={t(label)}
              sort={sort}
              onSortChange={onSortChange}
              className={cn(width, "hidden sm:flex")}
            />
          ))}
          <span className="w-8 shrink-0" />
        </div>
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
              <div className="flex size-9 shrink-0 items-center justify-center rounded-[10px] border border-border/60">
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.5} className="size-5" />
              </div>
            }
            name={
              <>
                <span className="truncate">{folder.name}</span>
                <span className="shrink-0 text-muted-foreground text-sm">
                  {formatItemCount(counts.get(folder.id) ?? 0, t)}
                </span>
              </>
            }
            modified={folder.updatedAt}
            activity={activity}
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
                  <span className="shrink-0 text-muted-foreground text-sm">
                    {t(modelLabelKey(item)!)}
                  </span>
                )}
                {item.favorite && (
                  <HugeiconsIcon
                    icon={StarPointedIcon}
                    aria-label={t("library.list.favorite")}
                    strokeWidth={1.75}
                    className="size-3.5 shrink-0 text-muted-foreground [&_path]:fill-current"
                  />
                )}
              </>
            }
            modified={item.updatedAt}
            opened={item.openedAt}
            size={item.sizeBytes}
            activity={activity}
          />
        ))}
      </div>
    </div>
  );
}
