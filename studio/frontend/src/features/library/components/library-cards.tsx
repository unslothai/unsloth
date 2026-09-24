// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { Folder01Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useContext, useRef, useState } from "react";
import type { LibraryFolder, LibraryItem } from "../api";
import {
  KIND_ICONS,
  KIND_ICON_CLASS,
  fileKind,
  hasThumbnail,
  modelLabel,
} from "../file-kind";
import { formatCardTime, pluralize } from "../format";
import { useColumnCount, useLibraryThumbnail, useSeen } from "../hooks";
import { useLibraryActions } from "../actions-context";
import { CardSelectionContext } from "./card-selection";
import { LibraryActionsMenu } from "./library-actions";

const CARD_SURFACE = "bg-muted/70 dark:bg-card";

/** The item's type icon, tinted per kind. */
export function KindIcon({ item, className }: { item: LibraryItem; className?: string }) {
  const kind = fileKind(item);
  return (
    <HugeiconsIcon
      icon={KIND_ICONS[kind]}
      strokeWidth={1.5}
      className={cn(KIND_ICON_CLASS[kind], className)}
    />
  );
}

/** A lazily loaded image or video frame, sized by its own aspect ratio once it arrives. */
function ImageThumb({ item, className }: { item: LibraryItem; className?: string }) {
  const holder = useRef<HTMLDivElement>(null);
  const { url, failed } = useLibraryThumbnail(item, useSeen(holder));
  const [loaded, setLoaded] = useState(false);
  if (failed) {
    return (
      <div className={cn("flex aspect-square items-center justify-center", className)}>
        <KindIcon item={item} className="h-auto w-1/4 max-w-10" />
      </div>
    );
  }
  return (
    <div ref={holder} className={cn("relative overflow-hidden", className)}>
      {!loaded && <div className="aspect-square w-full animate-pulse bg-muted" />}
      {url && (
        <img
          src={url}
          alt={item.name}
          draggable={false}
          onLoad={() => setLoaded(true)}
          className={cn("block w-full", loaded ? "h-auto" : "absolute inset-0 opacity-0")}
        />
      )}
      {loaded && fileKind(item) === "video" && (
        <span className="pointer-events-none absolute inset-0 m-auto flex aspect-square w-1/4 max-w-10 items-center justify-center rounded-full bg-black/45 text-white backdrop-blur-sm">
          <HugeiconsIcon icon={PlayIcon} strokeWidth={2} className="size-1/2 [&_path]:fill-current" />
        </span>
      )}
    </div>
  );
}

function CardFrame({
  selectKey,
  onOpen,
  menu,
  children,
  className,
  label,
}: {
  selectKey: string;
  onOpen: () => void;
  menu: ReactNode;
  children: ReactNode;
  className?: string;
  label: string;
}) {
  const select = useContext(CardSelectionContext);
  const selected = select?.selection.has(selectKey) ?? false;
  // Once anything is selected, a click adds to the selection instead of opening.
  const selecting = (select?.selection.size ?? 0) > 0;
  return (
    <div className="group/library-card relative">
      <button
        type="button"
        aria-label={label}
        aria-pressed={select ? selected : undefined}
        onClick={select && selecting ? () => select.toggle(selectKey) : onOpen}
        className={cn(
          "block w-full overflow-hidden rounded-xl text-left outline-none ring-offset-2 ring-offset-background transition focus-visible:ring-2 focus-visible:ring-ring",
          selected && "ring-3 ring-foreground ring-offset-0",
          className,
        )}
      >
        {children}
      </button>
      {select && (
        <Checkbox
          checked={selected}
          onCheckedChange={() => select.toggle(selectKey)}
          aria-label={`Select ${label}`}
          className={cn(
            "absolute bottom-2.5 right-2.5 size-6 rounded-full border-border bg-background opacity-0 shadow-sm transition-opacity group-hover/library-card:opacity-100 focus-visible:opacity-100 data-checked:border-border data-checked:bg-background data-checked:text-foreground dark:bg-background dark:data-checked:bg-background [&_svg]:size-4",
            selected && "opacity-100",
          )}
        />
      )}
      <div className="absolute right-2 top-2">{menu}</div>
    </div>
  );
}

export function ItemCard({ item, showTime = true }: { item: LibraryItem; showTime?: boolean }) {
  const actions = useLibraryActions();
  const menu = <LibraryActionsMenu target={{ kind: "item", item }} variant="overlay" />;

  if (hasThumbnail(item)) {
    return (
      <CardFrame
        selectKey={`item:${item.id}`}
        label={item.name}
        onOpen={() => actions.openItem(item)}
        menu={menu}
        className="border border-border/60 bg-muted"
      >
        <ImageThumb item={item} />
      </CardFrame>
    );
  }
  return (
    <CardFrame
      selectKey={`item:${item.id}`}
      label={item.name}
      onOpen={() => actions.openItem(item)}
      menu={menu}
      className={cn(CARD_SURFACE, "hover:bg-muted dark:hover:bg-accent/60")}
    >
      <div className="flex aspect-square flex-col p-4">
        <p className="line-clamp-2 break-all pr-7 font-medium text-[14px] leading-snug text-foreground">
          {item.name}
        </p>
        <div className="flex flex-1 items-center justify-center">
          <KindIcon item={item} className="size-10" />
        </div>
        {showTime && (
          <p className="truncate pr-6 text-[13px] text-muted-foreground">
            {[modelLabel(item), formatCardTime(item.updatedAt)].filter(Boolean).join(" · ")}
          </p>
        )}
      </div>
    </CardFrame>
  );
}

export function FolderCard({
  folder,
  itemCount,
}: {
  folder: LibraryFolder;
  itemCount: number;
}) {
  const actions = useLibraryActions();
  return (
    <div>
      <CardFrame
        selectKey={`folder:${folder.id}`}
        label={folder.name}
        onOpen={() => actions.openFolder(folder.id)}
        menu={<LibraryActionsMenu target={{ kind: "folder", folder }} variant="overlay" />}
        className={cn(CARD_SURFACE, "hover:bg-muted dark:hover:bg-accent/60")}
      >
        <div className="flex aspect-square items-center justify-center">
          <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.5} className="size-10" />
        </div>
      </CardFrame>
      <button
        type="button"
        onClick={() => actions.openFolder(folder.id)}
        className="mt-2 block w-full px-1 text-left"
      >
        <p className="truncate font-medium text-[14px] text-foreground">{folder.name}</p>
        <p className="text-[13px] text-muted-foreground">{pluralize(itemCount, "item")}</p>
      </button>
    </div>
  );
}

/** Staggered columns: images keep their shape, so cards fill row by row into the shortest-looking
 *  column. Round-robin keeps newest-first reading order without measuring anything. */
export function Masonry<T>({
  items,
  getKey,
  render,
}: {
  items: T[];
  getKey: (item: T) => string;
  render: (item: T) => ReactNode;
}) {
  const container = useRef<HTMLDivElement>(null);
  const columns = useColumnCount(container);
  const buckets: T[][] = Array.from({ length: columns }, () => []);
  items.forEach((item, index) => buckets[index % columns]!.push(item));
  return (
    <div ref={container} className="flex items-start gap-4">
      {buckets.map((bucket, column) => (
        <div key={column} className="flex min-w-0 flex-1 flex-col gap-4">
          {bucket.map((item) => (
            <div key={getKey(item)}>{render(item)}</div>
          ))}
        </div>
      ))}
    </div>
  );
}

/** Uniform columns for folders, which are all the same shape. Same column count as the masonry
 *  below it, so the two sections line up. */
export function FolderGrid({
  folders,
  counts,
}: {
  folders: LibraryFolder[];
  counts: Map<string, number>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const columns = useColumnCount(container);
  return (
    <div
      ref={container}
      className="grid gap-4"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {folders.map((folder) => (
        <FolderCard key={folder.id} folder={folder} itemCount={counts.get(folder.id) ?? 0} />
      ))}
    </div>
  );
}

/** Small square used by list rows: the image itself, or the type icon on a tile. */
export function ItemTile({ item }: { item: LibraryItem }) {
  if (hasThumbnail(item)) {
    return (
      <div className="size-9 shrink-0 overflow-hidden rounded-xl border border-border/60 bg-muted [&_img]:size-9 [&_img]:object-cover [&_img]:object-top">
        <ImageThumb item={item} />
      </div>
    );
  }
  return (
    <div className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-border/60">
      <KindIcon item={item} className="size-5" />
    </div>
  );
}
