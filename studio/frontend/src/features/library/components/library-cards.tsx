// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Folder01Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, type RefObject, useContext, useRef, useState } from "react";
import type { LibraryFolder, LibraryItem } from "../api";
import {
  KIND_ICONS,
  KIND_ICON_CLASS,
  fileKind,
  hasThumbnail,
} from "../file-kind";
import { formatCardTime, formatItemCount } from "../format";
import { useColumnCount, useLibraryThumbnail, useSeen } from "../hooks";
import { useLibraryActions } from "../actions-context";
import { CARD_COLUMNS, useLibrarySettingsStore } from "../settings-store";
import { GLASS_CONTROL, GLASS_SURFACE, OVERLAY_CONTROL, RAISED_SURFACE } from "../surface";
import { CardSelectionContext } from "./card-selection";
import { LibraryActionsMenu } from "./library-actions";

const CARD_SHADOW =
  "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.06)] dark:shadow-none";

const CARD_SURFACE = cn(
  RAISED_SURFACE,
  CARD_SHADOW,
  "group-hover/library-card:bg-neutral-100 group-hover/library-card:shadow-none dark:group-hover/library-card:bg-accent/60",
);

export function KindIcon({ item, className }: { item: LibraryItem; className?: string }) {
  const kind = fileKind(item);
  return (
    <HugeiconsIcon
      icon={KIND_ICONS[kind]}
      strokeWidth={1.5}
      className={cn(KIND_ICON_CLASS[kind], className, kind === "model" && "scale-95")}
    />
  );
}

const MIN_THUMB_RATIO = 2 / 3;
const MAX_THUMB_RATIO = 3 / 2;

function ImageThumb({
  item,
  className,
  square = false,
}: {
  item: LibraryItem;
  className?: string;
  square?: boolean;
}) {
  const holder = useRef<HTMLDivElement>(null);
  const { url, failed } = useLibraryThumbnail(item, useSeen(holder));
  const [ratio, setRatio] = useState<number | null>(null);
  const loaded = ratio !== null;
  const [brokenUrl, setBrokenUrl] = useState<string | null>(null);
  if (failed || (url !== null && url === brokenUrl)) {
    return (
      <div className={cn("flex aspect-square items-center justify-center", className)}>
        <KindIcon item={item} className="h-auto w-1/4 max-w-9" />
      </div>
    );
  }
  return (
    <div
      ref={holder}
      className={cn("relative overflow-hidden", square && "aspect-square", className)}
      style={
        loaded && !square
          ? { aspectRatio: 1 / Math.min(Math.max(ratio, MIN_THUMB_RATIO), MAX_THUMB_RATIO) }
          : undefined
      }
    >
      {!loaded && <div className="aspect-square w-full animate-pulse bg-muted" />}
      {url && (
        <img
          src={url}
          alt={item.name}
          draggable={false}
          onLoad={(event) => {
            const { naturalWidth, naturalHeight } = event.currentTarget;
            setRatio(naturalWidth > 0 ? naturalHeight / naturalWidth : 1);
          }}
          onError={() => setBrokenUrl(url)}
          className={cn(
            "absolute inset-0 block size-full object-cover",
            !loaded && "opacity-0",
            loaded && ratio > MAX_THUMB_RATIO && "object-top",
          )}
        />
      )}
      {loaded && fileKind(item) === "video" && (
        <span className="pointer-events-none absolute inset-0 m-auto flex aspect-square w-1/4 max-w-10 items-center justify-center rounded-full bg-white/30 text-white backdrop-blur-md dark:bg-black/40">
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
  glass = false,
}: {
  selectKey: string;
  onOpen: () => void;
  menu: ReactNode;
  children: ReactNode;
  className?: string;
  label: string;
  glass?: boolean;
}) {
  const t = useT();
  const select = useContext(CardSelectionContext);
  const selected = select?.selection.has(selectKey) ?? false;
  const selecting = (select?.selection.size ?? 0) > 0;
  return (
    <div className="group/library-card relative">
      <button
        type="button"
        aria-label={label}
        aria-pressed={select && selecting ? selected : undefined}
        onClick={select && selecting ? () => select.toggle(selectKey) : onOpen}
        className={cn(
          "block w-full overflow-hidden rounded-xl text-left outline-none transition focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          selected && "ring-3 ring-foreground",
          className,
        )}
      >
        {children}
      </button>
      {select && (
        <Checkbox
          checked={selected}
          onCheckedChange={() => select.toggle(selectKey)}
          aria-label={t("library.selectItem", { name: label })}
          className={cn(
            OVERLAY_CONTROL,
            "absolute bottom-4 right-4 size-5 rounded-full border-0 opacity-0 transition-opacity group-hover/library-card:opacity-100 focus-visible:opacity-100 data-checked:bg-white data-checked:text-foreground dark:data-checked:bg-neutral-200 dark:data-checked:text-neutral-900 [&_svg]:size-3.5",
            glass && GLASS_CONTROL,
            selected && "opacity-100",
          )}
        />
      )}
      <div className="absolute right-2 top-2">{menu}</div>
    </div>
  );
}

export function ItemCard({ item }: { item: LibraryItem }) {
  const locale = useLocale();
  const actions = useLibraryActions();
  const showTime = useLibrarySettingsStore((s) => s.showCardDates);
  const square = useLibrarySettingsStore((s) => s.imageLayout === "square");
  const thumb = hasThumbnail(item);
  return (
    <CardFrame
      selectKey={`item:${item.id}`}
      label={item.name}
      onOpen={() => actions.openItem(item)}
      menu={
        <LibraryActionsMenu
          target={{ kind: "item", item }}
          variant="overlay"
          className={thumb ? GLASS_CONTROL : undefined}
        />
      }
      glass={thumb}
      className={thumb ? cn("relative bg-muted", CARD_SHADOW) : CARD_SURFACE}
    >
      {thumb ? (
        <>
          <ImageThumb item={item} square={square && fileKind(item) === "image"} />
          {showTime && (
            <span
              className={cn(
                GLASS_SURFACE,
                "pointer-events-none absolute bottom-2 left-2 rounded-full px-2 py-0.5 text-ui-12 opacity-0 transition-opacity group-hover/library-card:opacity-100 group-has-[:focus-visible]/library-card:opacity-100 pointer-coarse:opacity-100",
              )}
            >
              {formatCardTime(item.updatedAt, locale)}
            </span>
          )}
        </>
      ) : (
        <div className="flex aspect-square flex-col px-5 pb-3.5 pt-5">
          <p className="line-clamp-2 break-all pr-7 font-medium text-ui-14 leading-snug text-foreground">
            {item.name}
          </p>
          <div className="flex flex-1 items-center justify-center">
            <KindIcon item={item} className="size-9" />
          </div>
          <p className="truncate pr-6 text-ui-13 text-muted-foreground">
            {showTime && formatCardTime(item.updatedAt, locale)}
          </p>
        </div>
      )}
    </CardFrame>
  );
}

function FolderCard({
  folder,
  itemCount,
}: {
  folder: LibraryFolder;
  itemCount: number;
}) {
  const t = useT();
  const actions = useLibraryActions();
  return (
    <div>
      <CardFrame
        selectKey={`folder:${folder.id}`}
        label={folder.name}
        onOpen={() => actions.openFolder(folder.id)}
        menu={<LibraryActionsMenu target={{ kind: "folder", folder }} variant="overlay" />}
        className={CARD_SURFACE}
      >
        <div className="flex aspect-square items-center justify-center">
          <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.5} className="size-9" />
        </div>
      </CardFrame>
      <button
        type="button"
        onClick={() => actions.openFolder(folder.id)}
        className="mt-2 block w-full px-1 text-left"
      >
        <p className="truncate font-medium text-ui-14 text-foreground">{folder.name}</p>
        <p className="text-ui-13 text-muted-foreground">{formatItemCount(itemCount, t)}</p>
      </button>
    </div>
  );
}

function useCardColumns(container: RefObject<HTMLDivElement | null>): number {
  const { minWidth, max } = CARD_COLUMNS[useLibrarySettingsStore((s) => s.cardSize)];
  return useColumnCount(container, minWidth, max);
}

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
  const columns = useCardColumns(container);
  const buckets: T[][] = Array.from({ length: columns }, () => []);
  items.forEach((item, index) => buckets[index % columns]!.push(item));
  return (
    <div ref={container} className="flex items-start gap-5">
      {buckets.map((bucket, column) => (
        <div key={column} className="flex min-w-0 flex-1 flex-col gap-5">
          {bucket.map((item) => (
            <div key={getKey(item)}>{render(item)}</div>
          ))}
        </div>
      ))}
    </div>
  );
}

export function FolderGrid({
  folders,
  counts,
}: {
  folders: LibraryFolder[];
  counts: Map<string, number>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const columns = useCardColumns(container);
  return (
    <div
      ref={container}
      className="grid gap-5"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {folders.map((folder) => (
        <FolderCard key={folder.id} folder={folder} itemCount={counts.get(folder.id) ?? 0} />
      ))}
    </div>
  );
}

export function ItemTile({ item }: { item: LibraryItem }) {
  if (hasThumbnail(item)) {
    return (
      <div className="size-9 shrink-0 overflow-hidden rounded-[10px] border border-border/60 bg-muted [&_img]:size-9 [&_img]:object-cover [&_img]:object-top">
        <ImageThumb item={item} />
      </div>
    );
  }
  return (
    <div className="flex size-9 shrink-0 items-center justify-center rounded-[10px] border border-border/60">
      <KindIcon item={item} className="size-5" />
    </div>
  );
}
