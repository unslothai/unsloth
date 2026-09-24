// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AiEditingIcon,
  AudioWave01Icon,
  File02Icon,
  FilterMailIcon,
  FlimSlateIcon,
  Folder01Icon,
  GridViewIcon,
  Image02Icon,
  LeftToRightListBulletIcon,
  Note01Icon,
  Pdf01Icon,
  Presentation01Icon,
  Search01Icon,
  Settings02Icon,
  Tick02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { SheetIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { ChevronDownIcon } from "lucide-react";
import type { ReactNode } from "react";
import type { LibrarySource } from "../api";
import type { LibraryTypeFilter } from "../file-kind";
import { EMPTY_FILTERS, type LibraryFilters, filtersActive } from "../filters";
import type { LibraryView } from "../settings-store";

const ICON = "size-icon";
const ROUND_BUTTON =
  "flex size-9 shrink-0 items-center justify-center rounded-full text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring data-[active=true]:bg-muted data-open:bg-muted";

const SOURCE_OPTIONS: { value: LibrarySource; label: string; icon: IconSvgElement }[] = [
  { value: "uploaded", label: "Uploaded", icon: Upload01Icon },
  { value: "generated", label: "Generated", icon: AiEditingIcon },
];

const TYPE_OPTIONS: { value: LibraryTypeFilter; label: string; icon: IconSvgElement }[] = [
  { value: "images", label: "Images", icon: Image02Icon },
  { value: "videos", label: "Videos", icon: FlimSlateIcon },
  { value: "audio", label: "Audio", icon: AudioWave01Icon },
  { value: "documents", label: "Documents", icon: File02Icon },
  { value: "spreadsheets", label: "Spreadsheets", icon: SheetIcon },
  { value: "presentations", label: "Presentations", icon: Presentation01Icon },
  { value: "pdfs", label: "PDFs", icon: Pdf01Icon },
];

function toggled<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

function FilterMenu({
  filters,
  onChange,
  showTypes,
}: {
  filters: LibraryFilters;
  onChange: (next: LibraryFilters) => void;
  showTypes: boolean;
}) {
  const option = (
    { value, label, icon }: { value: string; label: string; icon: IconSvgElement },
    checked: boolean,
    toggle: () => void,
  ) => (
    <DropdownMenuItem
      key={value}
      // Stay open so several filters can be picked in one go.
      onSelect={(event) => {
        event.preventDefault();
        toggle();
      }}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      <span className="flex-1">{label}</span>
      {checked && <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-4" />}
    </DropdownMenuItem>
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="Filter"
          data-active={filtersActive(filters)}
          className={ROUND_BUTTON}
        >
          <HugeiconsIcon icon={FilterMailIcon} strokeWidth={1.75} className="size-5" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuLabel className="px-3 pb-1 pt-2 text-muted-foreground font-normal">
          Source
        </DropdownMenuLabel>
        {SOURCE_OPTIONS.map((entry) =>
          option(entry, filters.sources.has(entry.value), () =>
            onChange({ ...filters, sources: toggled(filters.sources, entry.value) }),
          ),
        )}
        {showTypes && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuLabel className="px-3 pb-1 pt-2 text-muted-foreground font-normal">
              File type
            </DropdownMenuLabel>
            {TYPE_OPTIONS.map((entry) =>
              option(entry, filters.types.has(entry.value), () =>
                onChange({ ...filters, types: toggled(filters.types, entry.value) }),
              ),
            )}
          </>
        )}
        {filtersActive(filters) && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuItem onSelect={() => onChange(EMPTY_FILTERS)}>
              <span className="pl-[calc(26px*var(--ui-space-scale,1))] text-muted-foreground">Clear filters</span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export type NewAction =
  | "note"
  | "image"
  | "folder"
  | "upload";

const NEW_OPTIONS: { value: NewAction; label: string; icon: IconSvgElement }[] = [
  { value: "note", label: "Note", icon: Note01Icon },
  { value: "image", label: "Image", icon: Image02Icon },
  { value: "folder", label: "Folder", icon: Folder01Icon },
];

function NewMenu({ onSelect }: { onSelect: (action: NewAction) => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-9 shrink-0 items-center gap-1.5 rounded-full bg-foreground pl-4 pr-3 font-medium text-[14px] text-background outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring"
        >
          New
          <ChevronDownIcon className="size-4" strokeWidth={2} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        {NEW_OPTIONS.map(({ value, label, icon }) => (
          <DropdownMenuItem key={value} onSelect={() => onSelect(value)}>
            <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
            {label}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator className="mx-3" />
        <DropdownMenuItem onSelect={() => onSelect("upload")}>
          <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className={ICON} />
          Upload files
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function LibraryToolbar({
  title,
  filters,
  onFiltersChange,
  filterMode,
  view,
  onViewChange,
  search,
  onSearchChange,
  searchPlaceholder,
  onNew,
  onSettings,
}: {
  title: ReactNode;
  filters: LibraryFilters;
  onFiltersChange: (next: LibraryFilters) => void;
  /** Folders has nothing to filter; Images only filters by source. */
  filterMode: "none" | "source" | "all";
  view: LibraryView;
  onViewChange: (view: LibraryView) => void;
  search: string;
  onSearchChange: (value: string) => void;
  searchPlaceholder: string;
  onNew: (action: NewAction) => void;
  onSettings: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-4">
      <div className="min-w-0 flex-1">{title}</div>
      <div className="flex min-w-0 items-center gap-2">
        {filterMode !== "none" && (
          <FilterMenu
            filters={filters}
            onChange={onFiltersChange}
            showTypes={filterMode === "all"}
          />
        )}
        {filterMode !== "none" && (
          <span className="mx-1.5 h-6 w-px shrink-0 bg-border" aria-hidden="true" />
        )}
        <button
          type="button"
          aria-label="Grid view"
          data-active={view === "grid"}
          onClick={() => onViewChange("grid")}
          className={ROUND_BUTTON}
        >
          <HugeiconsIcon icon={GridViewIcon} strokeWidth={1.75} className="size-5" />
        </button>
        <button
          type="button"
          aria-label="List view"
          data-active={view === "list"}
          onClick={() => onViewChange("list")}
          className={ROUND_BUTTON}
        >
          <HugeiconsIcon icon={LeftToRightListBulletIcon} strokeWidth={1.75} className="size-5" />
        </button>
        <label className="relative ml-2 flex h-9 w-[min(26rem,40vw)] min-w-40 items-center rounded-full border border-border px-4 focus-within:border-ring">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
          />
          <input
            type="search"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder={searchPlaceholder}
            className="ml-2.5 min-w-0 flex-1 bg-transparent text-[14px] outline-none placeholder:text-muted-foreground [&::-webkit-search-cancel-button]:hidden"
          />
        </label>
        <NewMenu onSelect={onNew} />
        <button
          type="button"
          aria-label="Library settings"
          onClick={onSettings}
          className={cn(ROUND_BUTTON, "text-muted-foreground hover:text-foreground dark:text-foreground")}
        >
          <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-5" />
        </button>
      </div>
    </header>
  );
}
