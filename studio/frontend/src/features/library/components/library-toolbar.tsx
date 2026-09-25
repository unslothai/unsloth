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
import { type TranslationKey, useT } from "@/i18n";
import { SheetIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { ArrowDownUpIcon, ChevronDownIcon } from "lucide-react";
import type { LibrarySource } from "../api";
import type { LibraryTypeFilter } from "../file-kind";
import { EMPTY_FILTERS, type LibraryFilters, filtersActive } from "../filters";
import type { LibrarySortKey, LibraryView } from "../settings-store";

const ICON = "size-icon";
const ROUND_BUTTON =
  "flex size-9 shrink-0 items-center justify-center rounded-full text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring data-[active=true]:bg-white data-[active=true]:shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:data-[active=true]:bg-card dark:data-[active=true]:shadow-none data-open:bg-muted";

interface MenuOption<T extends string> {
  value: T;
  label: TranslationKey;
  icon: IconSvgElement;
}

const SOURCE_OPTIONS: MenuOption<LibrarySource>[] = [
  { value: "uploaded", label: "library.toolbar.uploaded", icon: Upload01Icon },
  { value: "generated", label: "library.toolbar.generated", icon: AiEditingIcon },
];

const TYPE_OPTIONS: MenuOption<LibraryTypeFilter>[] = [
  { value: "images", label: "library.tabs.images", icon: Image02Icon },
  { value: "videos", label: "library.tabs.videos", icon: FlimSlateIcon },
  { value: "audio", label: "library.tabs.audio", icon: AudioWave01Icon },
  { value: "documents", label: "library.toolbar.documents", icon: File02Icon },
  { value: "spreadsheets", label: "library.toolbar.spreadsheets", icon: SheetIcon },
  { value: "presentations", label: "library.toolbar.presentations", icon: Presentation01Icon },
  { value: "pdfs", label: "library.toolbar.pdfs", icon: Pdf01Icon },
];

const VIEW_OPTIONS = [
  { value: "grid", label: "library.toolbar.gridView", icon: GridViewIcon },
  { value: "list", label: "library.toolbar.listView", icon: LeftToRightListBulletIcon },
] satisfies MenuOption<LibraryView>[];

export type LibrarySortChoice = "default" | LibrarySortKey;

const SORT_OPTIONS: { value: LibrarySortChoice; label: TranslationKey }[] = [
  { value: "default", label: "library.toolbar.sortDefault" },
  { value: "name", label: "library.toolbar.sortName" },
  { value: "modified", label: "library.toolbar.sortModified" },
  { value: "size", label: "library.toolbar.sortSize" },
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
  const t = useT();
  const option = (
    { value, label, icon }: MenuOption<string>,
    checked: boolean,
    toggle: () => void,
  ) => (
    <DropdownMenuItem
      key={value}
      onSelect={(event) => {
        event.preventDefault();
        toggle();
      }}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      <span className="flex-1">{t(label)}</span>
      {/* Always rendered so the menu width does not change when ticked. */}
      <HugeiconsIcon
        icon={Tick02Icon}
        strokeWidth={2}
        className={cn("ml-2 size-4", !checked && "invisible")}
      />
    </DropdownMenuItem>
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={t("library.toolbar.filter")}
          data-active={filtersActive(filters)}
          className={ROUND_BUTTON}
        >
          <HugeiconsIcon icon={FilterMailIcon} strokeWidth={1.75} className="size-5" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" sideOffset={4} className="w-max min-w-36">
        {showTypes && (
          <>
            <DropdownMenuLabel className="px-3 pb-1 pt-2 text-muted-foreground font-normal">
              {t("library.toolbar.fileType")}
            </DropdownMenuLabel>
            {TYPE_OPTIONS.map((entry) =>
              option(entry, filters.types.has(entry.value), () =>
                onChange({ ...filters, types: toggled(filters.types, entry.value) }),
              ),
            )}
            <DropdownMenuSeparator className="mx-3" />
          </>
        )}
        <DropdownMenuLabel className="px-3 pb-1 pt-2 text-muted-foreground font-normal">
          {t("library.toolbar.source")}
        </DropdownMenuLabel>
        {SOURCE_OPTIONS.map((entry) =>
          option(entry, filters.sources.has(entry.value), () =>
            onChange({ ...filters, sources: toggled(filters.sources, entry.value) }),
          ),
        )}
        {filtersActive(filters) && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuItem onSelect={() => onChange(EMPTY_FILTERS)}>
              <span className="pl-[calc(26px*var(--ui-space-scale,1))] text-muted-foreground">
                {t("library.toolbar.clearFilters")}
              </span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function SortMenu({
  value,
  onChange,
}: {
  value: LibrarySortChoice;
  onChange: (next: LibrarySortChoice) => void;
}) {
  const t = useT();
  const current = SORT_OPTIONS.find((option) => option.value === value);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3 text-ui-14 text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring data-open:bg-muted data-open:text-foreground dark:text-foreground/70"
        >
          <ArrowDownUpIcon strokeWidth={1.75} className="size-[calc(16px*var(--ui-space-scale,1))]" />
          {value === "default" || !current ? t("library.toolbar.sort") : t(current.label)}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" sideOffset={4} className="w-max min-w-36">
        {SORT_OPTIONS.map(({ value: option, label }) => {
          const checked = option === value;
          return (
            <DropdownMenuItem
              key={option}
              role="menuitemradio"
              aria-checked={checked}
              onSelect={() => onChange(option)}
            >
              <span className="flex-1">{t(label)}</span>
              <span
                aria-hidden="true"
                className={cn(
                  "ml-2 flex size-4 shrink-0 items-center justify-center rounded-full border-[1.5px]",
                  checked ? "border-foreground bg-foreground" : "border-muted-foreground/60",
                )}
              >
                {checked && <span className="size-1.5 rounded-full bg-background" />}
              </span>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export type NewAction =
  | "note"
  | "image"
  | "video"
  | "audio"
  | "model"
  | "folder"
  | "upload";

const NEW_OPTIONS: MenuOption<NewAction>[] = [
  { value: "note", label: "library.create.note", icon: Note01Icon },
  { value: "image", label: "library.create.image", icon: Image02Icon },
  { value: "video", label: "library.create.video", icon: FlimSlateIcon },
  { value: "audio", label: "library.create.audio", icon: AudioWave01Icon },
  { value: "model", label: "library.create.model", icon: TestTubeOutlineIcon },
  { value: "folder", label: "library.create.folder", icon: Folder01Icon },
];

function NewMenu({ onSelect }: { onSelect: (action: NewAction) => void }) {
  const t = useT();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-9 shrink-0 items-center gap-1.5 rounded-full bg-foreground pl-4 pr-3 font-medium text-ui-14 text-background outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t("common.new")}
          <ChevronDownIcon className="size-4" strokeWidth={2} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        {NEW_OPTIONS.map(({ value, label, icon }) => (
          <DropdownMenuItem key={value} onSelect={() => onSelect(value)}>
            <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
            {t(label)}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator className="mx-3" />
        <DropdownMenuItem onSelect={() => onSelect("upload")}>
          <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className={ICON} />
          {t("library.empty.uploadFiles")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function LibraryToolbar({
  filters,
  onFiltersChange,
  filterMode,
  view,
  onViewChange,
  sort,
  search,
  onSearchChange,
  searchPlaceholder,
  onNew,
  onSettings,
}: {
  filters: LibraryFilters;
  onFiltersChange: (next: LibraryFilters) => void;
  filterMode: "none" | "source" | "all";
  view: LibraryView;
  onViewChange: (view: LibraryView) => void;
  /** Grid view only: list view sorts by column headers. */
  sort?: { value: LibrarySortChoice; onChange: (next: LibrarySortChoice) => void };
  search: string;
  onSearchChange: (value: string) => void;
  searchPlaceholder: string;
  onNew: (action: NewAction) => void;
  onSettings: () => void;
}) {
  const t = useT();
  return (
      <div className="flex min-w-0 items-center gap-2">
        {filterMode !== "none" && (
          <FilterMenu
            filters={filters}
            onChange={onFiltersChange}
            showTypes={filterMode === "all"}
          />
        )}
        {sort && <SortMenu value={sort.value} onChange={sort.onChange} />}
        {VIEW_OPTIONS.map(({ value, label, icon }) => (
          <button
            key={value}
            type="button"
            aria-label={t(label)}
            data-active={view === value}
            aria-pressed={view === value}
            onClick={() => onViewChange(value)}
            className={ROUND_BUTTON}
          >
            <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
          </button>
        ))}
        {/* Outlined on light; a lighter fill than the page on dark, where an outline reads as a hole. */}
        <label className="relative ml-2 flex h-9 w-[min(18rem,28vw)] min-w-40 items-center rounded-full border border-border px-4 focus-within:border-ring dark:border-transparent dark:bg-card dark:focus-within:border-ring">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground dark:text-foreground/70"
          />
          <input
            type="search"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder={searchPlaceholder}
            className="ml-2.5 min-w-0 flex-1 bg-transparent text-ui-14 outline-none placeholder:text-muted-foreground dark:placeholder:text-foreground/55 [&::-webkit-search-cancel-button]:hidden"
          />
        </label>
        <NewMenu onSelect={onNew} />
        <button
          type="button"
          aria-label={t("library.toolbar.settings")}
          onClick={onSettings}
          className={cn(ROUND_BUTTON, "text-muted-foreground hover:text-foreground dark:text-foreground")}
        >
          <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-5" />
        </button>
      </div>
  );
}
