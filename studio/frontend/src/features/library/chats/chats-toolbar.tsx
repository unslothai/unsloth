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
import { type TranslationKey, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  FilterMailIcon,
  Folder02Icon,
  FolderAddIcon,
  LayerIcon,
  PencilEdit02Icon,
  PinIcon,
  Search01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowDownUpIcon, Columns2Icon, GitBranchIcon } from "lucide-react";
import type { ReactNode } from "react";
import { ROUND_BUTTON, SortRadio, VIEW_OPTIONS } from "../components/library-toolbar";
import type { LibraryView } from "../settings-store";
import {
  type ChatFilters,
  type ChatFlag,
  type ChatGroupBy,
  EMPTY_CHAT_FILTERS,
  NO_PROJECT,
  NO_SECTION,
  chatFiltersActive,
} from "./model";

const ICON = "size-icon";
const MENU_LABEL = "px-3 pb-1 pt-2 text-muted-foreground font-normal";

const FLAG_OPTIONS: { value: ChatFlag; label: TranslationKey; icon: ReactNode }[] = [
  {
    value: "favorite",
    label: "library.tabs.favorites",
    icon: <HugeiconsIcon icon={StarPointedIcon} strokeWidth={1.75} className={ICON} />,
  },
  {
    value: "pinned",
    label: "library.chats.toolbar.pinned",
    icon: <HugeiconsIcon icon={PinIcon} strokeWidth={1.75} className={ICON} />,
  },
  {
    value: "forks",
    label: "library.chats.toolbar.forks",
    icon: <GitBranchIcon strokeWidth={1.75} className={ICON} />,
  },
  {
    value: "compare",
    label: "library.chats.toolbar.comparisons",
    icon: <Columns2Icon strokeWidth={1.75} className={ICON} />,
  },
];

const GROUP_OPTIONS: { value: ChatGroupBy; label: TranslationKey }[] = [
  { value: "none", label: "library.chats.toolbar.groupNone" },
  { value: "date", label: "library.chats.toolbar.groupDate" },
  { value: "project", label: "library.chats.toolbar.groupProject" },
  { value: "section", label: "shell.sections.section" },
];

function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set);
  if (!next.delete(value)) next.add(value);
  return next;
}

function CheckItem({
  icon,
  label,
  hint,
  checked,
  onToggle,
}: {
  icon?: ReactNode;
  label: string;
  hint?: string;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <DropdownMenuItem
      onSelect={(event) => {
        event.preventDefault();
        onToggle();
      }}
    >
      {icon}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {hint && <span className="ml-3 text-xs text-muted-foreground tabular-nums">{hint}</span>}
      <HugeiconsIcon
        icon={Tick02Icon}
        strokeWidth={2}
        className={cn("ml-2 size-4 shrink-0", !checked && "invisible")}
      />
    </DropdownMenuItem>
  );
}

export interface FilterFacets {
  projects: { id: string; name: string }[];
  sections: { id: string; name: string }[];
  models: { model: string; label: string; count: number }[];
  showProjects: boolean;
}

function FilterMenu({
  filters,
  onChange,
  facets,
}: {
  filters: ChatFilters;
  onChange: (next: ChatFilters) => void;
  facets: FilterFacets;
}) {
  const t = useT();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={t("library.toolbar.filter")}
          data-active={chatFiltersActive(filters)}
          className={ROUND_BUTTON}
        >
          <HugeiconsIcon icon={FilterMailIcon} strokeWidth={1.75} className="size-5" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        sideOffset={4}
        className="library-actions-menu max-h-[min(--spacing(120),var(--radix-dropdown-menu-content-available-height))] w-64"
      >
        <DropdownMenuLabel className={MENU_LABEL}>{t("library.chats.toolbar.show")}</DropdownMenuLabel>
        {FLAG_OPTIONS.map(({ value, label, icon }) => (
          <CheckItem
            key={value}
            icon={icon}
            label={t(label)}
            checked={filters.flags.has(value)}
            onToggle={() => onChange({ ...filters, flags: toggled(filters.flags, value) })}
          />
        ))}
        {facets.showProjects && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuLabel className={MENU_LABEL}>
              {t("library.chats.toolbar.project")}
            </DropdownMenuLabel>
            <CheckItem
              label={t("settings.data.library.noProject")}
              checked={filters.projects.has(NO_PROJECT)}
              onToggle={() => onChange({ ...filters, projects: toggled(filters.projects, NO_PROJECT) })}
            />
            {facets.projects.map((project) => (
              <CheckItem
                key={project.id}
                icon={<HugeiconsIcon icon={Folder02Icon} strokeWidth={1.75} className={ICON} />}
                label={project.name}
                checked={filters.projects.has(project.id)}
                onToggle={() =>
                  onChange({ ...filters, projects: toggled(filters.projects, project.id) })
                }
              />
            ))}
          </>
        )}
        {facets.sections.length > 0 && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuLabel className={MENU_LABEL}>{t("shell.sections.section")}</DropdownMenuLabel>
            <CheckItem
              label={t("library.chats.toolbar.withoutSection")}
              checked={filters.sections.has(NO_SECTION)}
              onToggle={() => onChange({ ...filters, sections: toggled(filters.sections, NO_SECTION) })}
            />
            {facets.sections.map((section) => (
              <CheckItem
                key={section.id}
                icon={<HugeiconsIcon icon={LayerIcon} strokeWidth={1.75} className={ICON} />}
                label={section.name}
                checked={filters.sections.has(section.id)}
                onToggle={() =>
                  onChange({ ...filters, sections: toggled(filters.sections, section.id) })
                }
              />
            ))}
          </>
        )}
        {facets.models.length > 0 && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuLabel className={MENU_LABEL}>
              {t("library.chats.toolbar.model")}
            </DropdownMenuLabel>
            {facets.models.map(({ model, label, count }) => (
              <CheckItem
                key={model}
                label={label}
                hint={String(count)}
                checked={filters.models.has(model)}
                onToggle={() => onChange({ ...filters, models: toggled(filters.models, model) })}
              />
            ))}
          </>
        )}
        {chatFiltersActive(filters) && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuItem onSelect={() => onChange(EMPTY_CHAT_FILTERS)}>
              <span className="text-muted-foreground">{t("library.toolbar.clearFilters")}</span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export interface SortChoice<K extends string> {
  value: K;
  label: TranslationKey;
}

export function SortMenu<K extends string>({
  options,
  value,
  desc,
  onChange,
  groupBy,
  groupOptions = GROUP_OPTIONS.map((option) => option.value),
  onGroupByChange,
  pinnedFirst,
  onPinnedFirstChange,
}: {
  options: SortChoice<K>[];
  value: K;
  desc: boolean;
  onChange: (value: K, desc: boolean) => void;
  groupBy?: ChatGroupBy;
  groupOptions?: ChatGroupBy[];
  onGroupByChange?: (groupBy: ChatGroupBy) => void;
  pinnedFirst?: boolean;
  onPinnedFirstChange?: (pinnedFirst: boolean) => void;
}) {
  const t = useT();
  const current = options.find((option) => option.value === value);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3 text-ui-14 text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring data-open:bg-muted data-open:text-foreground dark:text-foreground/70"
        >
          <ArrowDownUpIcon strokeWidth={1.75} className="size-[calc(16px*var(--ui-space-scale,1))]" />
          {current ? t(current.label) : t("library.toolbar.sort")}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" sideOffset={4} className="library-actions-menu w-max min-w-44">
        <DropdownMenuLabel className={MENU_LABEL}>{t("library.toolbar.sort")}</DropdownMenuLabel>
        {options.map((option) => (
          <SortRadio
            key={option.value}
            label={t(option.label)}
            checked={option.value === value}
            // Name defaults to A-Z, times to newest first; re-picking keeps the direction.
            onSelect={() =>
              onChange(option.value, option.value === value ? desc : option.value !== "name")
            }
          />
        ))}
        <DropdownMenuSeparator className="mx-3" />
        <SortRadio
          label={t("library.toolbar.sortAscending")}
          checked={!desc}
          onSelect={() => onChange(value, false)}
        />
        <SortRadio
          label={t("library.toolbar.sortDescending")}
          checked={desc}
          onSelect={() => onChange(value, true)}
        />
        {groupBy !== undefined && onGroupByChange && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuLabel className={MENU_LABEL}>{t("library.chats.toolbar.groupBy")}</DropdownMenuLabel>
            {GROUP_OPTIONS.filter((option) => groupOptions.includes(option.value)).map((option) => (
              <SortRadio
                key={option.value}
                label={t(option.label)}
                checked={groupBy === option.value}
                onSelect={() => onGroupByChange(option.value)}
              />
            ))}
          </>
        )}
        {pinnedFirst !== undefined && onPinnedFirstChange && (
          <>
            <DropdownMenuSeparator className="mx-3" />
            <CheckItem
              icon={<HugeiconsIcon icon={PinIcon} strokeWidth={1.75} className={ICON} />}
              label={t("library.chats.toolbar.pinnedFirst")}
              checked={pinnedFirst}
              onToggle={() => onPinnedFirstChange(!pinnedFirst)}
            />
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function NewMenu({
  onNewChat,
  onNewProject,
  onNewSection,
}: {
  onNewChat: () => void;
  onNewProject: () => void;
  onNewSection: () => void;
}) {
  const t = useT();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-9 shrink-0 items-center gap-1.5 rounded-full bg-foreground pl-4 pr-3 font-medium text-ui-14 text-background outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t("common.new")}
          <HugeiconsIcon icon={ChevronDownStandardIcon} strokeWidth={2} className="size-4" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="library-actions-menu w-44">
        <DropdownMenuItem onSelect={onNewChat}>
          <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className={ICON} />
          {t("library.chats.toolbar.newChat")}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onNewProject}>
          <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className={ICON} />
          {t("library.chats.toolbar.newProject")}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onNewSection}>
          <HugeiconsIcon icon={LayerIcon} strokeWidth={1.75} className={ICON} />
          {t("shell.sections.newSection")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function ChatsToolbar({
  filters,
  onFiltersChange,
  facets,
  sort,
  view,
  onViewChange,
  search,
  onSearchChange,
  searchPlaceholder,
  onNewChat,
  onNewProject,
  onNewSection,
}: {
  filters?: ChatFilters;
  onFiltersChange: (next: ChatFilters) => void;
  facets: FilterFacets;
  sort: ReactNode;
  view: LibraryView;
  onViewChange: (view: LibraryView) => void;
  search: string;
  onSearchChange: (value: string) => void;
  searchPlaceholder: string;
  onNewChat: () => void;
  onNewProject: () => void;
  onNewSection: () => void;
}) {
  const t = useT();
  return (
    <div className="flex min-w-0 items-center gap-2">
      {filters && <FilterMenu filters={filters} onChange={onFiltersChange} facets={facets} />}
      {sort}
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
      <label className="relative ml-2 flex h-9 w-[min(15rem,24vw)] min-w-40 items-center rounded-full border border-border px-4 focus-within:border-ring dark:border-transparent dark:bg-card dark:focus-within:border-ring">
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
          aria-label={searchPlaceholder}
          className="ml-2.5 min-w-0 flex-1 bg-transparent text-ui-14 outline-none placeholder:text-muted-foreground dark:placeholder:text-foreground/55 [&::-webkit-search-cancel-button]:hidden"
        />
      </label>
      <NewMenu onNewChat={onNewChat} onNewProject={onNewProject} onNewSection={onNewSection} />
    </div>
  );
}
