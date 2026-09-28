// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  type ConvExportFormat,
  EXPORT_FORMATS_LIST,
  type ProjectRecord,
  type SidebarCustomSection,
  type SidebarItem,
  OpenChatFolderItem,
  OpenProjectFolderItem,
  compareModelDisplayName,
} from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import {
  ChevronDownStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { MessageCircleIcon, StarPointedIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  Archive03Icon,
  ArchiveRestoreIcon,
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  Folder01Icon,
  Folder02Icon,
  FolderExportIcon,
  LayerIcon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  Columns2Icon,
  GitBranchIcon,
} from "lucide-react";
import { type ReactNode, createContext, useContext } from "react";
import { formatActivityTime, formatCardTime } from "../format";
import { FILE_LIST_COLUMNS } from "../components/library-list";
import { SortRadio } from "../components/library-toolbar";
import { CARD_SHADOW, OVERLAY_CONTROL, RAISED_SURFACE } from "../surface";

// Same card style as the file tabs.
const CARD = cn(
  RAISED_SURFACE,
  CARD_SHADOW,
  "group/chat relative flex cursor-pointer flex-col gap-3 rounded-xl px-5 pb-3.5 pt-5 transition hover:bg-neutral-100 hover:shadow-none dark:hover:bg-accent/60",
);
import {
  type ChatContents,
  type ChatSort,
  type ChatSortKey,
  type DateBucket,
  type DateField,
  type ProjectStats,
  type SectionStats,
  chatTime,
  projectTime,
} from "./model";

const ICON = "size-icon";
// Library menu surface, so these menus match the file tabs' in both themes.
const MENU = "library-actions-menu";
// Fits the window, and each group scrolls past ~7 rows so Sections stays reachable.
const MOVE_TO_MENU =
  "max-h-[var(--radix-dropdown-menu-content-available-height)] overflow-y-auto";
const MOVE_TO_LIST =
  "no-scrollbar -my-0.5 max-h-[calc(260px*var(--ui-space-scale,1))] overflow-y-auto overscroll-contain";
const MENU_LABEL = "px-3 pb-1 pt-2 font-normal text-muted-foreground";

/** Move target; a null project or section id removes the chat from its current one. */
export type ChatDestination =
  | { kind: "project"; id: string | null }
  | { kind: "section"; id: string | null }
  | { kind: "newProject" }
  | { kind: "newSection" };

export interface ChatsActions {
  projects: ProjectRecord[];
  projectNames: ReadonlyMap<string, string>;
  pinned: ReadonlySet<string>;
  pinnedProjects: ReadonlySet<string>;
  favorites: ReadonlySet<string>;
  favoriteProjects: ReadonlySet<string>;
  /** False in Favorites, where chats do not share the file selection. */
  selectable: boolean;
  models: ReadonlyMap<string, string[]>;
  sections: readonly SidebarCustomSection[];
  /** Chat row id -> the section it is filed in. */
  sectionOf: ReadonlyMap<string, string>;
  /** Project id -> the section it is filed in. */
  projectSectionOf: ReadonlyMap<string, string>;
  /** The date the lists show. */
  dateField: DateField;
  /** Per chat row id; absent until read. */
  chatContents: ReadonlyMap<string, ChatContents>;
  /** Sources per project; null when unknown. */
  projectSources: ReadonlyMap<string, number> | null;
  selection: ReadonlySet<string>;
  toggleSelected: (id: string) => void;
  open: (chat: SidebarItem) => void;
  rename: (chat: SidebarItem) => void;
  togglePin: (chat: SidebarItem) => void;
  setFavorite: (chats: SidebarItem[], favorite: boolean) => void;
  toggleFavoriteProject: (projectId: string) => void;
  fork: (chat: SidebarItem) => void;
  move: (chats: SidebarItem[], destination: ChatDestination) => void;
  /** Moves a project into or out of a section; projects never nest. */
  moveProject: (project: ProjectRecord, destination: ChatDestination) => void;
  /** Narrows the list to one section. Omitted where there are no filters (Favorites). */
  viewSection: (sectionId: string) => void;
  newChatInSection: (sectionId: string) => void;
  renameSection: (section: SidebarCustomSection) => void;
  removeSection: (section: SidebarCustomSection) => void;
  exportSection: (
    section: SidebarCustomSection,
    format: ConvExportFormat,
  ) => void;
  archive: (chats: SidebarItem[]) => void;
  unarchive: (chats: SidebarItem[]) => void;
  exportChats: (chats: SidebarItem[], format: ConvExportFormat) => void;
  remove: (chats: SidebarItem[]) => void;
  viewProject: (projectId: string) => void;
  newChatIn: (projectId: string | null) => void;
  editProject: (project: ProjectRecord) => void;
  togglePinProject: (projectId: string) => void;
  exportProject: (project: ProjectRecord, format: ConvExportFormat) => void;
  deleteProject: (project: ProjectRecord) => void;
}

const ChatsActionsContext = createContext<ChatsActions | null>(null);
export const ChatsActionsProvider = ChatsActionsContext.Provider;

function useChatsActions(): ChatsActions {
  const actions = useContext(ChatsActionsContext);
  if (!actions) throw new Error("useChatsActions outside ChatsActionsProvider");
  return actions;
}

function MenuItem({
  icon,
  label,
  onSelect,
  destructive,
  disabled,
}: {
  icon: IconSvgElement;
  label: string;
  onSelect: () => void;
  destructive?: boolean;
  disabled?: boolean;
}) {
  return (
    <DropdownMenuItem
      onSelect={onSelect}
      disabled={disabled}
      variant={destructive ? "destructive" : "default"}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      <span className="truncate">{label}</span>
    </DropdownMenuItem>
  );
}

export function ExportSubmenu({
  onExport,
}: {
  onExport: (format: ConvExportFormat) => void;
}) {
  const t = useT();
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger className="gap-2.5">
        <HugeiconsIcon
          icon={Download01Icon}
          strokeWidth={1.75}
          className={ICON}
        />
        {t("common.export")}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className={cn(MENU, "w-48")}>
        {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
          <DropdownMenuItem key={fmt} onSelect={() => onExport(fmt)}>
            {label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

/** "Move to" menu, as in the sidebar. The current location is omitted, not greyed out. */
export function MoveSubmenu({
  project,
  section,
  onMove,
  sectionsOnly = false,
}: {
  /** Shared project of the moved chats (null for none); undefined when they differ. */
  project?: string | null;
  /** Same, for sections. */
  section?: string | null;
  onMove: (destination: ChatDestination) => void;
  /** Sections only, for moving a project (projects never nest). */
  sectionsOnly?: boolean;
}) {
  const t = useT();
  const { projects, projectNames, sections } = useChatsActions();
  const projectTargets = projects.filter((entry) => entry.id !== project);
  const sectionTargets = sections.filter((entry) => entry.id !== section);
  const leavingSection = section
    ? sections.find((entry) => entry.id === section)
    : undefined;
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger className="gap-2.5">
        <HugeiconsIcon
          icon={FolderExportIcon}
          strokeWidth={1.75}
          className={ICON}
        />
        {t("shell.sections.moveTo")}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className={cn(MENU, MOVE_TO_MENU, "w-56")}>
        {!sectionsOnly && (
          <>
            <DropdownMenuLabel className={MENU_LABEL}>
              {t("shell.navigation.projects")}
            </DropdownMenuLabel>
            <MenuItem
              icon={PlusSignIcon}
              label={t("library.chats.toolbar.newProject")}
              onSelect={() => onMove({ kind: "newProject" })}
            />
            {projectTargets.length > 0 && (
              <div className={MOVE_TO_LIST}>
                {projectTargets.map((entry) => (
                  <MenuItem
                    key={entry.id}
                    icon={Folder01Icon}
                    label={entry.name}
                    onSelect={() => onMove({ kind: "project", id: entry.id })}
                  />
                ))}
              </div>
            )}
            {project !== null && (
              <MenuItem
                icon={Cancel01Icon}
                label={
                  project && projectNames.has(project)
                    ? t("shell.sections.removeFrom", {
                        name: projectNames.get(project) ?? "",
                      })
                    : t("shell.sections.removeFromProject")
                }
                onSelect={() => onMove({ kind: "project", id: null })}
              />
            )}
            <DropdownMenuSeparator className="mx-3" />
          </>
        )}
        <DropdownMenuLabel className={MENU_LABEL}>
          {t("shell.sections.sectionsHeading")}
        </DropdownMenuLabel>
        <MenuItem
          icon={PlusSignIcon}
          label={t("shell.sections.newSection")}
          onSelect={() => onMove({ kind: "newSection" })}
        />
        {sectionTargets.length > 0 && (
          <div className={MOVE_TO_LIST}>
            {sectionTargets.map((entry) => (
              <MenuItem
                key={entry.id}
                icon={LayerIcon}
                label={entry.name}
                onSelect={() => onMove({ kind: "section", id: entry.id })}
              />
            ))}
          </div>
        )}
        {section !== null && (
          <MenuItem
            icon={Cancel01Icon}
            label={
              leavingSection
                ? t("shell.sections.removeFrom", { name: leavingSection.name })
                : t("shell.sections.removeFromSection")
            }
            onSelect={() => onMove({ kind: "section", id: null })}
          />
        )}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

/** "More actions" trigger for rows and cards, matching the file tabs. */
function MenuTrigger({ variant }: { variant: "row" | "card" }) {
  const t = useT();
  return (
    <DropdownMenuTrigger asChild>
      <button
        type="button"
        aria-label={t("library.menu.moreActions")}
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full outline-none transition-opacity focus-visible:opacity-100 data-[state=open]:opacity-100",
          variant === "row"
            ? "text-muted-foreground opacity-0 hover:bg-accent hover:text-foreground group-hover/chat:opacity-100"
            : cn(
                OVERLAY_CONTROL,
                "text-muted-foreground opacity-0 hover:text-foreground group-hover/chat:opacity-100 dark:text-white dark:hover:bg-neutral-600",
              ),
        )}
      >
        <HugeiconsIcon
          icon={MoreHorizontalIcon}
          strokeWidth={1.75}
          className="size-5"
        />
      </button>
    </DropdownMenuTrigger>
  );
}

function FavoriteItem({
  favorite,
  onToggle,
}: {
  favorite: boolean;
  onToggle: () => void;
}) {
  const t = useT();
  return (
    <DropdownMenuItem onSelect={onToggle}>
      <HugeiconsIcon
        icon={StarPointedIcon}
        strokeWidth={1.75}
        className={cn(ICON, favorite && "[&_path]:fill-current")}
      />
      {t(
        favorite
          ? "library.menu.removeFromFavorites"
          : "library.menu.addToFavorites",
      )}
    </DropdownMenuItem>
  );
}

function FavoriteMark() {
  const t = useT();
  return (
    <HugeiconsIcon
      icon={StarPointedIcon}
      aria-label={t("library.list.favorite")}
      strokeWidth={1.75}
      className="size-3.5 shrink-0 text-muted-foreground [&_path]:fill-current"
    />
  );
}

/** Stops clicks here (portals included) from opening the chat underneath. */
function Isolate({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: only swallows bubbling clicks
    <span className={className} onClick={(event) => event.stopPropagation()}>
      {children}
    </span>
  );
}

function ChatMenu({
  chat,
  archived,
  variant,
}: {
  chat: SidebarItem;
  archived: boolean;
  variant: "row" | "card";
}) {
  const t = useT();
  const actions = useChatsActions();
  const pinned = actions.pinned.has(chat.id);
  return (
    <Isolate
      className={cn(variant === "card" && "absolute right-2 top-2 z-10")}
    >
      <DropdownMenu>
        <MenuTrigger variant={variant} />
        <DropdownMenuContent align="end" className={cn(MENU, "w-52")}>
          <MenuItem
            icon={MessageCircleIcon}
            label={t("library.chats.menu.open")}
            onSelect={() => actions.open(chat)}
          />
          {!archived && <OpenChatFolderItem item={chat} />}
          <DropdownMenuSeparator className="mx-3" />
          <MenuItem
            icon={Edit03Icon}
            label={t("common.rename")}
            onSelect={() => actions.rename(chat)}
          />
          {!archived && (
            <>
              <MenuItem
                icon={pinned ? PinOffIcon : PinIcon}
                label={t(
                  pinned
                    ? "settings.data.library.unpin"
                    : "settings.data.library.pin",
                )}
                onSelect={() => actions.togglePin(chat)}
              />
              <FavoriteItem
                favorite={actions.favorites.has(chat.id)}
                onToggle={() =>
                  actions.setFavorite([chat], !actions.favorites.has(chat.id))
                }
              />
            </>
          )}
          <DropdownMenuSeparator className="mx-3" />
          {!archived && (
            <>
              {chat.type === "single" && (
                <DropdownMenuItem onSelect={() => actions.fork(chat)}>
                  <GitBranchIcon strokeWidth={1.75} className={ICON} />
                  {t("library.chats.menu.fork")}
                </DropdownMenuItem>
              )}
              <MoveSubmenu
                project={chat.projectId ?? null}
                section={actions.sectionOf.get(chat.id) ?? null}
                onMove={(destination) => actions.move([chat], destination)}
              />
            </>
          )}
          <ExportSubmenu
            onExport={(format) => actions.exportChats([chat], format)}
          />
          <DropdownMenuSeparator className="mx-3" />
          {archived ? (
            <MenuItem
              icon={ArchiveRestoreIcon}
              label={t("settings.data.library.unarchive")}
              onSelect={() => actions.unarchive([chat])}
            />
          ) : (
            <MenuItem
              icon={Archive03Icon}
              label={t("settings.data.library.archive")}
              onSelect={() => actions.archive([chat])}
            />
          )}
          <MenuItem
            icon={Delete02Icon}
            label={t("common.delete")}
            destructive
            onSelect={() => actions.remove([chat])}
          />
        </DropdownMenuContent>
      </DropdownMenu>
    </Isolate>
  );
}

const TILE_ICON = "size-[calc(18px*var(--ui-space-scale,1))]";

/** Chat kind icon (bubble, compare, fork) as drawn elsewhere in the app, without a box. */
function ChatTile({
  chat,
  className,
}: {
  chat: SidebarItem;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex size-9 shrink-0 items-center justify-center text-muted-foreground",
        className,
      )}
    >
      {chat.type === "compare" ? (
        <Columns2Icon strokeWidth={1.5} className={TILE_ICON} />
      ) : chat.isFork ? (
        <GitBranchIcon strokeWidth={1.5} className={TILE_ICON} />
      ) : (
        <HugeiconsIcon
          icon={MessageCircleIcon}
          strokeWidth={1.5}
          className={TILE_ICON}
        />
      )}
    </div>
  );
}

function Badge({ children }: { children: ReactNode }) {
  return (
    <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-ui-11 font-medium text-muted-foreground">
      {children}
    </span>
  );
}

function ChatBadges({ chat }: { chat: SidebarItem }) {
  const t = useT();
  const { pinned, favorites } = useChatsActions();
  return (
    <>
      {favorites.has(chat.id) && <FavoriteMark />}
      {pinned.has(chat.id) && (
        <HugeiconsIcon
          icon={PinIcon}
          aria-label={t("library.chats.badges.pinned")}
          strokeWidth={1.75}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
      )}
      {chat.isFork && <Badge>{t("library.chats.badges.fork")}</Badge>}
      {chat.type === "compare" && (
        <Badge>{t("library.chats.badges.compare")}</Badge>
      )}
    </>
  );
}

function chatTitle(chat: SidebarItem, t: ReturnType<typeof useT>): string {
  return chat.title?.trim() || t("settings.data.library.untitled");
}

function modelLabel(
  chat: SidebarItem,
  models: ReadonlyMap<string, string[]>,
): string {
  return (models.get(chat.id) ?? []).map(compareModelDisplayName).join(" · ");
}

// Plain text, not a pill: a pill inside a hovered row looked like a second row.
const CHIP =
  "inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-sm text-left outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring";

/** Project and section links that narrow the list; nothing when the chat has neither. */
function ChatLocation({
  chat,
  showProject,
  showSection,
}: {
  chat: SidebarItem;
  showProject: boolean;
  showSection: boolean;
}) {
  const t = useT();
  const { projectNames, viewProject, sectionOf, sections, viewSection } =
    useChatsActions();
  const projectId = showProject ? chat.projectId : null;
  const sectionId = showSection ? sectionOf.get(chat.id) : undefined;
  const section = sectionId
    ? sections.find((entry) => entry.id === sectionId)
    : undefined;
  if (!projectId && !section) return null;
  const projectName = projectId
    ? (projectNames.get(projectId) ??
      t("settings.data.library.unavailableProject"))
    : "";
  return (
    <span className="flex min-w-0 items-center gap-3">
      {projectId && (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            viewProject(projectId);
          }}
          className={cn(CHIP, "shrink")}
        >
          <HugeiconsIcon
            icon={Folder01Icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
          />
          <span className="truncate">{projectName}</span>
        </button>
      )}
      {section && (
        <SectionChip
          section={section}
          onView={viewSection}
          // The project name truncates first so a short section name stays whole.
          className={projectId ? "max-w-[55%] shrink-0" : "shrink"}
        />
      )}
    </span>
  );
}

function SectionChip({
  section,
  onView,
  className,
}: {
  section: SidebarCustomSection;
  onView: (sectionId: string) => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onView(section.id);
      }}
      className={cn(CHIP, className)}
    >
      <HugeiconsIcon
        icon={LayerIcon}
        strokeWidth={1.75}
        className="size-3.5 shrink-0"
      />
      <span className="truncate">{section.name}</span>
    </button>
  );
}

/** "own": under its own header; "files": among files; object: among chats (All). */
export type CollectionRowLayout = "own" | "files" | { showLocation: boolean };

/** File-list Modified and Size cells, for chats or projects listed among files. */
function FileColumns({ modified }: { modified: number }) {
  const locale = useLocale();
  return (
    <>
      <span className={cn(FILE_LIST_COLUMNS.modified, FILE_LIST_COLUMNS.cell)}>
        {formatCardTime(modified, locale)}
      </span>
      <span className={cn(FILE_LIST_COLUMNS.size, FILE_LIST_COLUMNS.cell)} />
    </>
  );
}

function CollectionCount({ children }: { children: ReactNode }) {
  return (
    <span className="shrink-0 text-ui-13 text-muted-foreground">
      {children}
    </span>
  );
}

/** Time inside a date group, which already names the day: so no relative "6 hr. ago". */
function groupedTime(
  ts: number,
  bucket: DateBucket["kind"],
  locale: string,
): string {
  const date = new Date(ts);
  if (bucket === "today" || bucket === "yesterday") {
    return date.toLocaleTimeString(locale, {
      hour: "numeric",
      minute: "2-digit",
    });
  }
  if (bucket === "week") {
    return date.toLocaleDateString(locale, { weekday: "long" });
  }
  return date.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year:
      date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

/** Relative times by default; inside a date group, `groupedTime`. */
export interface RowTimes {
  bucket?: DateBucket["kind"];
}

/** A date in the chosen field: created as a date, the others relative. Empty for 0. */
function formatDate(
  ts: number,
  field: DateField,
  times: RowTimes,
  locale: ReturnType<typeof useLocale>,
  t: ReturnType<typeof useT>,
): string {
  if (!ts) return "";
  if (times.bucket) return groupedTime(ts, times.bucket, locale);
  return field === "created"
    ? formatCardTime(ts, locale)
    : formatActivityTime(ts, locale, t);
}

function countLabel(
  count: number,
  one: TranslationKey,
  many: TranslationKey,
  t: ReturnType<typeof useT>,
): string {
  return count === 1 ? t(one) : t(many, { count });
}

function chatContentsLabel(
  contents: ChatContents | undefined,
  t: ReturnType<typeof useT>,
): string {
  if (!contents) return "";
  const parts = [
    countLabel(
      contents.messages,
      "library.chats.list.oneMessage",
      "library.chats.list.messageCount",
      t,
    ),
  ];
  if (contents.images > 0) {
    parts.push(
      countLabel(
        contents.images,
        "library.chats.list.oneImage",
        "library.chats.list.imageCount",
        t,
      ),
    );
  }
  if (contents.html > 0) {
    parts.push(
      countLabel(
        contents.html,
        "library.chats.list.oneHtml",
        "library.chats.list.htmlCount",
        t,
      ),
    );
  }
  return parts.join(" · ");
}

function projectContentsLabel(
  stats: ProjectStats | undefined,
  sources: number | undefined,
  t: ReturnType<typeof useT>,
): string {
  const chats = chatCount(stats?.chats ?? 0, t);
  if (!sources) return chats;
  return `${chats} · ${countLabel(sources, "library.chats.list.oneSource", "library.chats.list.sourceCount", t)}`;
}

const DATE_LABELS: Record<DateField, TranslationKey> = {
  created: "library.chats.list.created",
  updated: "library.chats.list.lastActive",
  modified: "library.chats.list.lastModified",
};

/** One date column: its title picks the date, its arrow flips the order. */
export function DateHeader({
  fields,
  field,
  sortKey,
  desc,
  onFieldChange,
  onToggle,
}: {
  fields: readonly DateField[];
  field: DateField;
  sortKey: string;
  desc: boolean;
  onFieldChange: (field: DateField) => void;
  onToggle: () => void;
}) {
  const t = useT();
  const active = sortKey === field;
  const Arrow = active && !desc ? ArrowUpIcon : ArrowDownIcon;
  const label = t(DATE_LABELS[field]);
  return (
    <span className={cn(DATE_COLUMN, "items-center gap-1 @xl:flex")}>
      {fields.length > 1 ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              className={cn(
                "flex items-center gap-1 rounded-sm outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:text-foreground",
                active && "text-foreground",
              )}
            >
              {label}
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={2}
                className="size-3"
              />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className={cn(MENU, "w-44")}>
            {fields.map((option) => (
              <SortRadio
                key={option}
                label={t(DATE_LABELS[option])}
                checked={option === field}
                onSelect={() => onFieldChange(option)}
              />
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <span className={cn(active && "text-foreground")}>{label}</span>
      )}
      <button
        type="button"
        onClick={onToggle}
        aria-label={t(
          active && desc
            ? "library.toolbar.sortAscending"
            : "library.toolbar.sortDescending",
        )}
        aria-sort={active ? (desc ? "descending" : "ascending") : undefined}
        className={cn(
          "flex size-5 items-center justify-center rounded-full outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring",
          active ? "text-foreground" : "opacity-50",
        )}
      >
        <Arrow className="size-3.5" strokeWidth={2} />
      </button>
    </span>
  );
}

/** Header props for a list's date column. */
export interface DateColumn {
  fields: readonly DateField[];
  sortKey: string;
  desc: boolean;
  onFieldChange: (field: DateField) => void;
  onToggle: () => void;
}

function SelectBox({
  chat,
  visible,
  className,
}: {
  chat: SidebarItem;
  visible: boolean;
  className?: string;
}) {
  const t = useT();
  const { selection, toggleSelected, selectable } = useChatsActions();
  if (!selectable) return null;
  return (
    <Isolate className={className}>
      <Checkbox
        checked={selection.has(chat.id)}
        onCheckedChange={() => toggleSelected(chat.id)}
        aria-label={t("settings.data.library.selectItem", {
          title: chatTitle(chat, t),
        })}
        className={cn(
          "rounded-full border-neutral-300 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/chat:opacity-100 dark:border-input",
          visible && "opacity-100",
        )}
      />
    </Isolate>
  );
}

// Same row insets as the file tabs, so columns line up.
const ROW_INSET = "pl-4 pr-6";
const CELL = "truncate text-ui-13 text-muted-foreground";
// Columns follow the list's width, not the window's; the date column hides last.
const DATE_COLUMN = "hidden w-32 shrink-0 @xl:block";
const CONTENTS_COLUMN = "hidden w-64 shrink-0 @3xl:block";
const LOCATION_COLUMN = "hidden w-48 shrink-0 @4xl:block";

function SortHeader({
  column,
  label,
  sort,
  onSortChange,
  className,
}: {
  column: ChatSortKey;
  label: string;
  sort: ChatSort;
  onSortChange: (key: ChatSortKey) => void;
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

export function ChatListHeader({
  sort,
  onSortChange,
  date,
  showLocation,
  allSelected,
  selecting,
  onToggleAll,
}: {
  sort: ChatSort;
  onSortChange: (key: ChatSortKey) => void;
  date: DateColumn;
  showLocation: boolean;
  allSelected: boolean;
  selecting: boolean;
  onToggleAll: () => void;
}) {
  const t = useT();
  const { selectable, dateField } = useChatsActions();
  return (
    // Padding outside the row, as in the file list, so the checkbox and Name align with rows.
    <div className="pb-2">
      <div
        className={cn(
          "group/chat relative flex items-center gap-4 text-ui-13 text-muted-foreground",
          ROW_INSET,
        )}
      >
        {selectable && (
          <div className="absolute right-full top-1/2 mr-3 flex -translate-y-1/2">
            <Checkbox
              checked={allSelected}
              onCheckedChange={onToggleAll}
              aria-label={t("settings.data.library.selectAll")}
              className={cn(
                "rounded-full border-neutral-300 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/chat:opacity-100 dark:border-input",
                selecting && "opacity-100",
              )}
            />
          </div>
        )}
        <span className="min-w-0 flex-1">
          <SortHeader
            column="name"
            label={t("library.list.name")}
            sort={sort}
            onSortChange={onSortChange}
          />
        </span>
        {showLocation && (
          <span className={LOCATION_COLUMN}>
            {t("library.chats.list.location")}
          </span>
        )}
        <span className={CONTENTS_COLUMN}>
          {t("library.chats.list.contents")}
        </span>
        <DateHeader {...date} field={dateField} />
        <span className="w-8 shrink-0" />
      </div>
    </div>
  );
}

export function ChatRow({
  chat,
  archived,
  showProject,
  showSection,
  locationColumn = showProject || showSection,
  fileColumns = false,
  times = {},
}: {
  chat: SidebarItem;
  archived: boolean;
  /** Off inside a project or when grouped by it (sections likewise). */
  showProject: boolean;
  showSection: boolean;
  /** Draw the Location column even if this row names nothing (All: a project may need it). */
  locationColumn?: boolean;
  /** Among files (Favorites): use the file list's columns. */
  fileColumns?: boolean;
  times?: RowTimes;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  const selected = actions.selection.has(chat.id);
  const model = modelLabel(chat, actions.models);
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the title button is the keyboard target
    <div
      onClick={() =>
        actions.selection.size > 0
          ? actions.toggleSelected(chat.id)
          : actions.open(chat)
      }
      className={cn(
        "group/chat relative flex cursor-pointer items-center gap-4 rounded-[14px] transition-colors hover:bg-muted dark:hover:bg-muted/60",
        ROW_INSET,
        selected && "bg-muted dark:bg-muted/60",
      )}
    >
      <SelectBox
        chat={chat}
        visible={actions.selection.size > 0}
        className="absolute right-full top-1/2 mr-3 flex -translate-y-1/2"
      />
      <div className="flex min-w-0 flex-1 items-center gap-4 py-2">
        <ChatTile chat={chat} />
        <div className="flex min-w-0 flex-col">
          <span className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                if (actions.selection.size > 0) actions.toggleSelected(chat.id);
                else actions.open(chat);
              }}
              className="truncate rounded text-left text-ui-14 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {chatTitle(chat, t)}
            </button>
            <ChatBadges chat={chat} />
          </span>
          {model && (
            <span className="truncate text-ui-12 text-muted-foreground">
              {model}
            </span>
          )}
        </div>
      </div>
      {fileColumns ? (
        <FileColumns modified={chat.updatedAt} />
      ) : (
        <>
          {locationColumn && (
            <span className={cn(LOCATION_COLUMN, CELL)}>
              <ChatLocation
                chat={chat}
                showProject={showProject}
                showSection={showSection}
              />
            </span>
          )}
          <span className={cn(CONTENTS_COLUMN, CELL)}>
            {chatContentsLabel(actions.chatContents.get(chat.id), t)}
          </span>
          <span className={cn(DATE_COLUMN, CELL)}>
            {formatDate(
              chatTime(chat, actions.dateField),
              actions.dateField,
              times,
              locale,
              t,
            )}
          </span>
        </>
      )}
      <ChatMenu chat={chat} archived={archived} variant="row" />
    </div>
  );
}

export function ChatCard({
  chat,
  archived,
  showProject,
  showSection,
  times = {},
}: {
  chat: SidebarItem;
  archived: boolean;
  showProject: boolean;
  showSection: boolean;
  times?: RowTimes;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  const selected = actions.selection.has(chat.id);
  const selecting = actions.selection.size > 0;
  const model = modelLabel(chat, actions.models);
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the title button is the keyboard target
    <div
      onClick={() =>
        selecting ? actions.toggleSelected(chat.id) : actions.open(chat)
      }
      className={cn(CARD, "min-h-40", selected && "ring-2 ring-foreground")}
    >
      <div className="relative flex items-center gap-2">
        <ChatTile
          chat={chat}
          className={cn(
            "transition-opacity group-hover/chat:opacity-0",
            selecting && "opacity-0",
          )}
        />
        <SelectBox
          chat={chat}
          visible={selecting}
          className="absolute left-2.5 top-1/2 flex -translate-y-1/2"
        />
        <span className="flex min-w-0 items-center gap-1.5">
          <ChatBadges chat={chat} />
        </span>
      </div>
      <ChatMenu chat={chat} archived={archived} variant="card" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          if (selecting) actions.toggleSelected(chat.id);
          else actions.open(chat);
        }}
        className="block w-full rounded text-left font-medium text-ui-15 leading-snug text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {/* Clamp an inner span: buttons ignore line-clamp. */}
        <span className="line-clamp-3">{chatTitle(chat, t)}</span>
      </button>
      <div className="mt-auto flex min-w-0 flex-col gap-1 text-ui-12 text-muted-foreground">
        {model && <span className="truncate">{model}</span>}
        <span className="flex min-w-0 items-center justify-between gap-2">
          <span className="min-w-0">
            <ChatLocation
              chat={chat}
              showProject={showProject}
              showSection={showSection}
            />
          </span>
          <span className="shrink-0">
            {formatDate(
              chatTime(chat, actions.dateField),
              actions.dateField,
              times,
              locale,
              t,
            )}
          </span>
        </span>
      </div>
    </div>
  );
}

/** Group name and count in words; a bare number beside "Today" read as part of the name. */
export function GroupHeading({
  children,
  count,
  countLabel,
}: {
  children: ReactNode;
  count: number;
  /** In place of "N chats", for a group of something else. */
  countLabel?: string;
}) {
  const t = useT();
  return (
    <h3 className="mb-1 mt-7 flex items-baseline gap-2 pl-4 font-sans text-ui-13 first:mt-3">
      <span className="font-medium text-foreground">{children}</span>
      <span className="text-ui-12 text-muted-foreground">
        {countLabel ?? chatCount(count, t)}
      </span>
    </h3>
  );
}

/** Project or section tile, as on the Projects page; brand tint on row hover. */
function CollectionTile({ icon }: { icon: IconSvgElement }) {
  return (
    <span className="flex size-9 shrink-0 items-center justify-center rounded-[10px] bg-muted text-foreground/70 transition-colors group-hover/chat:bg-primary/10 group-hover/chat:text-primary">
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
    </span>
  );
}

/** Project or section page header, styled like a project's home in Chat. */
export function CollectionHeader({
  icon,
  name,
  parent,
  onParent,
  marks,
  description,
  actions,
}: {
  icon: IconSvgElement;
  name: string;
  /** Back-link label: Projects or Sections. */
  parent: string;
  onParent: () => void;
  marks?: ReactNode;
  description?: string;
  actions: ReactNode;
}) {
  const t = useT();
  return (
    <div className="flex flex-col gap-4">
      <nav
        aria-label={t("library.breadcrumb")}
        className="flex items-center gap-1.5 text-ui-14"
      >
        <button
          type="button"
          onClick={onParent}
          className="rounded text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
        >
          {parent}
        </button>
        <HugeiconsIcon
          icon={ChevronRightStandardIcon}
          strokeWidth={2}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
        <span aria-current="page" className="truncate text-foreground">
          {name}
        </span>
      </nav>
      <div className="flex items-center gap-4">
        <span className="flex size-13 shrink-0 items-center justify-center rounded-[18px] bg-muted text-foreground/80">
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-6.5" />
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <h2 className="flex min-w-0 items-center gap-2.5 text-ui-30 font-medium leading-tight text-foreground">
            <span className="truncate">{name}</span>
            {marks}
          </h2>
          {description && (
            <p className="line-clamp-2 max-w-3xl text-ui-14 text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      </div>
    </div>
  );
}

/** Round "more" button, matching a project's home in Chat. */
export const HEADER_MORE_BUTTON =
  "inline-flex size-9 shrink-0 items-center justify-center rounded-full text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:bg-muted data-[state=open]:text-foreground";

function ProjectMenu({
  project,
  variant,
}: {
  project: ProjectRecord;
  variant: "row" | "card";
}) {
  const t = useT();
  const actions = useChatsActions();
  const pinned = actions.pinnedProjects.has(project.id);
  return (
    <Isolate
      className={cn(variant === "card" && "absolute right-2 top-2 z-10")}
    >
      <DropdownMenu>
        <MenuTrigger variant={variant} />
        <DropdownMenuContent align="end" className={cn(MENU, "w-52")}>
          <MenuItem
            icon={Folder01Icon}
            label={t("library.chats.menu.viewChats")}
            onSelect={() => actions.viewProject(project.id)}
          />
          <MenuItem
            icon={PencilEdit02Icon}
            label={t("library.chats.menu.newChatInProject")}
            onSelect={() => actions.newChatIn(project.id)}
          />
          <OpenProjectFolderItem projectId={project.id} />
          <DropdownMenuSeparator className="mx-3" />
          <MenuItem
            icon={Settings02Icon}
            label={t("library.chats.menu.editProject")}
            onSelect={() => actions.editProject(project)}
          />
          <MenuItem
            icon={pinned ? PinOffIcon : PinIcon}
            label={t(
              pinned
                ? "settings.data.library.unpin"
                : "settings.data.library.pin",
            )}
            onSelect={() => actions.togglePinProject(project.id)}
          />
          <FavoriteItem
            favorite={actions.favoriteProjects.has(project.id)}
            onToggle={() => actions.toggleFavoriteProject(project.id)}
          />
          <DropdownMenuSeparator className="mx-3" />
          <MoveSubmenu
            sectionsOnly
            section={actions.projectSectionOf.get(project.id) ?? null}
            onMove={(destination) => actions.moveProject(project, destination)}
          />
          <ExportSubmenu
            onExport={(format) => actions.exportProject(project, format)}
          />
          <DropdownMenuSeparator className="mx-3" />
          <MenuItem
            icon={Delete02Icon}
            label={t("library.chats.menu.deleteProject")}
            destructive
            onSelect={() => actions.deleteProject(project)}
          />
        </DropdownMenuContent>
      </DropdownMenu>
    </Isolate>
  );
}

function chatCount(count: number, t: ReturnType<typeof useT>): string {
  return count === 1
    ? t("settings.data.library.oneChat")
    : t("settings.data.library.chatCount", { count });
}

export function ProjectCard({
  project,
  stats,
}: {
  project: ProjectRecord;
  stats?: ProjectStats;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  const pinned = actions.pinnedProjects.has(project.id);
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the name button is the keyboard target
    <div
      onClick={() => actions.viewProject(project.id)}
      className={cn(CARD, "min-h-44")}
    >
      <div className="flex items-center gap-2">
        <CollectionTile icon={Folder02Icon} />
        {actions.favoriteProjects.has(project.id) && <FavoriteMark />}
        {pinned && (
          <HugeiconsIcon
            icon={PinIcon}
            aria-label={t("library.chats.badges.pinned")}
            strokeWidth={1.75}
            className="size-3.5 text-muted-foreground"
          />
        )}
      </div>
      <ProjectMenu project={project} variant="card" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          actions.viewProject(project.id);
        }}
        className="truncate rounded text-left font-medium text-ui-15 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {project.name}
      </button>
      <p className="line-clamp-2 text-ui-13 text-muted-foreground">
        {project.instructions?.trim() ||
          t("library.chats.project.noInstructions")}
      </p>
      <div className="mt-auto flex items-center justify-between gap-2 text-ui-12 text-muted-foreground">
        <span>
          {chatCount(stats?.chats ?? 0, t)}
          {(stats?.archived ?? 0) > 0 &&
            ` · ${t("library.chats.project.archivedCount", { count: stats?.archived ?? 0 })}`}
        </span>
        <span className="shrink-0">
          {formatActivityTime(
            stats?.lastActive ?? project.updatedAt,
            locale,
            t,
          )}
        </span>
      </div>
    </div>
  );
}

export function ProjectRow({
  project,
  stats,
  layout = "own",
}: {
  project: ProjectRecord;
  stats?: ProjectStats;
  layout?: CollectionRowLayout;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  const pinned = actions.pinnedProjects.has(project.id);
  const sectionId = actions.projectSectionOf.get(project.id);
  const section = sectionId
    ? actions.sections.find((entry) => entry.id === sectionId)
    : undefined;
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the name button is the keyboard target
    <div
      onClick={() => actions.viewProject(project.id)}
      className={cn(
        "group/chat relative flex cursor-pointer items-center gap-4 rounded-[14px] transition-colors hover:bg-muted dark:hover:bg-muted/60",
        ROW_INSET,
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-4 py-2">
        <CollectionTile icon={Folder02Icon} />
        <div className="flex min-w-0 flex-col">
          <span className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                actions.viewProject(project.id);
              }}
              className="truncate rounded text-left text-ui-14 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {project.name}
            </button>
            {actions.favoriteProjects.has(project.id) && <FavoriteMark />}
            {pinned && (
              <HugeiconsIcon
                icon={PinIcon}
                aria-label={t("library.chats.badges.pinned")}
                strokeWidth={1.75}
                className="size-3.5 shrink-0 text-muted-foreground"
              />
            )}
            {layout === "files" && (
              <CollectionCount>
                {chatCount(stats?.chats ?? 0, t)}
              </CollectionCount>
            )}
          </span>
          {project.instructions?.trim() && (
            <span className="truncate text-ui-12 text-muted-foreground">
              {project.instructions.trim()}
            </span>
          )}
        </div>
      </div>
      {layout !== "files" && layout !== "own" && layout.showLocation && (
        <span className={cn(LOCATION_COLUMN, CELL)}>
          {section && (
            <SectionChip
              section={section}
              onView={actions.viewSection}
              className="shrink"
            />
          )}
        </span>
      )}
      {layout === "files" ? (
        <FileColumns modified={stats?.lastActive ?? project.updatedAt} />
      ) : (
        <>
          <span className={cn(CONTENTS_COLUMN, CELL)}>
            {projectContentsLabel(
              stats,
              actions.projectSources?.get(project.id),
              t,
            )}
          </span>
          <span className={cn(DATE_COLUMN, CELL)}>
            {formatDate(
              projectTime(project, stats, actions.dateField),
              actions.dateField,
              {},
              locale,
              t,
            )}
          </span>
        </>
      )}
      <ProjectMenu project={project} variant="row" />
    </div>
  );
}

export function ProjectListHeader({ date }: { date: DateColumn }) {
  const t = useT();
  const { dateField } = useChatsActions();
  return (
    <div
      className={cn(
        "flex items-center gap-4 pb-2 text-ui-13 text-muted-foreground",
        ROW_INSET,
      )}
    >
      <span className="min-w-0 flex-1">{t("library.list.name")}</span>
      <span className={CONTENTS_COLUMN}>
        {t("library.chats.list.contents")}
      </span>
      <DateHeader {...date} field={dateField} />
      <span className="w-8 shrink-0" />
    </div>
  );
}

function SectionMenu({
  section,
  variant,
}: {
  section: SidebarCustomSection;
  variant: "row" | "card";
}) {
  return (
    <Isolate
      className={cn(variant === "card" && "absolute right-2 top-2 z-10")}
    >
      <DropdownMenu>
        <MenuTrigger variant={variant} />
        <DropdownMenuContent align="end" className={cn(MENU, "w-52")}>
          <SectionMenuItems section={section} />
        </DropdownMenuContent>
      </DropdownMenu>
    </Isolate>
  );
}

/** Section actions, shared by its row menu and its page. */
export function SectionMenuItems({
  section,
  onPage = false,
}: {
  section: SidebarCustomSection;
  onPage?: boolean;
}) {
  const t = useT();
  const actions = useChatsActions();
  return (
    <>
      {!onPage && (
        <MenuItem
          icon={LayerIcon}
          label={t("library.chats.menu.viewChats")}
          onSelect={() => actions.viewSection(section.id)}
        />
      )}
      {!onPage && (
        <MenuItem
          icon={PencilEdit02Icon}
          label={t("library.chats.menu.newChatInSection")}
          onSelect={() => actions.newChatInSection(section.id)}
        />
      )}
      <MenuItem
        icon={Edit03Icon}
        label={t("shell.sections.renameTitle")}
        onSelect={() => actions.renameSection(section)}
      />
      <ExportSubmenu
        onExport={(format) => actions.exportSection(section, format)}
      />
      <DropdownMenuSeparator className="mx-3" />
      <MenuItem
        icon={Delete02Icon}
        label={t("shell.sections.remove")}
        destructive
        onSelect={() => actions.removeSection(section)}
      />
    </>
  );
}

function sectionCountLabel(
  stats: SectionStats | undefined,
  t: ReturnType<typeof useT>,
): string {
  const chats = chatCount(stats?.chats ?? 0, t);
  const projects = stats?.projects ?? 0;
  if (projects === 0) return chats;
  return `${countLabel(projects, "library.chats.project.oneProject", "library.chats.project.projectCount", t)} · ${chats}`;
}

export function SectionCard({
  section,
  stats,
}: {
  section: SidebarCustomSection;
  stats?: SectionStats;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the name button is the keyboard target
    <div
      onClick={() => actions.viewSection(section.id)}
      className={cn(CARD, "min-h-40")}
    >
      <div className="flex items-center gap-2">
        <CollectionTile icon={LayerIcon} />
      </div>
      <SectionMenu section={section} variant="card" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          actions.viewSection(section.id);
        }}
        className="truncate rounded text-left font-medium text-ui-15 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {section.name}
      </button>
      <div className="mt-auto flex items-center justify-between gap-2 text-ui-12 text-muted-foreground">
        <span className="truncate">{sectionCountLabel(stats, t)}</span>
        {(stats?.lastActive ?? 0) > 0 && (
          <span className="shrink-0">
            {formatActivityTime(stats?.lastActive ?? 0, locale, t)}
          </span>
        )}
      </div>
    </div>
  );
}

export function SectionRow({
  section,
  stats,
  layout = "own",
}: {
  section: SidebarCustomSection;
  stats?: SectionStats;
  layout?: CollectionRowLayout;
}) {
  const t = useT();
  const locale = useLocale();
  const actions = useChatsActions();
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the name button is the keyboard target
    <div
      onClick={() => actions.viewSection(section.id)}
      className={cn(
        "group/chat relative flex cursor-pointer items-center gap-4 rounded-[14px] transition-colors hover:bg-muted dark:hover:bg-muted/60",
        ROW_INSET,
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-4 py-2">
        <CollectionTile icon={LayerIcon} />
        <span className="flex min-w-0 items-center gap-2">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              actions.viewSection(section.id);
            }}
            className="truncate rounded text-left text-ui-14 text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {section.name}
          </button>
          {layout === "files" && (
            <CollectionCount>{sectionCountLabel(stats, t)}</CollectionCount>
          )}
        </span>
      </div>
      {/* Sections have no location: an empty cell keeps the columns aligned. */}
      {layout !== "own" && layout !== "files" && layout.showLocation && (
        <span className={LOCATION_COLUMN} />
      )}
      <span className={cn(CONTENTS_COLUMN, CELL)}>
        {sectionCountLabel(stats, t)}
      </span>
      {/* A section records only when its chats were last active. */}
      <span className={cn(DATE_COLUMN, CELL)}>
        {layout === "own" || actions.dateField === "updated"
          ? formatDate(stats?.lastActive ?? 0, "updated", {}, locale, t)
          : ""}
      </span>
      <SectionMenu section={section} variant="row" />
    </div>
  );
}

export function SectionListHeader({ date }: { date: DateColumn }) {
  const t = useT();
  return (
    <div
      className={cn(
        "flex items-center gap-4 pb-2 text-ui-13 text-muted-foreground",
        ROW_INSET,
      )}
    >
      <span className="min-w-0 flex-1">{t("library.list.name")}</span>
      <span className={CONTENTS_COLUMN}>
        {t("library.chats.list.contents")}
      </span>
      <DateHeader {...date} field="updated" />
      <span className="w-8 shrink-0" />
    </div>
  );
}
