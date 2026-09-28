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
  BulkExportItems,
  type ConvExportFormat,
  type ConversationExportFormat,
  chatExportOptions,
  type ProjectRecord,
  type SidebarCustomSection,
  type SidebarItem,
  OpenChatFolderItem,
  OpenProjectFolderItem,
  canForkChatRow,
  pickAndImportChats,
  useChatNavigationStore,
  useChatRuntimeStore,
  useForkInFlight,
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
  FolderAddIcon,
  Folder02Icon,
  FolderExportIcon,
  LayerIcon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  Settings02Icon,
  Upload01Icon,
  ViewIcon,
  ViewOffSlashIcon,
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
import {
  CARD_ICON_CLASS as FILE_CARD_ICON_CLASS,
  CARD_SURFACE as FILE_CARD_SURFACE,
} from "../components/library-cards";
import { FILE_LIST_COLUMNS } from "../components/library-list";
import { useLibrarySettingsStore } from "../settings-store";
import { SortRadio } from "../components/library-toolbar";
import { CARD_SHADOW, OVERLAY_CONTROL, RAISED_SURFACE } from "../surface";

// No overflow-hidden, so a two-line name grows its row instead of clipping; the row stretches to match.
const CARD = cn(
  RAISED_SURFACE,
  CARD_SHADOW,
  "group/chat relative flex aspect-[8/7] cursor-pointer flex-col gap-2.5 self-stretch rounded-xl px-5 pb-3.5 pt-5 transition hover:bg-neutral-100 hover:shadow-none dark:hover:bg-accent/60",
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
  sectionTime,
} from "./model";

const ICON = "size-icon";
const MENU = "library-actions-menu";
// Fits the window, and each group scrolls past ~7 rows so Sections stays reachable.
const MOVE_TO_MENU =
  "max-h-[var(--radix-dropdown-menu-content-available-height)] overflow-y-auto";
const MOVE_TO_LIST =
  "no-scrollbar -my-0.5 max-h-[calc(260px*var(--ui-space-scale,1))] overflow-y-auto overscroll-contain";
const MENU_LABEL = "px-3 pb-1 pt-2 font-normal text-muted-foreground";

export type ChatExportChoice =
  | { kind: "chat"; format: ConversationExportFormat }
  | { kind: "bulk"; format: ConvExportFormat; merged: boolean };

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
  favoriteSections: ReadonlySet<string>;
  /** Off in Favorites, where every entry is starred. */
  favoriteMarks: boolean;
  /** False in Favorites, where chats do not share the file selection. */
  selectable: boolean;
  models: ReadonlyMap<string, string[]>;
  sections: readonly SidebarCustomSection[];
  sectionOf: ReadonlyMap<string, string>;
  projectSectionOf: ReadonlyMap<string, string>;
  dateField: DateField;
  chatContents: ReadonlyMap<string, ChatContents>;
  selection: ReadonlySet<string>;
  /** `range`: shift-click, from the last toggled row. */
  toggleSelected: (id: string, range?: boolean) => void;
  open: (chat: SidebarItem) => void;
  rename: (chat: SidebarItem) => void;
  togglePin: (chat: SidebarItem) => void;
  setFavorite: (chats: SidebarItem[], favorite: boolean) => void;
  toggleFavoriteProject: (projectId: string) => void;
  toggleFavoriteSection: (sectionId: string) => void;
  fork: (chat: SidebarItem) => void;
  move: (chats: SidebarItem[], destination: ChatDestination) => void;
  /** Moves a project into or out of a section; projects never nest. */
  moveProject: (project: ProjectRecord, destination: ChatDestination) => void;
  /** Narrows the list to one section. Omitted where there are no filters (Favorites). */
  viewSection: (sectionId: string) => void;
  newChatInSection: (sectionId: string) => void;
  /** New project dialog; the project is filed in the section. */
  newProjectInSection: (sectionId: string) => void;
  renameSection: (section: SidebarCustomSection) => void;
  removeSection: (section: SidebarCustomSection) => void;
  exportSection: (section: SidebarCustomSection, choice: ChatExportChoice) => void;
  archive: (chats: SidebarItem[]) => void;
  unarchive: (chats: SidebarItem[]) => void;
  exportChats: (chats: SidebarItem[], choice: ChatExportChoice) => void;
  remove: (chats: SidebarItem[]) => void;
  viewProject: (projectId: string) => void;
  filterProject: (projectId: string) => void;
  newChatIn: (projectId: string | null) => void;
  editProject: (project: ProjectRecord) => void;
  togglePinProject: (projectId: string) => void;
  exportProject: (project: ProjectRecord, choice: ChatExportChoice) => void;
  projectChatCounts: ReadonlyMap<string, number>;
  sectionChatCounts: ReadonlyMap<string, number>;
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
  bulk = false,
  disabled = false,
}: {
  onExport: (choice: ChatExportChoice) => void;
  bulk?: boolean;
  disabled?: boolean;
}) {
  const t = useT();
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger className="gap-2.5" disabled={disabled}>
        <HugeiconsIcon
          icon={Download01Icon}
          strokeWidth={1.75}
          className={ICON}
        />
        {t("common.export")}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className={cn(MENU, bulk ? "w-56" : "w-48")}>
        {bulk ? (
          <BulkExportItems onExport={(format, merged) => onExport({ kind: "bulk", format, merged })} />
        ) : (
          chatExportOptions().map(({ label, format }) => (
            <DropdownMenuItem key={format} onSelect={() => onExport({ kind: "chat", format })}>
              {label}
            </DropdownMenuItem>
          ))
        )}
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
                    icon={Folder02Icon}
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

/** Off while generating or forking, as in the sidebar. Mounts only when the menu opens. */
function ForkItem({ chat }: { chat: SidebarItem }) {
  const t = useT();
  const actions = useChatsActions();
  const generating = useChatRuntimeStore((s) => Boolean(s.runningByThreadId[chat.id]));
  const forking = useForkInFlight((s) => s.forking);
  return (
    <DropdownMenuItem
      disabled={!canForkChatRow(chat) || generating || forking}
      onSelect={() => actions.fork(chat)}
    >
      <GitBranchIcon strokeWidth={1.75} className={ICON} />
      {t("library.chats.menu.fork")}
    </DropdownMenuItem>
  );
}

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
  if (!useChatsActions().favoriteMarks) return null;
  return (
    <HugeiconsIcon
      icon={StarPointedIcon}
      role="img"
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
  const threadIds = chat.threadIds?.length ? chat.threadIds : [chat.id];
  const unread = useChatNavigationStore((s) =>
    threadIds.some((id) => s.unreadThreadIds.has(id)),
  );
  return (
    <Isolate
      className={cn(variant === "card" && "absolute end-2 top-2 z-10")}
    >
      <DropdownMenu>
        <MenuTrigger variant={variant} />
        {/* No Open chat: clicking the chat opens it. */}
        <DropdownMenuContent align="end" className={cn(MENU, "w-52")}>
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
              <MenuItem
                icon={unread ? ViewIcon : ViewOffSlashIcon}
                label={t(
                  unread
                    ? "shell.selection.markRead"
                    : "shell.selection.markUnread",
                )}
                onSelect={() => {
                  const store = useChatNavigationStore.getState();
                  if (unread) store.clearThreadsUnread(threadIds);
                  else
                    store.markThreadsUnread(
                      threadIds,
                      Object.fromEntries(threadIds.map((id) => [id, chat.id])),
                    );
                }}
              />
            </>
          )}
          <DropdownMenuSeparator className="mx-3" />
          {!archived && (
            <>
              {chat.type === "single" && <ForkItem chat={chat} />}
              <MoveSubmenu
                project={chat.projectId ?? null}
                section={actions.sectionOf.get(chat.id) ?? null}
                onMove={(destination) => actions.move([chat], destination)}
              />
            </>
          )}
          <ExportSubmenu onExport={(choice) => actions.exportChats([chat], choice)} />
          {!archived && <OpenChatFolderItem item={chat} />}
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

const TILE =
  "flex size-9 shrink-0 items-center justify-center rounded-[10px] bg-muted text-foreground/70 transition-colors group-hover/chat:bg-primary/10 group-hover/chat:text-primary";
const TILE_ICON = "size-5";

function ChatTile({
  chat,
  className,
}: {
  chat: SidebarItem;
  className?: string;
}) {
  return (
    <div className={cn(TILE, className)}>
      {chat.type === "compare" ? (
        <Columns2Icon strokeWidth={1.75} className={TILE_ICON} />
      ) : chat.isFork ? (
        <GitBranchIcon strokeWidth={1.75} className={TILE_ICON} />
      ) : (
        <HugeiconsIcon
          icon={MessageCircleIcon}
          strokeWidth={1.75}
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

function PinMark() {
  const t = useT();
  return (
    <HugeiconsIcon
      icon={PinIcon}
      role="img"
      aria-label={t("library.chats.badges.pinned")}
      strokeWidth={1.75}
      className="size-3.5 shrink-0 text-muted-foreground"
    />
  );
}

function ChatBadges({ chat, marksOnly = false }: { chat: SidebarItem; marksOnly?: boolean }) {
  const t = useT();
  const { pinned, favorites } = useChatsActions();
  return (
    <>
      {favorites.has(chat.id) && <FavoriteMark />}
      {pinned.has(chat.id) && <PinMark />}
      {!marksOnly && chat.isFork && <Badge>{t("library.chats.badges.fork")}</Badge>}
      {!marksOnly && chat.type === "compare" && (
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

function ChatLocation({
  chat,
  showProject,
  showSection,
  plain = false,
}: {
  chat: SidebarItem;
  showProject: boolean;
  showSection: boolean;
  plain?: boolean;
}) {
  const t = useT();
  const { projectNames, filterProject, sectionOf, sections, viewSection } =
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
  if (plain) {
    return (
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          if (projectId) filterProject(projectId);
          else if (section) viewSection(section.id);
        }}
        // Icon first, so a location is not read as a count.
        className={cn(CHIP, "max-w-full")}
      >
        <HugeiconsIcon
          icon={projectId ? Folder02Icon : LayerIcon}
          strokeWidth={1.75}
          className="size-3.5 shrink-0"
        />
        <span className="truncate">{projectId ? projectName : section?.name}</span>
      </button>
    );
  }
  return (
    <span className="flex min-w-0 items-center gap-3">
      {projectId && (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            filterProject(projectId);
          }}
          className={cn(CHIP, "shrink")}
        >
          <HugeiconsIcon
            icon={Folder02Icon}
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

function FileColumns({ modified }: { modified: number }) {
  const locale = useLocale();
  return (
    <>
      <span className={cn(FILE_LIST_COLUMNS.modified, FILE_LIST_COLUMNS.cell)}>
        {modified ? formatCardTime(modified, locale) : ""}
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

export interface RowTimes {
  bucket?: DateBucket["kind"];
}

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
  return countLabel(
    contents.messages,
    "library.chats.list.oneMessage",
    "library.chats.list.messageCount",
    t,
  );
}

const DATE_LABELS: Record<DateField, TranslationKey> = {
  created: "library.chats.list.created",
  updated: "library.chats.list.lastActive",
  modified: "library.chats.list.lastModified",
};

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
                "group/date flex items-center gap-1 outline-none transition-colors hover:text-foreground focus-visible:text-foreground data-[state=open]:text-foreground",
                active && "text-foreground",
              )}
            >
              {label}
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={2}
                className="size-3 opacity-0 transition-opacity group-hover/date:opacity-100 group-focus-visible/date:opacity-100 group-data-[state=open]/date:opacity-100"
              />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="start"
            className={cn(MENU, "w-44")}
            // Focus returning to the title after a pick left a ring around it.
            onCloseAutoFocus={(event) => event.preventDefault()}
          >
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
        // onClick to read shift.
        onClick={(event) => {
          event.preventDefault();
          toggleSelected(chat.id, event.shiftKey);
        }}
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

const ROW_INSET = "pl-4 pr-6";
const CELL = "truncate text-ui-13 text-muted-foreground";
// Columns follow the list's width, not the window's; the date column hides last.
const DATE_COLUMN = "hidden w-32 shrink-0 @xl:block";
const CONTENTS_COLUMN = "hidden w-36 shrink-0 @3xl:block";
const LOCATION_COLUMN = "hidden w-40 shrink-0 @4xl:block";

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
  const t = useT();
  const active = sort.key === column;
  const Arrow = sort.desc ? ArrowDownIcon : ArrowUpIcon;
  // aria-sort needs a columnheader, so the state goes in the label.
  const direction = t(sort.desc ? "library.toolbar.sortDescending" : "library.toolbar.sortAscending");
  return (
    <button
      type="button"
      onClick={() => onSortChange(column)}
      aria-pressed={active}
      aria-label={active ? `${label}, ${direction}` : undefined}
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
    <div className="pb-2">
      <div
        className={cn(
          "group/chat relative flex items-center gap-4 text-ui-13 text-muted-foreground",
          ROW_INSET,
        )}
      >
        {selectable && (
          <div className="absolute end-full top-1/2 me-3 flex -translate-y-1/2">
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
  locationColumn?: boolean;
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
      onClick={(event) =>
        actions.selection.size > 0
          ? actions.toggleSelected(chat.id, event.shiftKey)
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
        className="absolute end-full top-1/2 me-3 flex -translate-y-1/2"
      />
      <div className="flex min-w-0 flex-1 items-center gap-4 py-2">
        <ChatTile chat={chat} />
        <div className="flex min-w-0 flex-col">
          <span className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                if (actions.selection.size > 0) actions.toggleSelected(chat.id, event.shiftKey);
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
  const location =
    (showProject && chat.projectId) ||
    (showSection && actions.sections.some((entry) => entry.id === actions.sectionOf.get(chat.id))) ? (
      <ChatLocation chat={chat} showProject={showProject} showSection={showSection} plain />
    ) : undefined;
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the title button is the keyboard target
    <div
      onClick={(event) =>
        selecting ? actions.toggleSelected(chat.id, event.shiftKey) : actions.open(chat)
      }
      className={cn(CARD, selected && "ring-2 ring-foreground")}
    >
      <div className="flex items-center gap-2">
        <ChatTile chat={chat} />
        <ChatBadges chat={chat} marksOnly />
      </div>
      <ChatMenu chat={chat} archived={archived} variant="card" />
      <SelectBox chat={chat} visible={selecting} className="absolute bottom-3.5 end-4 flex" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          if (selecting) actions.toggleSelected(chat.id, event.shiftKey);
          else actions.open(chat);
        }}
        className={CARD_TITLE}
      >
        {/* Clamp an inner span: buttons ignore line-clamp. */}
        <span className="line-clamp-2">{chatTitle(chat, t)}</span>
      </button>
      <CardFooter
        meta={location}
        date={formatDate(chatTime(chat, actions.dateField), actions.dateField, times, locale, t)}
        className={selecting ? "pe-7" : undefined}
      />
    </div>
  );
}

function CardFooter({ meta, date, className }: { meta?: ReactNode; date: string; className?: string }) {
  return (
    <div
      className={cn(
        "mt-auto flex min-w-0 flex-col items-start gap-1 text-ui-12 text-muted-foreground",
        className,
      )}
    >
      {meta && <span className="w-full min-w-0 truncate">{meta}</span>}
      <span className="truncate">{date}</span>
    </div>
  );
}

const CARD_TITLE =
  "block w-full rounded text-left font-medium text-ui-14 leading-snug text-foreground outline-none [overflow-wrap:anywhere] focus-visible:ring-2 focus-visible:ring-ring";

/** Group name and count in words; a bare number beside "Today" read as part of the name. */
export function GroupHeading({
  children,
  count,
  countLabel,
}: {
  children: ReactNode;
  count: number;
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

function CollectionTile({ icon }: { icon: IconSvgElement }) {
  return (
    <span className={TILE}>
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={TILE_ICON} />
    </span>
  );
}

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
          className="size-3.5 shrink-0 text-muted-foreground rtl:rotate-180"
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
      className={cn(variant === "card" && "absolute end-2 top-2 z-10")}
    >
      <DropdownMenu>
        <MenuTrigger variant={variant} />
        <DropdownMenuContent align="end" className={cn(MENU, "w-52")}>
          <MenuItem
            icon={PencilEdit02Icon}
            label={t("library.chats.menu.newChatInProject")}
            onSelect={() => actions.newChatIn(project.id)}
          />
          <OpenProjectFolderItem projectId={project.id} />
          <DropdownMenuSeparator className="mx-3" />
          <MenuItem
            icon={Settings02Icon}
            label={t("library.chats.menu.edit")}
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
            bulk
            disabled={!actions.projectChatCounts.get(project.id)}
            onExport={(choice) => actions.exportProject(project, choice)}
          />
          <MenuItem
            icon={Upload01Icon}
            label={t("settings.chat.importChats")}
            onSelect={() => void pickAndImportChats({ projectId: project.id, name: project.name })}
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
      className={CARD}
    >
      <div className="flex items-center gap-2">
        <CollectionTile icon={Folder02Icon} />
        {actions.favoriteProjects.has(project.id) && <FavoriteMark />}
        {pinned && <PinMark />}
      </div>
      <ProjectMenu project={project} variant="card" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          actions.viewProject(project.id);
        }}
        className={CARD_TITLE}
      >
        <span className="line-clamp-2">{project.name}</span>
      </button>
      <CardFooter
        meta={chatCount(stats?.chats ?? 0, t)}
        date={formatDate(projectTime(project, stats, actions.dateField), actions.dateField, {}, locale, t)}
      />
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
                role="img"
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
            {chatCount(stats?.chats ?? 0, t)}
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
      className={cn(variant === "card" && "absolute end-2 top-2 z-10")}
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
      {/* On the page, its New button has both. */}
      {!onPage && (
        <>
          <MenuItem
            icon={PencilEdit02Icon}
            label={t("library.chats.toolbar.newChat")}
            onSelect={() => actions.newChatInSection(section.id)}
          />
          <MenuItem
            icon={FolderAddIcon}
            label={t("library.chats.toolbar.newProject")}
            onSelect={() => actions.newProjectInSection(section.id)}
          />
        </>
      )}
      {/* Edit, as in the sidebar's section menu. */}
      <MenuItem
        icon={Settings02Icon}
        label={t("shell.sections.edit")}
        onSelect={() => actions.renameSection(section)}
      />
      <FavoriteItem
        favorite={actions.favoriteSections.has(section.id)}
        onToggle={() => actions.toggleFavoriteSection(section.id)}
      />
      <ExportSubmenu
        bulk
        disabled={!actions.sectionChatCounts.get(section.id)}
        onExport={(choice) => actions.exportSection(section, choice)}
      />
      {/* Into no project, filed in the section. */}
      <MenuItem
        icon={Upload01Icon}
        label={t("settings.chat.importChats")}
        onSelect={() =>
          void pickAndImportChats({ projectId: null, sectionId: section.id, name: section.name })
        }
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
      className={CARD}
    >
      <div className="flex items-center gap-2">
        <CollectionTile icon={LayerIcon} />
        {actions.favoriteSections.has(section.id) && <FavoriteMark />}
      </div>
      <SectionMenu section={section} variant="card" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          actions.viewSection(section.id);
        }}
        className={CARD_TITLE}
      >
        <span className="line-clamp-2">{section.name}</span>
      </button>
      <CardFooter
        meta={sectionCountLabel(stats, t)}
        date={formatDate(sectionTime(section, stats, actions.dateField), actions.dateField, {}, locale, t)}
      />
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
          {actions.favoriteSections.has(section.id) && <FavoriteMark />}
          {layout === "files" && (
            <CollectionCount>{sectionCountLabel(stats, t)}</CollectionCount>
          )}
        </span>
      </div>
      {layout === "files" ? (
        <FileColumns modified={sectionTime(section, stats, "modified")} />
      ) : (
        <>
          {layout !== "own" && layout.showLocation && (
            <span className={LOCATION_COLUMN} />
          )}
          <span className={cn(CONTENTS_COLUMN, CELL)}>
            {sectionCountLabel(stats, t)}
          </span>
          <span className={cn(DATE_COLUMN, CELL)}>
            {formatDate(
              sectionTime(section, stats, actions.dateField),
              actions.dateField,
              {},
              locale,
              t,
            )}
          </span>
        </>
      )}
      <SectionMenu section={section} variant="row" />
    </div>
  );
}

export function SectionListHeader({ date }: { date: DateColumn }) {
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

function FavoriteTile({
  title,
  icon,
  time,
  onOpen,
  menu,
}: {
  title: string;
  icon: ReactNode;
  time: number;
  onOpen: () => void;
  menu: ReactNode;
}) {
  const locale = useLocale();
  const showTime = useLibrarySettingsStore((s) => s.showCardDates);
  return (
    <div className="group/library-card group/chat relative">
      <button
        type="button"
        aria-label={title}
        onClick={onOpen}
        className={cn(
          "block w-full overflow-hidden rounded-xl text-left outline-none transition focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          FILE_CARD_SURFACE,
        )}
      >
        <div className="flex aspect-square flex-col px-5 pb-3.5 pt-5">
          <p className="line-clamp-2 break-words pr-7 font-medium text-ui-14 leading-snug text-foreground">
            {title}
          </p>
          <div className="flex flex-1 items-center justify-center text-foreground">
            {icon}
          </div>
          <p className="truncate pr-6 text-ui-13 text-muted-foreground">
            {showTime && time ? formatCardTime(time, locale) : ""}
          </p>
        </div>
      </button>
      {menu}
    </div>
  );
}

export function FavoriteChatTile({ chat }: { chat: SidebarItem }) {
  const t = useT();
  const actions = useChatsActions();
  const icon =
    chat.type === "compare" ? (
      <Columns2Icon strokeWidth={1.5} className={FILE_CARD_ICON_CLASS} />
    ) : chat.isFork ? (
      <GitBranchIcon strokeWidth={1.5} className={FILE_CARD_ICON_CLASS} />
    ) : (
      <HugeiconsIcon
        icon={MessageCircleIcon}
        strokeWidth={1.5}
        className={FILE_CARD_ICON_CLASS}
      />
    );
  return (
    <FavoriteTile
      title={chatTitle(chat, t)}
      icon={icon}
      time={chat.updatedAt}
      onOpen={() => actions.open(chat)}
      menu={<ChatMenu chat={chat} archived={false} variant="card" />}
    />
  );
}

export function FavoriteProjectTile({
  project,
  stats,
}: {
  project: ProjectRecord;
  stats?: ProjectStats;
}) {
  const actions = useChatsActions();
  return (
    <FavoriteTile
      title={project.name}
      icon={
        <HugeiconsIcon
          icon={Folder02Icon}
          strokeWidth={1.5}
          className={FILE_CARD_ICON_CLASS}
        />
      }
      time={stats?.lastActive ?? project.updatedAt}
      onOpen={() => actions.viewProject(project.id)}
      menu={<ProjectMenu project={project} variant="card" />}
    />
  );
}

export function FavoriteSectionTile({
  section,
  stats,
}: {
  section: SidebarCustomSection;
  stats?: SectionStats;
}) {
  const actions = useChatsActions();
  return (
    <FavoriteTile
      title={section.name}
      icon={
        <HugeiconsIcon
          icon={LayerIcon}
          strokeWidth={1.5}
          className={FILE_CARD_ICON_CLASS}
        />
      }
      time={stats?.lastActive ?? 0}
      onOpen={() => actions.viewSection(section.id)}
      menu={<SectionMenu section={section} variant="card" />}
    />
  );
}
