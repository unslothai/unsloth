// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import {
  exportConversationByFormat,
  DeleteChatFilesSwitch,
  EditProjectDialog,
  NewProjectDialog,
  type ProjectRecord,
  SectionNameDialog,
  type SidebarCustomSection,
  type SidebarItem,
  archiveChatItems,
  clearNewChatDraft,
  compareModelDisplayName,
  deleteChatItems,
  deleteChatProject,
  exportThreads,
  forkChatRow,
  moveChatItemToProject,
  normalizeSectionName,
  notifyChatHistoryUpdated,
  rangeBetween,
  removeCustomSectionWithUndo,
  renameChatItem,
  showForkCreatedToast,
  unarchiveChatItem,
  useChatPreferencesStore,
  useChatProjects,
  useChatRuntimeStore,
  useChatSidebarItems,
  useForkInFlight,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  useSidebarOrganizationStore,
  useFileProjectInSection,
} from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { MessageCircleIcon, StarPointedIcon } from "@/lib/hugeicons-derived";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Archive03Icon,
  ArchiveRestoreIcon,
  Cancel01Icon,
  CheckmarkSquare02Icon,
  Delete02Icon,
  Folder01Icon,
  FolderAddIcon,
  LayerIcon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactNode,
  type SetStateAction,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { NameDialog } from "../components/library-dialogs";
import { type HeaderTab, LibraryHeader } from "../components/library-header";
import type { LibrarySearch } from "../search";
import { CardGrid } from "../components/library-cards";
import { LIST_ROW_GAP } from "../components/library-list";
import type { LibraryView } from "../settings-store";
import {
  ChatCard,
  ChatListHeader,
  type ChatDestination,
  type ChatExportChoice,
  ChatRow,
  type ChatsActions,
  ChatsActionsProvider,
  CollectionHeader,
  type DateColumn,
  ExportSubmenu,
  FavoriteChatTile,
  FavoriteProjectTile,
  FavoriteSectionTile,
  HEADER_MORE_BUTTON,
  GroupHeading,
  MoveSubmenu,
  ProjectCard,
  ProjectListHeader,
  ProjectRow,
  SectionCard,
  SectionListHeader,
  SectionMenuItems,
  SectionRow,
} from "./chats-items";
import { ChatsToolbar, type SortChoice, SortMenu } from "./chats-toolbar";
import {
  type ChatFilters,
  type ChatGroup,
  type ChatGroupBy,
  type ChatSortKey,
  type ChatsSection,
  type DateBucket,
  DATE_FIELDS,
  type DateField,
  EMPTY_CHAT_FILTERS,
  NO_PROJECT,
  NO_SECTION,
  type ProjectSortKey,
  chatFiltersActive,
  filterChats,
  groupChats,
  matchesTerms,
  mixEntries,
  modelFacets,
  modelsByChat,
  CHATS_SECTIONS,
  type SectionSortKey,
  projectStats,
  searchTerms,
  sectionStats,
  sortChats,
  sortProjects,
  sortSections,
} from "./model";
import { useChatContents } from "./contents";
import { useChatFavoritesStore } from "./favorites-store";
import { useChatsPrefsStore } from "./prefs-store";

const PAGE_SIZE = 150;

const BAR_PILL =
  "flex h-9 items-center gap-2 rounded-full px-4 text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50";
const BAR_ROUND =
  "flex size-9 items-center justify-center rounded-full outline-none transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring";
const BAR_OUTLINE =
  "border border-border transition-colors hover:bg-sidebar-accent dark:border-transparent dark:bg-accent/60 dark:hover:bg-accent";

const CHAT_SORTS: SortChoice<ChatSortKey>[] = [
  { value: "updated", label: "library.chats.list.lastActive" },
  { value: "modified", label: "library.chats.list.lastModified" },
  { value: "created", label: "library.chats.list.created" },
  { value: "name", label: "library.list.name" },
];

const PROJECT_SORTS: SortChoice<ProjectSortKey>[] = [
  { value: "updated", label: "library.chats.list.lastActive" },
  { value: "modified", label: "library.chats.list.lastModified" },
  { value: "created", label: "library.chats.list.created" },
  { value: "name", label: "library.list.name" },
  { value: "chats", label: "library.chats.toolbar.sortChats" },
];

const BUCKET_LABELS: Record<"today" | "yesterday" | "week" | "month", TranslationKey> = {
  today: "library.chats.groups.today",
  yesterday: "library.chats.groups.yesterday",
  week: "library.chats.groups.week",
  month: "library.chats.groups.month",
};

const SECTION_LABELS: Record<ChatsSection, TranslationKey> = {
  all: "library.tabs.all",
  chats: "library.chats.sections.chats",
  projects: "library.chats.sections.projects",
  sections: "shell.sections.sectionsHeading",
  archived: "library.chats.sections.archived",
};

const SECTION_ICONS: Partial<Record<ChatsSection, IconSvgElement>> = {
  chats: MessageCircleIcon,
  projects: Folder01Icon,
  sections: LayerIcon,
  archived: Archive03Icon,
};

const SECTION_SORTS: SortChoice<SectionSortKey>[] = [
  { value: "updated", label: "library.chats.list.lastActive" },
  { value: "modified", label: "library.chats.list.lastModified" },
  { value: "created", label: "library.chats.list.created" },
  { value: "name", label: "library.list.name" },
  { value: "chats", label: "library.chats.toolbar.sortChats" },
];

const NO_CHATS: SidebarItem[] = [];

function isDateField(key: string): key is DateField {
  return (DATE_FIELDS as readonly string[]).includes(key);
}

type HeaderTabs = { items: HeaderTab[]; active: string; onChange: (key: string) => void };

type PendingDelete =
  | { kind: "chats"; chats: SidebarItem[]; deleteFiles: boolean }
  | { kind: "project"; project: ProjectRecord; deleteFiles: boolean };

function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: IconSvgElement;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="mx-auto mt-16 flex max-w-md flex-col items-center gap-2 text-center">
      <HugeiconsIcon icon={icon} strokeWidth={1.5} className="mb-2 size-7" />
      <h2 className="font-medium font-sans text-ui-21 text-foreground">{title}</h2>
      <p className="text-ui-15 text-muted-foreground">{description}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

function LoadingRows() {
  return (
    <div className="mt-6 flex flex-col gap-2">
      {Array.from({ length: 8 }, (_, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: fixed placeholders with no identity
        <Skeleton key={index} className="h-12 rounded-[14px]" />
      ))}
    </div>
  );
}

function errorDescription(err: unknown): string | undefined {
  return err instanceof Error ? err.message : undefined;
}

export interface FavoriteChatEntries {
  rows: ReactNode[];
  cards: { key: string; node: ReactNode }[];
}

/** The Chats tab. `embedded`: starred items only, for Favorites, with no header or filters. */
export function ChatsLibrary({
  search,
  title,
  tabs,
  embedded,
}: {
  search: LibrarySearch;
  title?: ReactNode;
  tabs?: HeaderTabs;
  embedded?: {
    query: string;
    view: LibraryView;
    render: (entries: FavoriteChatEntries) => ReactNode;
  };
}) {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();

  const section: ChatsSection = search.chatSection ? "chats" : (search.chatView ?? "all");
  // Open section page; `section` is the pill. Projects have no page here: they open in Chat.
  const openSectionId = search.chatSection ?? null;

  // Metadata only, like the sidebar. The default reads every message.
  const {
    items,
    archivedItems,
    loaded,
  } = useChatSidebarItems({ requireMessages: false });
  const { projects, hasLoaded: projectsLoaded } = useChatProjects();
  const pinnedIds = usePinnedChatsStore((s) => s.pinnedIds);
  const togglePinned = usePinnedChatsStore((s) => s.togglePin);
  const setPinned = usePinnedChatsStore((s) => s.setPinned);
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const togglePinProject = usePinnedProjectsStore((s) => s.togglePin);
  const unpinProject = usePinnedProjectsStore((s) => s.unpin);
  const alwaysDeleteChatFiles = useChatPreferencesStore((s) => s.alwaysDeleteChatFiles);
  const confirmDeleteChats = useChatPreferencesStore((s) => s.confirmDeleteChats);
  // Shared with the sidebar: filing a chat here files it there.
  const sections = useSidebarOrganizationStore((s) => s.customSections);
  const sectionByChatId = useSidebarOrganizationStore((s) => s.sectionByChatId);
  const setChatsSection = useSidebarOrganizationStore((s) => s.setChatsSection);
  const createCustomSection = useSidebarOrganizationStore((s) => s.createCustomSection);
  const setSectionHidden = useSidebarOrganizationStore((s) => s.setSectionHidden);
  const sectionByProjectId = useSidebarOrganizationStore((s) => s.sectionByProjectId);
  const renameCustomSection = useSidebarOrganizationStore((s) => s.renameCustomSection);
  const setPendingNewChatSection = useSidebarOrganizationStore((s) => s.setPendingNewChatSection);
  const prefs = useChatsPrefsStore();
  const favoriteIds = useChatFavoritesStore((s) => s.chatIds);
  const favoriteProjectIds = useChatFavoritesStore((s) => s.projectIds);
  const setFavoriteChats = useChatFavoritesStore((s) => s.setChats);
  const setFavoriteProjects = useChatFavoritesStore((s) => s.setProjects);
  const favoriteSectionIds = useChatFavoritesStore((s) => s.sectionIds);
  const setFavoriteSections = useChatFavoritesStore((s) => s.setSections);
  const view = embedded ? embedded.view : prefs.view;

  const pinned = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const pinnedProjects = useMemo(() => new Set(pinnedProjectIds), [pinnedProjectIds]);
  const favorites = useMemo(() => new Set(favoriteIds), [favoriteIds]);
  const favoriteProjects = useMemo(() => new Set(favoriteProjectIds), [favoriteProjectIds]);
  const favoriteSections = useMemo(() => new Set(favoriteSectionIds), [favoriteSectionIds]);
  const projectNames = useMemo(() => new Map(projects.map((p) => [p.id, p.name])), [projects]);
  const sectionNames = useMemo(() => new Map(sections.map((s) => [s.id, s.name])), [sections]);
  const currentSection = openSectionId
    ? (sections.find((entry) => entry.id === openSectionId) ?? null)
    : null;
  const sectionOf = useMemo(
    () =>
      new Map(Object.entries(sectionByChatId).filter(([, sectionId]) => sectionNames.has(sectionId))),
    [sectionByChatId, sectionNames],
  );

  const models = useMemo(
    () => modelsByChat([...items, ...archivedItems]),
    [items, archivedItems],
  );

  const [ownQuery, setQuery] = useState("");
  const query = embedded ? embedded.query : ownQuery;
  const listQuery = useDeferredValue(query);
  const [filters, setFilters] = useState<ChatFilters>(EMPTY_CHAT_FILTERS);
  const [selection, setSelectionState] = useState<Set<string>>(new Set());
  const selectionAnchor = useRef<string | null>(null);
  // Clearing drops the shift-click anchor too, or the next range reaches a row no longer selected.
  const setSelection = useCallback((next: SetStateAction<Set<string>>) => {
    if (typeof next !== "function" && next.size === 0) selectionAnchor.current = null;
    setSelectionState(next);
  }, []);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [renaming, setRenaming] = useState<SidebarItem | null>(null);
  const [editing, setEditing] = useState<ProjectRecord | null>(null);
  const [creatingProject, setCreatingProject] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [movingIntoNew, setMovingIntoNew] = useState<{
    kind: "project" | "section";
    chats: SidebarItem[];
    project?: ProjectRecord;
  } | null>(null);
  // Just-created section: its store update lands a render after the page opens.
  const [awaitedSection, setAwaitedSection] = useState<string | null>(null);
  const [renamingSection, setRenamingSection] = useState<SidebarCustomSection | null>(null);
  if (awaitedSection !== null && sections.some((entry) => entry.id === awaitedSection)) {
    setAwaitedSection(null);
  }

  const scope = `${section}:${openSectionId ?? ""}`;
  const [shownScope, setShownScope] = useState(scope);
  if (shownScope !== scope) {
    setShownScope(scope);
    setQuery("");
    setFilters(EMPTY_CHAT_FILTERS);
    setSelection(new Set());
    setVisibleCount(PAGE_SIZE);
  }

  const go = (next: Partial<LibrarySearch>, replace = false) =>
    void navigate({
      to: "/library",
      search: { show: "chats", ...next },
      replace,
    });

  const sectionGone = openSectionId !== null && currentSection === null && openSectionId !== awaitedSection;
  useEffect(() => {
    if (sectionGone) void navigate({ to: "/library", search: { show: "chats", chatView: "sections" }, replace: true });
  }, [sectionGone, navigate]);

  const archived = section === "archived";
  const pool = useMemo(
    () =>
      embedded
        ? items.filter((chat) => favorites.has(chat.id))
        : archived
          ? archivedItems
          : items,
    [embedded, items, archivedItems, archived, favorites],
  );
  const scoped = useMemo(
    () =>
      openSectionId ? pool.filter((chat) => sectionOf.get(chat.id) === openSectionId) : pool,
    [pool, openSectionId, sectionOf],
  );
  const context = useMemo(
    () => ({ pinned, favorites, projectNames, models, sectionOf, sectionNames }),
    [pinned, favorites, projectNames, models, sectionOf, sectionNames],
  );
  const groupOptions: ChatGroupBy[] = [
    "none",
    "date",
    "project",
    ...(sections.length > 0 && !openSectionId ? (["section"] as const) : []),
  ];
  const ungrouped = embedded || section === "all";
  const groupBy = !ungrouped && groupOptions.includes(prefs.groupBy) ? prefs.groupBy : "none";
  // Pinned chats get their own group; floating them within groups broke time order.
  const pinnedFirst = prefs.pinnedFirst && !archived && !embedded;
  const visibleChats = useMemo(() => {
    const matched = filterChats(scoped, listQuery, filters, context);
    return sortChats(matched, prefs.sort, pinned, pinnedFirst, locale);
  }, [scoped, listQuery, filters, context, prefs.sort, pinnedFirst, pinned, locale]);
  const stats = useMemo(
    () => projectStats(projects, items, archivedItems),
    [projects, items, archivedItems],
  );
  const { projectChatCounts, sectionChatCounts } = useMemo(() => {
    const byProject = new Map<string, number>();
    const bySection = new Map<string, number>();
    for (const chat of items) {
      if (chat.projectId) byProject.set(chat.projectId, (byProject.get(chat.projectId) ?? 0) + 1);
      const sectionId = sectionOf.get(chat.id);
      if (sectionId) bySection.set(sectionId, (bySection.get(sectionId) ?? 0) + 1);
    }
    return { projectChatCounts: byProject, sectionChatCounts: bySection };
  }, [items, sectionOf]);
  const visibleProjects = useMemo(() => {
    const terms = searchTerms(query);
    const matched = projects.filter(
      (p) => (!embedded || favoriteProjects.has(p.id)) && matchesTerms(terms, p.name, p.instructions),
    );
    return sortProjects(matched, prefs.projectSort, stats, pinnedProjects, locale);
  }, [projects, query, prefs.projectSort, stats, pinnedProjects, locale, embedded, favoriteProjects]);

  const projectIdSet = useMemo(() => new Set(projects.map((p) => p.id)), [projects]);
  const projectSectionOf = useMemo(
    () => new Map(Object.entries(sectionByProjectId)),
    [sectionByProjectId],
  );
  const sectionStatsById = useMemo(
    () => sectionStats(sections, items, sectionOf, projectSectionOf, projectIdSet),
    [sections, items, sectionOf, projectSectionOf, projectIdSet],
  );
  const visibleSections = useMemo(() => {
    const terms = searchTerms(query);
    const matched = sections.filter(
      (entry) => (!embedded || favoriteSections.has(entry.id)) && matchesTerms(terms, entry.name),
    );
    return sortSections(matched, prefs.sectionSort, sectionStatsById, locale);
  }, [sections, query, prefs.sectionSort, sectionStatsById, locale, embedded, favoriteSections]);
  const mixed = section === "all" && !embedded;
  const allEntries = useMemo(
    () =>
      mixed
        ? mixEntries(visibleChats, visibleProjects, visibleSections, {
            sort: prefs.sort,
            projectStats: stats,
            sectionStats: sectionStatsById,
            pinned,
            pinnedProjects,
            pinnedFirst,
            locale,
          })
        : null,
    [
      mixed,
      visibleChats,
      visibleProjects,
      visibleSections,
      prefs.sort,
      stats,
      sectionStatsById,
      pinned,
      pinnedProjects,
      pinnedFirst,
      locale,
    ],
  );
  const shownEntries = useMemo(
    () => allEntries?.slice(0, visibleCount) ?? null,
    [allEntries, visibleCount],
  );
  const shownChats = useMemo(
    () =>
      shownEntries
        ? shownEntries.flatMap((entry) => (entry.kind === "chat" ? [entry.item] : []))
        : visibleChats.slice(0, visibleCount),
    [shownEntries, visibleChats, visibleCount],
  );
  const groupTime = prefs.dateField;
  const groupingOptions = useMemo(
    () => ({
      time: groupTime,
      oldestFirst: prefs.sort.key !== "name" && !prefs.sort.desc,
      pinned: pinnedFirst && !ungrouped ? pinned : undefined,
      sectionOf,
    }),
    [groupTime, prefs.sort, pinnedFirst, ungrouped, pinned, sectionOf],
  );
  const groups = useMemo(
    () => groupChats(shownChats, groupBy, groupingOptions),
    [shownChats, groupBy, groupingOptions],
  );
  const shownOrder = useMemo(
    () => groups.flatMap((group) => group.items.map((chat) => chat.id)),
    [groups],
  );
  // Headings count all matches, not just the loaded page.
  const groupTotals = useMemo(
    () =>
      groupBy === "none"
        ? new Map<string, number>()
        : new Map(
            groupChats(visibleChats, groupBy, groupingOptions).map((g) => [g.key, g.items.length]),
          ),
    [visibleChats, groupBy, groupingOptions],
  );

  const sectionProjects = useMemo(() => {
    if (!openSectionId) return [];
    const terms = searchTerms(query);
    return sortProjects(
      projects.filter(
        (p) => projectSectionOf.get(p.id) === openSectionId && matchesTerms(terms, p.name, p.instructions),
      ),
      prefs.projectSort,
      stats,
      pinnedProjects,
      locale,
    );
  }, [openSectionId, projects, projectSectionOf, query, prefs.projectSort, stats, pinnedProjects, locale]);

  const allModelFacets = useMemo(() => modelFacets(scoped, models), [scoped, models]);
  const facets = useMemo(
    () => ({
      showProjects: true,
      projects: projects.map((p) => ({ id: p.id, name: p.name })),
      sections: openSectionId ? [] : sections.map((s) => ({ id: s.id, name: s.name })),
      models: allModelFacets
        .filter((facet, index) => index < 20 || filters.models.has(facet.model))
        .map(({ model, count }) => ({ model, count, label: compareModelDisplayName(model) })),
    }),
    [openSectionId, projects, sections, allModelFacets, filters.models],
  );

  // Drop ticked filters whose project, section or model is gone: they can't be unticked.
  if (loaded && projectsLoaded) {
    const liveProjects = [...filters.projects].filter((id) => id === NO_PROJECT || projectNames.has(id));
    const liveSections = [...filters.sections].filter(
      (id) => sections.length > 0 && (id === NO_SECTION || sectionNames.has(id)),
    );
    const liveModels = [...filters.models].filter((model) =>
      allModelFacets.some((f) => f.model === model),
    );
    if (
      liveProjects.length !== filters.projects.size ||
      liveSections.length !== filters.sections.size ||
      liveModels.length !== filters.models.size
    ) {
      setFilters({
        ...filters,
        projects: new Set(liveProjects),
        sections: new Set(liveSections),
        models: new Set(liveModels),
      });
    }
  }

  const visibleIds = useMemo(() => new Set(shownChats.map((chat) => chat.id)), [shownChats]);
  if ([...selection].some((id) => !visibleIds.has(id))) {
    setSelection(new Set([...selection].filter((id) => visibleIds.has(id))));
  }
  const selectedChats = shownChats.filter((chat) => selection.has(chat.id));

  useEffect(() => {
    if (selection.size === 0) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (event.target instanceof HTMLElement && event.target.closest("input, textarea")) return;
      if (document.querySelector('[role="menu"], [role="dialog"], [role="alertdialog"]')) return;
      setSelection(new Set());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selection.size]);

  const activeChatId = () => useChatRuntimeStore.getState().activeThreadId ?? undefined;

  const openProject = (id: string) => {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(id);
    void navigate({ to: "/chat", search: { project: id } });
  };

  const openChat = (chat: SidebarItem) => {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveProjectId(chat.projectId ?? null);
    const project = chat.projectId ? { project: chat.projectId } : {};
    if (chat.type === "compare") {
      runtime.setActiveThreadId(null);
      void navigate({ to: "/chat", search: { compare: chat.id, ...project } });
    } else {
      runtime.setActiveThreadId(chat.id);
      void navigate({ to: "/chat", search: { thread: chat.id, ...project } });
    }
  };

  /** Files the new chat on its first send; the sidebar watches for the mark set here. */
  const newChatInSection = (sectionId: string) => {
    const nonce = crypto.randomUUID();
    clearNewChatDraft();
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(null);
    runtime.setIncognito(false);
    setPendingNewChatSection(null);
    void navigate({ to: "/chat", search: { new: nonce } }).then(() =>
      setPendingNewChatSection({ sectionId, nonce }),
    );
  };

  const newChatIn = (id: string | null) => {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(id);
    void navigate({
      to: "/chat",
      search: id ? { project: id } : { new: crypto.randomUUID() },
    });
  };

  async function run(action: () => Promise<unknown>, success: string | null, failure: string) {
    try {
      await action();
      if (success) toast.success(success);
    } catch (err) {
      toast.error(failure, { description: errorDescription(err) });
    }
  }

  const moveToProject = (chats: SidebarItem[], destination: string | null, name?: string) => {
    setSelection(new Set());
    void run(
      async () => {
        await Promise.all(chats.map((chat) => moveChatItemToProject(chat, destination)));
        notifyChatHistoryUpdated();
      },
      destination
        ? t("settings.data.library.movedChatsToProject", {
            count: chats.length,
            project:
              name ?? projectNames.get(destination) ?? t("settings.data.library.unavailableProject"),
          })
        : t("settings.data.library.movedChatsToRecents", { count: chats.length }),
      t("settings.data.library.moveFailed"),
    );
  };

  /** null unfiles. Filing unpins (Pinned would still show it) and unhides the section. */
  const fileInSection = (chats: SidebarItem[], sectionId: string | null, name?: string) => {
    const ids = chats.map((chat) => chat.id);
    setSelection(new Set());
    setChatsSection(ids, sectionId);
    if (sectionId) {
      setPinned(
        ids.filter((id) => pinned.has(id)),
        false,
      );
      setSectionHidden(sectionId, false);
    }
    toast.success(
      sectionId
        ? t("settings.data.library.movedChatsToProject", {
            count: chats.length,
            project: name ?? sectionNames.get(sectionId) ?? "",
          })
        : t("library.chats.toast.removedFromSection", { count: chats.length }),
    );
  };

  const moveChats = (chats: SidebarItem[], destination: ChatDestination) => {
    if (destination.kind === "project") moveToProject(chats, destination.id);
    else if (destination.kind === "section") fileInSection(chats, destination.id);
    else if (destination.kind === "newProject") {
      setMovingIntoNew({ kind: "project", chats });
      setCreatingProject(true);
    } else setMovingIntoNew({ kind: "section", chats });
  };

  const fileProjectInSection = useFileProjectInSection();

  const moveProject = (project: ProjectRecord, destination: ChatDestination) => {
    if (destination.kind === "section") fileProjectInSection(project, destination.id);
    else if (destination.kind === "newSection") setMovingIntoNew({ kind: "section", chats: [], project });
  };

  async function eachChat(
    chats: SidebarItem[],
    act: (chat: SidebarItem) => Promise<unknown>,
    success: (count: number) => string,
    failure: string,
  ) {
    let done = 0;
    let error: unknown;
    for (const chat of chats) {
      try {
        await act(chat);
        done += 1;
      } catch (err) {
        error = err;
      }
    }
    if (done > 0) toast.success(success(done));
    if (done < chats.length) toast.error(failure, { description: errorDescription(error) });
  }

  const archiveChats = (chats: SidebarItem[]) => {
    setSelection(new Set());
    void eachChat(
      chats,
      (chat) => archiveChatItems([chat], activeChatId(), () => {}),
      (count) =>
        count === 1
          ? t("settings.data.archivedOneChat")
          : t("settings.data.archivedChatCount", { count }),
      t("settings.data.failedToArchiveChats"),
    );
  };

  const unarchiveChats = (chats: SidebarItem[]) => {
    setSelection(new Set());
    void eachChat(
      chats,
      unarchiveChatItem,
      (count) => t("settings.data.library.restoredChats", { count }),
      t("settings.data.library.restoreFailed"),
    );
  };

  const setChatsPinned = (chats: SidebarItem[], next: boolean) => {
    setPinned(
      chats.map((chat) => chat.id),
      next,
    );
    setSelection(new Set());
    toast.success(
      t(next ? "settings.data.library.pinnedChats" : "settings.data.library.unpinnedChats", {
        count: chats.length,
      }),
    );
  };

  const exportChats = async (chats: SidebarItem[], choice: ChatExportChoice, name?: string) => {
    const threadIds = [...new Set(chats.flatMap((chat) => chat.threadIds ?? [chat.id]))];
    try {
      if (choice.kind === "chat") {
        for (const id of threadIds) await exportConversationByFormat(id, choice.format);
      } else {
        const stem = name ?? (chats.length === 1 ? (chats[0]?.title ?? "chats") : "chats");
        await exportThreads(threadIds, choice.format, choice.merged, stem);
      }
    } catch (err) {
      if (!isDownloadCancelled(err)) toast.error(t("settings.data.exportFailed"));
    }
  };

  const forkChat = async (chat: SidebarItem) => {
    const inFlight = useForkInFlight.getState();
    if (inFlight.forking) return;
    inFlight.setForking(true);
    try {
      const result = await forkChatRow(chat);
      notifyChatHistoryUpdated();
      showForkCreatedToast(result.containerSnapshotWarning);
    } catch (err) {
      if ((err as { unslothForkRefused?: boolean } | null)?.unslothForkRefused) {
        toast.info(err instanceof Error ? err.message : t("library.chats.toast.forkFailed"));
      } else {
        toast.error(t("library.chats.toast.forkFailed"), { description: errorDescription(err) });
      }
    } finally {
      inFlight.setForking(false);
    }
  };

  async function confirmDelete(target: PendingDelete) {
    setPendingDelete(null);
    setSelection(new Set());
    if (target.kind === "chats") {
      await run(
        async () => {
          await deleteChatItems(target.chats, activeChatId(), () => {}, {
            deleteFiles: target.deleteFiles,
          });
          // After the delete: a failed one restores the chats with their marks.
          const ids = target.chats.map((chat) => chat.id);
          setFavoriteChats(ids, false);
          setPinned(ids, false);
        },
        t("settings.data.library.deletedChats", { count: target.chats.length }),
        t("settings.data.library.deleteFailed"),
      );
      return;
    }
    const { project } = target;
    await run(
      async () => {
        await deleteChatProject(project.id, { deleteFiles: target.deleteFiles });
        unpinProject(project.id);
        setFavoriteProjects([project.id], false);
        if (useChatRuntimeStore.getState().activeProjectId === project.id) {
          useChatRuntimeStore.getState().setActiveProjectId(null);
        }
        notifyChatHistoryUpdated();
      },
      t("library.chats.toast.projectDeleted", { name: project.name }),
      t("library.chats.toast.projectDeleteFailed"),
    );
  }

  const listed = view === "list" && !embedded;
  const chatContents = useChatContents(listed ? shownChats : NO_CHATS);

  const dateColumn = <K extends string>(
    sort: { key: K; desc: boolean },
    setSort: (next: { key: DateField; desc: boolean }) => void,
    fields: readonly DateField[],
    field: DateField,
  ): DateColumn => ({
    fields,
    sortKey: sort.key,
    desc: sort.desc,
    onFieldChange: (next) => {
      prefs.set({ dateField: next });
      setSort({ key: next, desc: isDateField(sort.key) ? sort.desc : true });
    },
    onToggle: () =>
      setSort({ key: field, desc: sort.key === field ? !sort.desc : true }),
  });
  const chatDateColumn = dateColumn(
    prefs.sort,
    (sort) => prefs.set({ sort }),
    DATE_FIELDS,
    prefs.dateField,
  );
  const projectDateColumn = dateColumn(
    prefs.projectSort,
    (projectSort) => prefs.set({ projectSort }),
    DATE_FIELDS,
    prefs.dateField,
  );
  const sectionDateColumn = dateColumn(
    prefs.sectionSort,
    (sectionSort) => prefs.set({ sectionSort }),
    DATE_FIELDS,
    prefs.dateField,
  );

  const actions: ChatsActions = {
    projects,
    projectNames,
    pinned,
    pinnedProjects,
    favorites,
    favoriteProjects,
    favoriteSections,
    favoriteMarks: !embedded,
    selectable: !embedded,
    models,
    selection,
    toggleSelected: (id, range = false) => {
      // Anchor on an id: the list can re-sort between clicks.
      const anchor = selectionAnchor.current;
      const ids = range && anchor ? rangeBetween(shownOrder, anchor, id) : [id];
      const on = !selection.has(id);
      setSelection((current) => {
        const next = new Set(current);
        for (const each of ids) {
          if (on) next.add(each);
          else next.delete(each);
        }
        return next;
      });
      selectionAnchor.current = id;
    },
    open: openChat,
    rename: setRenaming,
    togglePin: (chat) => togglePinned(chat.id),
    setFavorite: (chats, favorite) => {
      setFavoriteChats(
        chats.map((chat) => chat.id),
        favorite,
      );
      setSelection(new Set());
    },
    toggleFavoriteProject: (id) => setFavoriteProjects([id], !favoriteProjects.has(id)),
    toggleFavoriteSection: (id) => setFavoriteSections([id], !favoriteSections.has(id)),
    fork: (chat) => void forkChat(chat),
    move: moveChats,
    moveProject,
    archive: archiveChats,
    unarchive: unarchiveChats,
    exportChats: (chats, choice) => void exportChats(chats, choice),
    remove: (chats) => {
      const target = { kind: "chats", chats, deleteFiles: alwaysDeleteChatFiles } as const;
      if (confirmDeleteChats) setPendingDelete(target);
      else void confirmDelete(target);
    },
    viewProject: openProject,
    filterProject: embedded
      ? openProject
      : (id) => setFilters((current) => ({ ...current, projects: new Set([id]) })),
    sections,
    sectionOf,
    projectSectionOf,
    dateField: prefs.dateField,
    chatContents,
    viewSection: (id) => go({ chatSection: id }),
    newChatInSection: (id) => newChatInSection(id),
    renameSection: setRenamingSection,
    removeSection: (entry) => {
      const undo = removeCustomSectionWithUndo(entry);
      toast.success(t("shell.sections.deleted", { name: entry.name }), {
        action: { label: t("shell.sections.undo"), onClick: undo },
      });
      if (openSectionId === entry.id) go({ chatView: "sections" }, true);
    },
    exportSection: (entry, choice) =>
      void exportChats(
        items.filter((chat) => sectionOf.get(chat.id) === entry.id),
        choice,
        `section-${entry.name}`,
      ),
    newChatIn,
    editProject: setEditing,
    togglePinProject,
    projectChatCounts,
    sectionChatCounts,
    exportProject: (project, choice) =>
      void exportChats(
        items.filter((chat) => chat.projectId === project.id),
        choice,
        `project-${project.name}`,
      ),
    // Off by default: the workspace may hold the user's own files.
    deleteProject: (project) => setPendingDelete({ kind: "project", project, deleteFiles: false }),
  };

  const narrowed = Boolean(query.trim()) || chatFiltersActive(filters);
  const showProject =
    groupBy !== "project" && shownChats.some((chat) => chat.projectId);
  const showSection =
    !openSectionId && groupBy !== "section" && shownChats.some((chat) => sectionOf.has(chat.id));
  const allSelected = shownChats.length > 0 && shownChats.every((chat) => selection.has(chat.id));

  function groupLabel(group: ChatGroup<SidebarItem>): string {
    if (group.pinned) return t("library.chats.toolbar.pinned");
    if (group.sectionId !== undefined) {
      return group.sectionId === null
        ? t("library.chats.toolbar.withoutSection")
        : (sectionNames.get(group.sectionId) ?? "");
    }
    if (group.bucket) {
      const bucket = group.bucket;
      if (bucket.kind !== "older") return t(BUCKET_LABELS[bucket.kind]);
      return new Date(bucket.year, bucket.month, 1).toLocaleDateString(locale, {
        month: "long",
        year: "numeric",
      });
    }
    if (group.projectId === null) return t("settings.data.library.noProject");
    return (
      projectNames.get(group.projectId ?? "") ?? t("settings.data.library.unavailableProject")
    );
  }

  function renderChats() {
    if (!loaded) return <LoadingRows />;
    const filedProjects = renderSectionProjects();
    if (visibleChats.length === 0 && filedProjects) return filedProjects;
    if (visibleChats.length === 0) {
      if (narrowed) {
        return (
          <EmptyState
            icon={MessageCircleIcon}
            title={t("library.empty.noMatchesTitle")}
            description={t("library.empty.noMatchesDescription")}
          />
        );
      }
      if (archived) {
        return (
          <EmptyState
            icon={Archive03Icon}
            title={t("library.chats.empty.archivedTitle")}
            description={t("library.chats.empty.archivedDescription")}
          />
        );
      }
      if (openSectionId) {
        return (
          <EmptyState
            icon={LayerIcon}
            title={t("library.chats.empty.sectionTitle")}
            description={t("library.chats.empty.sectionDescription")}
            action={
              <Button
                variant="muted"
                className="rounded-full px-5"
                onClick={() => newChatInSection(openSectionId)}
              >
                {t("library.chats.toolbar.newChat")}
              </Button>
            }
          />
        );
      }
      return (
        <EmptyState
          icon={MessageCircleIcon}
          title={t("library.chats.empty.chatsTitle")}
          description={t("library.chats.empty.chatsDescription")}
          action={
            <Button variant="muted" className="rounded-full px-5" onClick={() => newChatIn(null)}>
              {t("library.chats.toolbar.newChat")}
            </Button>
          }
        />
      );
    }
    return (
      <>
        {filedProjects}
        {filedProjects && (
          <div className="mt-2">
            <GroupHeading count={visibleChats.length}>{t("library.chats.sections.chats")}</GroupHeading>
          </div>
        )}
        {chatListing()}
      </>
    );
  }

  function chatListing(spaced = !embedded) {
    const list = view === "list";
    const collectionLocation =
      shownEntries !== null && visibleProjects.some((project) => projectSectionOf.has(project.id));
    const locationColumn = showProject || showSection || collectionLocation;
    const layout = { showLocation: locationColumn };
    const total = allEntries?.length ?? visibleChats.length;
    const chatRow = (chat: SidebarItem, bucket?: DateBucket["kind"]) =>
      list ? (
        <ChatRow
          key={chat.id}
          chat={chat}
          archived={archived}
          showProject={showProject}
          showSection={showSection}
          locationColumn={locationColumn}
          times={{ bucket }}
        />
      ) : (
        <ChatCard
          key={chat.id}
          chat={chat}
          archived={archived}
          showProject={showProject}
          showSection={showSection}
          times={{ bucket }}
        />
      );
    const entryRow = (entry: NonNullable<typeof shownEntries>[number]) => {
      if (entry.kind === "chat") return chatRow(entry.item);
      if (entry.kind === "project") {
        const key = `project:${entry.item.id}`;
        return list ? (
          <ProjectRow key={key} project={entry.item} stats={stats.get(entry.item.id)} layout={layout} />
        ) : (
          <ProjectCard key={key} project={entry.item} stats={stats.get(entry.item.id)} />
        );
      }
      const key = `section:${entry.item.id}`;
      const sectionStat = sectionStatsById.get(entry.item.id);
      return list ? (
        <SectionRow key={key} section={entry.item} stats={sectionStat} layout={layout} />
      ) : (
        <SectionCard key={key} section={entry.item} stats={sectionStat} />
      );
    };
    return (
      <div className={cn("@container", spaced && "mt-6")}>
        {list && (
          <ChatListHeader
            sort={prefs.sort}
            onSortChange={(key) =>
              prefs.set({
                sort:
                  prefs.sort.key === key
                    ? { key, desc: !prefs.sort.desc }
                    : { key, desc: key !== "name" },
              })
            }
            date={chatDateColumn}
            showLocation={locationColumn}
            allSelected={allSelected}
            selecting={selection.size > 0}
            onToggleAll={() =>
              setSelection(allSelected ? new Set() : new Set(shownChats.map((chat) => chat.id)))
            }
          />
        )}
        {shownEntries ? (
          list ? (
            <div className={cn("mt-1 flex flex-col", LIST_ROW_GAP)}>{shownEntries.map(entryRow)}</div>
          ) : (
            <CardGrid>{shownEntries.map(entryRow)}</CardGrid>
          )
        ) : (
          groups.map((group) => (
            <section key={group.key} aria-label={groupBy === "none" ? undefined : groupLabel(group)}>
              {groupBy !== "none" && (
                <GroupHeading count={groupTotals.get(group.key) ?? group.items.length}>
                  {groupLabel(group)}
                </GroupHeading>
              )}
              {list ? (
                <div className={cn("flex flex-col", LIST_ROW_GAP, groupBy === "none" && "mt-1")}>
                  {group.items.map((chat) => chatRow(chat, group.bucket?.kind))}
                </div>
              ) : (
                <CardGrid>{group.items.map((chat) => chatRow(chat, group.bucket?.kind))}</CardGrid>
              )}
            </section>
          ))
        )}
        {total > visibleCount && (
          <div className="mt-6 flex justify-center">
            <Button
              variant="muted"
              className="rounded-full px-5"
              onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
            >
              {t("library.chats.list.showMore", { count: total - visibleCount })}
            </Button>
          </div>
        )}
      </div>
    );
  }

  function renderSections() {
    if (!loaded) return <LoadingRows />;
    if (visibleSections.length === 0) {
      return query.trim() ? (
        <EmptyState
          icon={LayerIcon}
          title={t("library.empty.noMatchesTitle")}
          description={t("library.empty.noMatchesDescription")}
        />
      ) : (
        <EmptyState
          icon={LayerIcon}
          title={t("library.chats.empty.sectionsTitle")}
          description={t("library.chats.empty.sectionsDescription")}
          action={
            <Button
              variant="dark"
              className="rounded-full px-5"
              onClick={() => setMovingIntoNew({ kind: "section", chats: [] })}
            >
              {t("shell.sections.newSection")}
            </Button>
          }
        />
      );
    }
    return <div className="mt-6">{sectionListing()}</div>;
  }

  function sectionListing() {
    if (view === "list") {
      return (
        <div className="@container">
          <SectionListHeader date={sectionDateColumn} />
          <div className={cn("mt-1 flex flex-col", LIST_ROW_GAP)}>
            {visibleSections.map((entry) => (
              <SectionRow key={entry.id} section={entry} stats={sectionStatsById.get(entry.id)} />
            ))}
          </div>
        </div>
      );
    }
    return (
      <CardGrid>
        <button
          type="button"
          onClick={() => setMovingIntoNew({ kind: "section", chats: [] })}
          className="flex min-h-40 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-ui-14 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
        >
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.5} className="size-6" />
          {t("shell.sections.newSection")}
        </button>
        {visibleSections.map((entry) => (
          <SectionCard key={entry.id} section={entry} stats={sectionStatsById.get(entry.id)} />
        ))}
      </CardGrid>
    );
  }

  function renderAll() {
    if (!loaded || !projectsLoaded) return <LoadingRows />;
    const nothing =
      visibleProjects.length === 0 && visibleSections.length === 0 && visibleChats.length === 0;
    if (nothing) return renderChats();
    return chatListing(true);
  }

  function renderSectionProjects() {
    if (sectionProjects.length === 0) return null;
    return (
      <div className="@container mt-6">
        <GroupHeading
          count={sectionProjects.length}
          countLabel={t(
            sectionProjects.length === 1
              ? "library.chats.project.oneProject"
              : "library.chats.project.projectCount",
            { count: sectionProjects.length },
          )}
        >
          {t("library.chats.sections.projects")}
        </GroupHeading>
        {view === "list" ? (
          <div className={cn("flex flex-col", LIST_ROW_GAP)}>
            {sectionProjects.map((project) => (
              <ProjectRow key={project.id} project={project} stats={stats.get(project.id)} />
            ))}
          </div>
        ) : (
          <CardGrid>
            {sectionProjects.map((project) => (
              <ProjectCard key={project.id} project={project} stats={stats.get(project.id)} />
            ))}
          </CardGrid>
        )}
      </div>
    );
  }

  function renderProjects() {
    if (!projectsLoaded || !loaded) return <LoadingRows />;
    if (visibleProjects.length === 0) {
      return query.trim() ? (
        <EmptyState
          icon={Folder01Icon}
          title={t("library.empty.noMatchesTitle")}
          description={t("library.empty.noMatchesDescription")}
        />
      ) : (
        <EmptyState
          icon={Folder01Icon}
          title={t("library.chats.empty.projectsTitle")}
          description={t("library.chats.empty.projectsDescription")}
          action={
            <Button variant="dark" className="rounded-full px-5" onClick={() => setCreatingProject(true)}>
              {t("library.chats.toolbar.newProject")}
            </Button>
          }
        />
      );
    }
    return <div className="mt-6">{projectListing()}</div>;
  }

  function projectListing() {
    if (view === "list") {
      return (
        <div className="@container">
          <ProjectListHeader date={projectDateColumn} />
          <div className={cn("mt-1 flex flex-col", LIST_ROW_GAP)}>
            {visibleProjects.map((project) => (
              <ProjectRow key={project.id} project={project} stats={stats.get(project.id)} />
            ))}
          </div>
        </div>
      );
    }
    return (
      <CardGrid>
        {!embedded && (
          <button
            type="button"
            onClick={() => setCreatingProject(true)}
            className="flex min-h-44 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-ui-14 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
          >
            <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.5} className="size-6" />
            {t("library.chats.toolbar.newProject")}
          </button>
        )}
        {visibleProjects.map((project) => (
          <ProjectCard key={project.id} project={project} stats={stats.get(project.id)} />
        ))}
      </CardGrid>
    );
  }

  const sectionPills = (
    <nav aria-label={t("library.chats.sections.ariaLabel")} className="flex flex-wrap items-center gap-x-8 gap-y-2 pl-3.5">
      {CHATS_SECTIONS.map((entry) => {
        const active = openSectionId ? entry === "sections" : entry === section;
        return (
          <button
            key={entry}
            type="button"
            aria-current={active ? "page" : undefined}
            onClick={() => go(entry === "all" ? {} : { chatView: entry })}
            className={cn(
              "flex h-8 items-center gap-2 rounded-sm font-heading text-ui-14 text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring",
              active && "font-medium text-foreground",
            )}
          >
            {SECTION_ICONS[entry] && (
              <HugeiconsIcon icon={SECTION_ICONS[entry]} strokeWidth={1.75} className="size-4 shrink-0" />
            )}
            {t(SECTION_LABELS[entry])}
          </button>
        );
      })}
    </nav>
  );

  const sectionHeader = currentSection && (
    <CollectionHeader
      icon={LayerIcon}
      name={currentSection.name}
      parent={t("shell.sections.sectionsHeading")}
      onParent={() => go({ chatView: "sections" })}
      actions={
        <>
          <Button variant="muted" className="rounded-full px-4" onClick={() => newChatInSection(currentSection.id)}>
            <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-4" />
            {t("library.chats.toolbar.newChat")}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button type="button" aria-label={t("library.menu.moreActions")} className={HEADER_MORE_BUTTON}>
                <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="library-actions-menu w-52">
              <SectionMenuItems section={currentSection} onPage />
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      }
    />
  );

  const searchPlaceholder =
    currentSection
      ? t("library.chats.search.project", { project: currentSection.name })
      : t(
          section === "projects"
            ? "library.chats.search.projects"
            : section === "sections"
              ? "library.chats.search.sections"
              : archived
                ? "library.chats.search.archived"
                : "library.chats.search.chats",
        );

  const sortControl =
    section === "sections" ? (
      <SortMenu
        options={SECTION_SORTS}
        value={prefs.sectionSort.key}
        desc={prefs.sectionSort.desc}
        onChange={(key, desc) => prefs.set({ sectionSort: { key, desc } })}
      />
    ) : section === "projects" ? (
      <SortMenu
        options={PROJECT_SORTS}
        value={prefs.projectSort.key}
        desc={prefs.projectSort.desc}
        onChange={(key, desc) =>
          prefs.set({
            projectSort: { key, desc },
            ...(isDateField(key) ? { dateField: key } : {}),
          })
        }
      />
    ) : (
      <SortMenu
        options={CHAT_SORTS}
        value={prefs.sort.key}
        desc={prefs.sort.desc}
        onChange={(key, desc) =>
          prefs.set({ sort: { key, desc }, ...(isDateField(key) ? { dateField: key } : {}) })
        }
        groupBy={section === "all" ? undefined : groupBy}
        groupOptions={groupOptions}
        onGroupByChange={(next) => prefs.set({ groupBy: next })}
        pinnedFirst={archived ? undefined : prefs.pinnedFirst}
        onPinnedFirstChange={(next) => prefs.set({ pinnedFirst: next })}
      />
    );

  const selectedPinned = selectedChats.length > 0 && selectedChats.every((chat) => pinned.has(chat.id));
  const selectedFavorite =
    selectedChats.length > 0 && selectedChats.every((chat) => favorites.has(chat.id));
  const shared = <T,>(of: (chat: SidebarItem) => T): T | undefined => {
    const first = selectedChats[0];
    if (!first) return undefined;
    const value = of(first);
    return selectedChats.every((chat) => of(chat) === value) ? value : undefined;
  };
  const sharedProject = shared((chat) => chat.projectId ?? null);
  const sharedSection = shared((chat) => sectionOf.get(chat.id) ?? null);

  const deleteCopy = (() => {
    if (!pendingDelete) return null;
    if (pendingDelete.kind === "project") {
      return {
        title: t("library.dialog.deleteTitle", { name: pendingDelete.project.name }),
        description: t("library.chats.dialog.deleteProjectDescription", {
          // Archived chats are deleted with the project too.
          count:
            (stats.get(pendingDelete.project.id)?.chats ?? 0) +
            (stats.get(pendingDelete.project.id)?.archived ?? 0),
        }),
        filesLabel: t("library.chats.dialog.deleteProjectFilesLabel"),
        files: pendingDelete.project.rootPath ?? t("library.chats.dialog.deleteProjectFiles"),
      };
    }
    const { chats } = pendingDelete;
    return chats.length === 1
      ? {
          title: t("library.dialog.deleteTitle", {
            name: chats[0]?.title || t("settings.data.library.untitled"),
          }),
          description: t("library.chats.dialog.deleteChatDescription"),
          filesLabel: undefined,
          files: undefined,
        }
      : {
          title: t("settings.data.library.deleteChatsTitle", { count: chats.length }),
          description: t("settings.data.library.deleteChatsWarning", { count: chats.length }),
          filesLabel: undefined,
          files: t("library.chats.dialog.deleteFilesMany"),
        };
  })();

  function favoriteEntries(): FavoriteChatEntries {
    const projectsShown = loaded && projectsLoaded ? visibleProjects : [];
    const sectionsShown = loaded ? visibleSections : [];
    const chatsShown = loaded ? visibleChats : [];
    return {
      rows: [
        ...projectsShown.map((project) => (
          <ProjectRow
            key={`project:${project.id}`}
            project={project}
            stats={stats.get(project.id)}
            layout="files"
          />
        )),
        ...sectionsShown.map((entry) => (
          <SectionRow
            key={`section:${entry.id}`}
            section={entry}
            stats={sectionStatsById.get(entry.id)}
            layout="files"
          />
        )),
        ...chatsShown.map((chat) => (
          <ChatRow
            key={`chat:${chat.id}`}
            chat={chat}
            archived={false}
            showProject={false}
            showSection={false}
            fileColumns
          />
        )),
      ],
      cards: [
        ...projectsShown.map((project) => ({
          key: `project:${project.id}`,
          node: <FavoriteProjectTile project={project} stats={stats.get(project.id)} />,
        })),
        ...sectionsShown.map((entry) => ({
          key: `section:${entry.id}`,
          node: <FavoriteSectionTile section={entry} stats={sectionStatsById.get(entry.id)} />,
        })),
        ...chatsShown.map((chat) => ({
          key: `chat:${chat.id}`,
          node: <FavoriteChatTile chat={chat} />,
        })),
      ],
    };
  }

  return (
    <ChatsActionsProvider value={actions}>
      {embedded ? (
        embedded.render(favoriteEntries())
      ) : (
        <main className="relative mx-auto w-full max-w-[calc(1560px*var(--ui-space-scale,1))] px-6 pb-24 pt-8 font-heading sm:px-10">
          <LibraryHeader
            title={title}
            controls={
              <ChatsToolbar
                filters={section === "projects" || section === "sections" ? undefined : filters}
                onFiltersChange={setFilters}
                facets={facets}
                sort={sortControl}
                view={view}
                onViewChange={(view) => prefs.set({ view })}
                search={query}
                onSearchChange={setQuery}
                searchPlaceholder={searchPlaceholder}
                onNewChat={() => (openSectionId ? newChatInSection(openSectionId) : newChatIn(null))}
                onNewProject={() => setCreatingProject(true)}
                onNewSection={() => setMovingIntoNew({ kind: "section", chats: [] })}
              />
            }
            tabs={tabs ?? null}
          />
          <div className="flex flex-col gap-5 pl-3 font-sans">
            {sectionPills}
            {sectionHeader && <div className="mt-3">{sectionHeader}</div>}
          </div>
          <div className="pl-3 font-sans">
            {section === "all"
              ? renderAll()
              : section === "projects"
                ? renderProjects()
                : section === "sections"
                  ? renderSections()
                  : renderChats()}
          </div>

          {selectedChats.length > 0 && (
            <div
              role="toolbar"
              aria-label={t("shell.selection.countSelected", { count: selectedChats.length })}
              className="fixed bottom-8 left-1/2 z-30 flex font-sans -translate-x-1/2 items-center gap-2 rounded-full bg-sidebar py-2 pl-6 pr-2 text-sidebar-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-[0_8px_28px_-6px_var(--background)]"
            >
              <span aria-live="polite" className="mr-4 whitespace-nowrap text-sm font-medium">
                {t("shell.selection.countSelected", { count: selectedChats.length })}
              </span>
              {archived ? (
                <button
                  type="button"
                  onClick={() => unarchiveChats(selectedChats)}
                  className={cn(BAR_PILL, "bg-foreground text-background transition-opacity hover:opacity-85")}
                >
                  <HugeiconsIcon icon={ArchiveRestoreIcon} strokeWidth={1.75} className="size-4" />
                  {t("settings.data.library.unarchive")}
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => setChatsPinned(selectedChats, !selectedPinned)}
                    className={cn(BAR_PILL, BAR_OUTLINE)}
                  >
                    <HugeiconsIcon icon={selectedPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-4" />
                    {t(selectedPinned ? "settings.data.library.unpin" : "settings.data.library.pin")}
                  </button>
                  <button
                    type="button"
                    onClick={() => archiveChats(selectedChats)}
                    className={cn(BAR_PILL, BAR_OUTLINE)}
                  >
                    <HugeiconsIcon icon={Archive03Icon} strokeWidth={1.75} className="size-4" />
                    {t("settings.data.library.archive")}
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => actions.remove(selectedChats)}
                className={cn(
                  BAR_PILL,
                  "border border-red-500/70 text-red-600 transition-colors hover:bg-red-500/15 dark:text-red-400",
                )}
              >
                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-4" />
                {t("common.delete")}
              </button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    aria-label={t("library.menu.moreActions")}
                    className={cn(BAR_ROUND, "data-[state=open]:bg-sidebar-accent")}
                  >
                    <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="center" side="top" className="library-actions-menu w-52">
                  {selectedChats.length < shownChats.length && (
                    <DropdownMenuItem
                      onSelect={() => setSelection(new Set(shownChats.map((chat) => chat.id)))}
                    >
                      <HugeiconsIcon icon={CheckmarkSquare02Icon} strokeWidth={1.75} className="size-icon" />
                      {t("settings.data.library.selectAll")}
                    </DropdownMenuItem>
                  )}
                  {!archived && (
                    <DropdownMenuItem onSelect={() => actions.setFavorite(selectedChats, !selectedFavorite)}>
                      <HugeiconsIcon
                        icon={StarPointedIcon}
                        strokeWidth={1.75}
                        className={cn("size-icon", selectedFavorite && "[&_path]:fill-current")}
                      />
                      {t(selectedFavorite ? "library.menu.removeFromFavorites" : "library.menu.addToFavorites")}
                    </DropdownMenuItem>
                  )}
                  {!archived && (
                    <MoveSubmenu
                      project={sharedProject}
                      section={sharedSection}
                      onMove={(destination) => moveChats(selectedChats, destination)}
                    />
                  )}
                  <ExportSubmenu
                    bulk={selectedChats.length > 1}
                    onExport={(choice) => void exportChats(selectedChats, choice)}
                  />
                </DropdownMenuContent>
              </DropdownMenu>
              <button
                type="button"
                aria-label={t("library.selection.clear")}
                onClick={() => setSelection(new Set())}
                className={BAR_ROUND}
              >
                <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-5" />
              </button>
            </div>
          )}
        </main>
      )}

      <NameDialog
        open={renaming !== null}
        title={t("library.chats.dialog.renameChat")}
        submitLabel={t("common.rename")}
        initialValue={renaming?.title ?? ""}
        onSubmit={async (name) => {
          if (!renaming) return;
          try {
            await renameChatItem(renaming, name);
            notifyChatHistoryUpdated();
          } catch (err) {
            toast.error(t("library.toast.renameFailed"), { description: errorDescription(err) });
            throw err;
          }
        }}
        onOpenChange={(open) => !open && setRenaming(null)}
      />
      <EditProjectDialog
        project={editing}
        onOpenChange={(open) => !open && setEditing(null)}
        onDelete={(project) => {
          setEditing(null);
          actions.deleteProject(project);
        }}
      />
      <NewProjectDialog
        open={creatingProject}
        onOpenChange={(open) => {
          setCreatingProject(open);
          if (!open && movingIntoNew?.kind === "project") setMovingIntoNew(null);
        }}
        title={t("library.chats.toolbar.newProject")}
        submitLabel={t("library.dialog.create")}
        onCreated={(project) => {
          if (movingIntoNew?.kind === "project") {
            moveToProject(movingIntoNew.chats, project.id, project.name);
            setMovingIntoNew(null);
            return;
          }
          openProject(project.id);
        }}
      />
      <SectionNameDialog
        open={movingIntoNew?.kind === "section"}
        mode="create"
        onOpenChange={(open) => !open && setMovingIntoNew(null)}
        onSubmit={(name) => {
          const sectionId = createCustomSection(name);
          if (!sectionId || !movingIntoNew) return;
          if (movingIntoNew.project) {
            fileProjectInSection(movingIntoNew.project, sectionId, normalizeSectionName(name));
            return;
          }
          if (movingIntoNew.chats.length === 0) {
            setAwaitedSection(sectionId);
            go({ chatSection: sectionId });
            return;
          }
          fileInSection(movingIntoNew.chats, sectionId, normalizeSectionName(name));
        }}
      />
      <SectionNameDialog
        open={renamingSection !== null}
        mode="rename"
        initialName={renamingSection?.name ?? ""}
        onOpenChange={(open) => !open && setRenamingSection(null)}
        onSubmit={(name) => renamingSection && renameCustomSection(renamingSection.id, name)}
      />
      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="break-words">{deleteCopy?.title}</AlertDialogTitle>
            <AlertDialogDescription>{deleteCopy?.description}</AlertDialogDescription>
          </AlertDialogHeader>
          {pendingDelete && (
            <DeleteChatFilesSwitch
              id="library-chats-delete-files"
              checked={pendingDelete.deleteFiles}
              onCheckedChange={(deleteFiles) => setPendingDelete({ ...pendingDelete, deleteFiles })}
              label={deleteCopy?.filesLabel}
              description={deleteCopy?.files}
            />
          )}
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => pendingDelete && void confirmDelete(pendingDelete)}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ChatsActionsProvider>
  );
}
