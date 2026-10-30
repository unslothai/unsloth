// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { isDownloadCancelled, pickNativeChatImport } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  chatExportOptions,
  DeleteChatFilesSwitch,
  OpenChatFolderItem,
  archiveChatItem,
  deleteChatItem,
  deleteChatProject,
  exportConversationByFormat,
  getSidebarItemThreadIds,
  notifyChatHistoryUpdated,
  renameChatItem,
  useChatNavigationStore,
  useChatPreferencesStore,
  useChatProjects,
  projectActivityAt,
  useChatRuntimeStore,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  type ConversationExportFormat,
  type ProjectRecord,
} from "@/features/chat";
import { useSettingsDialogStore } from "@/features/settings";
import { formatRelativeTime, useLocale, useT, type Locale, type TranslationKey } from "@/i18n";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildProjectsTourSteps } from "./tour";
import { EditProjectDialog } from "./components/edit-project-dialog";
import { NewProjectDialog } from "./components/new-project-dialog";
import {
  Archive03Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  Search01Icon,
  Upload01Icon,
  ViewIcon,
  ViewOffSlashIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { Folder02Icon } from "@/lib/folder-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ProjectIcon } from "@/lib/project-icon";
import { ArrowDownIcon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  COMBINED_EXPORT_FORMATS_LIST,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  EXPORT_FORMATS_LIST,
  type ConvExportFormat,
} from "./prompt-storage/prompt-storage-dialog";
import {
  fileImportSource,
  nativeImportSource,
  type ImportSource,
} from "./utils/chat-import";
import {
  listStoredChatThreads,
} from "./utils/chat-history-storage";
import { CHAT_HISTORY_UPDATED_EVENT } from "./api/chat-api";
import { createHistoryRefreshDebounce } from "./utils/chat-history-revision";
import { groupThreads, type SidebarItem } from "./hooks/use-chat-sidebar-items";
import { ProjectMenuItems } from "./components/project-menu-items";
import { SectionNameDialog } from "./components/section-name-dialog";
import { useFileProjectInSection } from "./hooks/use-file-project-in-section";
import {
  normalizeSectionName,
  useSidebarOrganizationStore,
} from "./stores/sidebar-organization-store";
import { runChatImport } from "./utils/import-chats";

const PROJECTS_PAGE_STEP = 12;
// Streaming fires the history event per chunk, so debounce reloads.
const PROJECT_CHATS_REFRESH_DEBOUNCE_MS = 300;
// Chunk saves are spaced wider than that on a slow CPU model, so they wait for a longer lull.
const PROJECT_CHATS_STREAM_REFRESH_DEBOUNCE_MS = 2000;
// Visible count before the fit-to-height measurement runs.
const PROJECTS_INITIAL_FALLBACK = 8;
const PROJECTS_ROW_HEIGHT = 68;

function formatUpdated(
  ts: number,
  t: (key: TranslationKey) => string,
  locale: Locale,
): string {
  if (!Number.isFinite(ts)) return "";
  const now = new Date();
  const then = new Date(ts);
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
  ).getTime();
  const startOfThen = new Date(
    then.getFullYear(),
    then.getMonth(),
    then.getDate(),
  ).getTime();
  const dayDiff = Math.round((startOfToday - startOfThen) / 86_400_000);
  if (dayDiff <= 0) return t("projectsPage.today");
  if (dayDiff === 1) return t("projectsPage.yesterday");
  if (dayDiff < 7) return formatRelativeTime(locale, -dayDiff, "day");
  return then.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

export function ProjectsPage({
  embedded = false,
  ordered,
  searching = false,
  orderedDesc,
  onToggleOrder,
}: {
  embedded?: boolean;
  /** The Library's filtered, sorted projects; its toolbar owns search and sort. */
  ordered?: ProjectRecord[];
  searching?: boolean;
  orderedDesc?: boolean;
  onToggleOrder?: () => void;
} = {}) {
  const t = useT();
  const locale = useLocale();
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const { projects, hasLoaded } = useChatProjects();

  const [query, setQuery] = useState("");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [baseFit, setBaseFit] = useState(PROJECTS_INITIAL_FALLBACK);
  const [extraCount, setExtraCount] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const reloadReadySent = useRef(false);
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const togglePinProject = usePinnedProjectsStore((s) => s.togglePin);
  const pinnedProjectIdSet = useMemo(
    () => new Set(pinnedProjectIds),
    [pinnedProjectIds],
  );
  const unreadThreadIds = useChatNavigationStore((s) => s.unreadThreadIds);
  const markThreadsUnread = useChatNavigationStore((s) => s.markThreadsUnread);
  const clearThreadsUnread = useChatNavigationStore((s) => s.clearThreadsUnread);
  const confirmDeleteChats = useChatPreferencesStore((s) => s.confirmDeleteChats);
  const alwaysDeleteChatFiles = useChatPreferencesStore(
    (s) => s.alwaysDeleteChatFiles,
  );
  const pinnedChatIds = usePinnedChatsStore((s) => s.pinnedIds);
  const togglePinChat = usePinnedChatsStore((s) => s.togglePin);
  const unpinChat = usePinnedChatsStore((s) => s.unpin);
  const pinnedChatIdSet = useMemo(() => new Set(pinnedChatIds), [pinnedChatIds]);

  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<ProjectRecord | null>(null);
  const [sectionFor, setSectionFor] = useState<ProjectRecord | null>(null);
  const createCustomSection = useSidebarOrganizationStore((s) => s.createCustomSection);
  const fileProjectInSection = useFileProjectInSection();
  const [deleting, setDeleting] = useState<ProjectRecord | null>(null);
  const [renamingChat, setRenamingChat] = useState<SidebarItem | null>(null);
  const [chatNameDraft, setChatNameDraft] = useState("");
  const [deletingChat, setDeletingChat] = useState<SidebarItem | null>(null);
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);

  const globalImportRef = useRef<HTMLInputElement>(null);
  const [importFile, setImportFile] = useState<ImportSource | null>(null);
  // A second pick mid-import would interleave two streams into one history.
  const [importing, setImporting] = useState(false);
  // null = Recents
  const [importTargetId, setImportTargetId] = useState<string | null>(null);
  const [openProjectIds, setOpenProjectIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // "error" is a failed first load; reopening retries rather than trusting it.
  const [projectChats, setProjectChats] = useState<
    Record<string, SidebarItem[] | "loading" | "error">
  >({});
  const [staleProjectIds, setStaleProjectIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const setStale = useCallback((projectId: string, stale: boolean) => {
    setStaleProjectIds((prev) => {
      if (prev.has(projectId) === stale) return prev;
      const next = new Set(prev);
      if (stale) next.add(projectId);
      else next.delete(projectId);
      return next;
    });
  }, []);

  // Per-project sequence so an overtaken response cannot resurrect moved chats.
  const loadSeqRef = useRef(new Map<string, number>());
  const loadProjectChats = useCallback((projectId: string, silent = false) => {
    const seq = (loadSeqRef.current.get(projectId) ?? 0) + 1;
    loadSeqRef.current.set(projectId, seq);
    if (!silent) {
      setProjectChats((prev) => ({ ...prev, [projectId]: "loading" }));
    }
    void listStoredChatThreads({ projectId, includeArchived: false })
      .then((threads) => {
        if (loadSeqRef.current.get(projectId) !== seq) return;
        setStale(projectId, false);
        setProjectChats((prev) => ({
          ...prev,
          [projectId]: groupThreads(threads).sort(
            (a, b) => b.updatedAt - a.updatedAt,
          ),
        }));
      })
      .catch(() => {
        if (loadSeqRef.current.get(projectId) !== seq) return;
        setProjectChats((prev) => {
          const rows = prev[projectId];
          if (silent && Array.isArray(rows) && rows.length > 0) {
            setStale(projectId, true);
            return prev;
          }
          return { ...prev, [projectId]: "error" };
        });
      });
  }, [setStale]);

  const openProjectIdsRef = useRef(openProjectIds);
  useEffect(() => {
    openProjectIdsRef.current = openProjectIds;
  }, [openProjectIds]);
  useEffect(() => {
    const debounce = createHistoryRefreshDebounce(
      () => {
        const open = openProjectIdsRef.current;
        // A closed project's load may still be in flight; it must not refill the cache.
        for (const [id, seq] of loadSeqRef.current) {
          if (!open.has(id)) loadSeqRef.current.set(id, seq + 1);
        }
        setProjectChats((prev) => {
          const kept: typeof prev = {};
          for (const id of open) if (prev[id] !== undefined) kept[id] = prev[id];
          return kept;
        });
        for (const id of open) loadProjectChats(id, true);
      },
      PROJECT_CHATS_REFRESH_DEBOUNCE_MS,
      PROJECT_CHATS_STREAM_REFRESH_DEBOUNCE_MS,
    );
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, debounce.onEvent);
    return () => {
      debounce.cancel();
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, debounce.onEvent);
    };
  }, [loadProjectChats]);

  async function handleImport(source: ImportSource, projectId: string | null) {
    setImporting(true);
    try {
      await runChatImport(source, {
        projectId,
        name: projectId
          ? (projects.find((p) => p.id === projectId)?.name ?? t("projectsPage.projectFallback"))
          : undefined,
      });
    } finally {
      setImporting(false);
    }
  }

  async function selectGlobalImportFile() {
    if (importing) return;
    if (!isTauri) {
      globalImportRef.current?.click();
      return;
    }
    try {
      const selected = await pickNativeChatImport();
      if (!selected) return;
      setImportTargetId(null);
      setImportFile(nativeImportSource(selected));
    } catch (error) {
      toast.error(t("projectsPage.importFailed"), {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async function commitImport() {
    if (!importFile) return;
    const file = importFile;
    const target = importTargetId;
    setImportFile(null);
    await handleImport(file, target);
  }

  const sortedProjects = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    const filtered = trimmed
      ? projects.filter((p) => p.name.toLowerCase().includes(trimmed))
      : projects.slice();
    // By last use, as the sidebar orders them: a project's updatedAt only moves on an edit.
    filtered.sort((a, b) =>
      sortDir === "asc"
        ? projectActivityAt(a) - projectActivityAt(b)
        : projectActivityAt(b) - projectActivityAt(a),
    );
    return filtered;
  }, [projects, query, sortDir]);
  const listedProjects = ordered ?? sortedProjects;
  const isSearching = ordered ? searching : query.trim() !== "";
  const visibleCount = baseFit + extraCount;
  const visibleProjects = isSearching
    ? listedProjects
    : listedProjects.slice(0, visibleCount);
  const hasMore = !isSearching && listedProjects.length > visibleCount;

  const tourSteps = useMemo(
    () => buildProjectsTourSteps({ hasProjects: visibleProjects.length > 0 }),
    [visibleProjects.length],
  );
  const tour = useGuidedTourController({ id: "projects", steps: tourSteps });

  useEffect(() => {
    if (!hasLoaded || reloadReadySent.current) {
      return;
    }
    reloadReadySent.current = true;
    signalReady();
  }, [hasLoaded, signalReady]);

  useEffect(() => {
    function measure() {
      const el = listRef.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top;
      const reserve = 24;
      const fits = Math.floor(
        (window.innerHeight - top - reserve) / PROJECTS_ROW_HEIGHT,
      );
      setBaseFit(Math.max(PROJECTS_PAGE_STEP, fits));
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [hasLoaded]);

  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setExtraCount((n) => n + PROJECTS_PAGE_STEP);
        }
      },
      { rootMargin: "300px" },
    );
    io.observe(el);
    return () => io.disconnect();
    // IntersectionObserver does not re-fire on a steady intersection, so re-observe.
  }, [hasMore, visibleCount]);

  function toggleProjectChats(projectId: string) {
    const opening = !openProjectIds.has(projectId);
    setOpenProjectIds((prev) => {
      const next = new Set(prev);
      if (opening) next.add(projectId);
      else next.delete(projectId);
      return next;
    });
    if (!opening) return;
    const cached = projectChats[projectId];
    const loaded = cached !== undefined && cached !== "error";
    if (loaded && !staleProjectIds.has(projectId)) return;
    // Stale rows stay on screen while the retry runs.
    loadProjectChats(projectId, loaded);
  }

  function openChat(item: SidebarItem, projectId: string) {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveProjectId(projectId);
    if (item.type === "compare") {
      runtime.setActiveThreadId(null);
      navigate({ to: "/chat", search: { compare: item.id, project: projectId } });
      return;
    }
    runtime.setActiveThreadId(item.id);
    navigate({ to: "/chat", search: { thread: item.id, project: projectId } });
  }

  function openProject(projectId: string) {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(projectId);
    navigate({ to: "/chat", search: { project: projectId } });
  }

  // A saved chat, as the sidebar's New chat.
  function newChatInProject(projectId: string) {
    useChatRuntimeStore.getState().setIncognito(false);
    openProject(projectId);
  }

  async function handleBulkProjectExport(
    scope: "projects" | "all",
    fmt: ConvExportFormat,
    merged: boolean,
  ) {
    try {
      const listed = await listStoredChatThreads({ includeArchived: false });
      const projectIds = new Set(projects.map((p) => p.id));
      const threads =
        scope === "projects"
          ? listed.filter((t) => t.projectId && projectIds.has(t.projectId))
          : listed;
      const ids = [...new Set(threads.map((t) => t.id))];
      if (ids.length === 0) { toast.info(t("projectsPage.nothingToExport")); return; }
      const ts = new Date().toISOString().slice(0, 10);
      const basename = `${scope === "all" ? "all-chats" : "all-projects"}-${ts}`;
      if (merged) {
        await exportBulkConversationsMerged(ids, fmt, basename);
      } else {
        await exportBulkConversationsSeparate(ids, fmt, basename);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("projectsPage.exportFailed"));
      }
    }
  }

  async function commitChatRename() {
    const target = renamingChat;
    const name = chatNameDraft.trim();
    setRenamingChat(null);
    if (!target || !name || name === target.title) return;
    try {
      await renameChatItem(target, name);
      notifyChatHistoryUpdated();
    } catch (err) {
      toast.error(t("projectsPage.renameChatFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  function activeThreadId(): string | undefined {
    return useChatRuntimeStore.getState().activeThreadId ?? undefined;
  }

  async function archiveChat(chat: SidebarItem) {
    try {
      await archiveChatItem(chat, activeThreadId(), () => {});
    } catch (err) {
      toast.error(t("projectsPage.archiveChatFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function deleteChat(chat: SidebarItem, deleteFiles: boolean) {
    try {
      await deleteChatItem(chat, activeThreadId(), () => {}, { deleteFiles });
      // Pins are stored apart from chats, so remove the id explicitly.
      unpinChat(chat.id);
    } catch (err) {
      toast.error(t("projectsPage.deleteChatFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  /** Always reset here, or a leftover switch deletes files this delete never asked for. */
  function openChatDelete(chat: SidebarItem) {
    setDeleteFilesOnDelete(alwaysDeleteChatFiles);
    setDeletingChat(chat);
  }

  function openProjectDelete(project: ProjectRecord) {
    setDeleteFilesOnDelete(false);
    setDeleting(project);
  }

  async function commitChatDelete() {
    const target = deletingChat;
    if (!target) return;
    const deleteFiles = deleteFilesOnDelete;
    setDeletingChat(null);
    setDeleteFilesOnDelete(false);
    await deleteChat(target, deleteFiles);
  }

  async function handleChatExport(
    chat: SidebarItem,
    format: ConversationExportFormat,
  ) {
    try {
      for (const id of getSidebarItemThreadIds(chat)) {
        await exportConversationByFormat(id, format);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("projectsPage.exportFailed"));
      }
    }
  }

  async function commitDelete() {
    const target = deleting;
    if (!target) return;
    const deleteFiles = deleteFilesOnDelete;
    setDeleting(null);
    setDeleteFilesOnDelete(false);
    try {
      await deleteChatProject(target.id, { deleteFiles });
      // As the sidebar's delete: its chats went with it, and the runtime must not reopen it.
      notifyChatHistoryUpdated();
      const runtime = useChatRuntimeStore.getState();
      if (runtime.activeProjectId === target.id) runtime.setActiveProjectId(null);
    } catch (err) {
      toast.error(t("projectsPage.deleteProjectFailed"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  // Split around {name} so the name can be emphasised.
  const deleteChatText = t("projectsPage.deleteChatDescription").split("{name}");
  const deleteProjectText = t("projectsPage.deleteProjectDescription").split("{name}");
  const importText = t("projectsPage.importDestination").split("{name}");

  const Shell = embedded ? "div" : "main";
  return (
    <Shell
      className={cn(
        "font-heading",
        !embedded && "mx-auto w-full max-w-5xl 4xl:max-w-6xl px-6 pb-10 pt-8 max-sm:px-4 sm:px-10",
      )}
    >
      {!embedded && <GuidedTour {...tour.tourProps} />}
      <input
        ref={globalImportRef}
        type="file"
        accept=".json,.jsonl,.ndjson,.csv"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            setImportTargetId(null);
            setImportFile(fileImportSource(file));
          }
          e.target.value = "";
        }}
      />
      {!embedded && (
      <div className="flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            {t("projectsPage.title")}
          </h1>
        <div className="flex w-full min-w-0 items-center justify-end gap-3 sm:w-auto sm:flex-1">
          <div className="relative min-w-0 flex-1 sm:max-w-md">
            <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
              <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4" />
            </span>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape" && query) {
                  e.preventDefault();
                  setQuery("");
                }
              }}
              placeholder={t("projectsPage.search")}
              className="h-control w-full rounded-full border-none bg-muted pl-10 pr-4 shadow-none dark:bg-card"
              aria-label={t("projectsPage.search")}
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                data-tour="projects-io"
                title={t("projectsPage.importExport")}
                className="size-control rounded-full border-none bg-muted shadow-none dark:bg-card"
              >
                <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem onSelect={() => void selectGlobalImportFile()}>
                <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
                {t("projectsPage.importChatsAction")}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>{t("projectsPage.exportAllProjects")}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-52">
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      {t("projectsPage.combined")}
                    </DropdownMenuLabel>
                    {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`ap-m-${fmt}`} onSelect={() => void handleBulkProjectExport("projects", fmt, true)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator />
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      {t("projectsPage.perChat")}
                    </DropdownMenuLabel>
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`ap-s-${fmt}`} onSelect={() => void handleBulkProjectExport("projects", fmt, false)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>{t("projectsPage.exportProjectsAndRecents")}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-52">
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      {t("projectsPage.combined")}
                    </DropdownMenuLabel>
                    {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`all-m-${fmt}`} onSelect={() => void handleBulkProjectExport("all", fmt, true)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator />
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      {t("projectsPage.perChat")}
                    </DropdownMenuLabel>
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`all-s-${fmt}`} onSelect={() => void handleBulkProjectExport("all", fmt, false)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="dark" data-tour="projects-new" onClick={() => setCreating(true)}>
            <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
            {t("projectsPage.newProject")}
          </Button>
        </div>
      </div>
      )}

      {!hasLoaded ? (
        <div className="mt-16">
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-ui-13 font-medium text-muted-foreground">
            <span className="flex-1">{t("projectsPage.name")}</span>
            <span className="shrink-0 sm:w-40">{t("projectsPage.updated")}</span>
            <span className="size-7 shrink-0" />
            <span className="size-7 shrink-0" />
            <span className="w-8 shrink-0" />
          </div>
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="flex items-center gap-3 rounded-md px-5 py-4"
            >
              <Skeleton className="mr-1 size-9 shrink-0 rounded-md" />
              <Skeleton className="h-4 w-40 rounded-full" />
              <span className="flex-1" />
              <Skeleton className="h-4 w-16 rounded-full" />
              <span className="w-8 shrink-0" />
            </div>
          ))}
        </div>
      ) : visibleProjects.length === 0 ? (
        <div className="mt-16 flex flex-col items-center justify-center gap-2 text-center text-muted-foreground">
          {projects.length === 0 ? (
            <>
              <p className="max-w-sm text-sm">
                {t("projectsPage.emptyDescription")}
              </p>
              <Button
                variant="outline"
                className="mt-2 border-none bg-background shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
                onClick={() => setCreating(true)}
              >
                <HugeiconsIcon icon={ProjectIcon} strokeWidth={1.75} className="size-icon" />
                {t("projectsPage.createFirst")}
              </Button>
            </>
          ) : (
            <>
              <p className="text-sm">{t("projectsPage.noMatches")}</p>
              <Button variant="ghost" size="sm" onClick={() => setQuery("")}>
                {t("projectsPage.clearSearch")}
              </Button>
            </>
          )}
        </div>
      ) : (
        <>
        <div className={embedded ? "mt-6" : "mt-16"}>
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-ui-13 font-medium text-muted-foreground">
            <span className="flex-1">{t("projectsPage.name")}</span>
            <button
              type="button"
              onClick={() =>
                onToggleOrder ? onToggleOrder() : setSortDir((dir) => (dir === "desc" ? "asc" : "desc"))
              }
              title={t((orderedDesc ?? sortDir === "desc") ? "projectsPage.newestFirst" : "projectsPage.oldestFirst")}
              className="flex shrink-0 cursor-pointer items-center gap-1 text-left transition-colors hover:text-foreground sm:w-40"
            >
              {t("projectsPage.updated")}
              <ArrowDownIcon
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  !(orderedDesc ?? sortDir === "desc") && "rotate-180",
                )}
              />
            </button>
            <span className="size-7 shrink-0" />
            <span className="size-7 shrink-0" />
            <span className="w-8 shrink-0" />
          </div>
          <div ref={listRef} data-tour="projects-list">
          {visibleProjects.map((project) => {
            const pinned = pinnedProjectIdSet.has(project.id);
            const chatsOpen = openProjectIds.has(project.id);
            const chats = projectChats[project.id];
            return (
            <div key={`wrap-${project.id}`}>
            <div
              key={project.id}
              className="group/project-row relative flex items-center gap-3 rounded-md px-5 py-4 text-left transition-colors duration-150 hover:bg-muted/70 dark:hover:bg-[rgb(255_255_255_/_calc(0.055*var(--contrast-wash-gain,1)))]"
            >
              <span className="flex min-w-0 flex-1 items-center gap-2">
                {/* A real button holding only text. Giving the whole row the button role made
                    its pin and menu presentational children: one control where there are four. */}
                <button
                  type="button"
                  onClick={() => openProject(project.id)}
                  className="flex min-w-0 cursor-pointer items-center gap-3 rounded-md text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <span className="mr-1 flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-foreground/70 transition-colors group-hover/project-row:bg-primary/10 group-hover/project-row:text-primary">
                    <HugeiconsIcon
                      icon={chatsOpen ? Folder02Icon : ProjectIcon}
                      strokeWidth={1.75}
                      className="size-5"
                    />
                  </span>
                  <span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">
                    {project.name}
                  </span>
                </button>
                <button
                  type="button"
                  aria-label={t(chatsOpen ? "projectsPage.hideChats" : "projectsPage.showChats")}
                  aria-expanded={chatsOpen}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleProjectChats(project.id);
                  }}
                  className={cn(
                    "flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground focus-visible:opacity-100 group-hover/project-row:opacity-100 pointer-coarse:opacity-100 dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
                    chatsOpen ? "opacity-100" : "opacity-0",
                  )}
                >
                  <HugeiconsIcon icon={ChevronDownStandardIcon}
                    strokeWidth={1.75}
                    className={cn(
                      "size-4 transition-transform",
                      !chatsOpen && "-rotate-90",
                    )}
                  />
                </button>
                {/* biome-ignore lint/a11y/useKeyWithClickEvents: the chevron is the keyboard target */}
                {/* biome-ignore lint/a11y/noStaticElementInteractions: a wider mouse target for the chevron */}
                <span
                  aria-hidden="true"
                  onClick={() => toggleProjectChats(project.id)}
                  className="-my-4 min-w-0 flex-1 cursor-pointer self-stretch"
                />
              </span>
              <span className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block">
                {formatUpdated(projectActivityAt(project), t, locale)}
              </span>
              <button
                type="button"
                aria-label={t(pinned ? "projectsPage.unpinProject" : "projectsPage.pinProject")}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePinProject(project.id);
                }}
                className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
              >
                <HugeiconsIcon
                  icon={pinned ? PinOffIcon : PinIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </button>
              {/* As ChatGPT's project rows: a new chat is one click, beside the menu that also has it. */}
              <button
                type="button"
                aria-label={t("projectsPage.newChatInProject")}
                title={t("projectsPage.newChatInProject")}
                onClick={(e) => {
                  e.stopPropagation();
                  newChatInProject(project.id);
                }}
                className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
              >
                <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-4" />
              </button>
              <div className="relative flex w-8 shrink-0 items-center justify-end">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      aria-label={t("projectsPage.projectOptions")}
                      className="absolute right-0 flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground data-[state=open]:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                    >
                      <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    side="bottom"
                    align="end"
                    sideOffset={0}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                    className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-52 font-heading rounded-xl border-0"
                  >
                    <ProjectMenuItems
                      project={project}
                      onNewChat={() => newChatInProject(project.id)}
                      onEdit={() => setEditing(project)}
                      onDelete={() => openProjectDelete(project)}
                      onNewSection={() => setSectionFor(project)}
                    />
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
            {chatsOpen && (
              <div className="mb-2 flex flex-col gap-0.5 pl-[calc(52px*var(--ui-space-scale,1))]">
                {chats === undefined || chats === "loading" ? (
                  <Skeleton className="ml-5 h-6 w-48 rounded-full" />
                ) : chats === "error" ? (
                  <button
                    type="button"
                    onClick={() => loadProjectChats(project.id)}
                    className="cursor-pointer self-start py-1 pl-5 text-left text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                  >
                    {t("projectsPage.loadChatsFailed")}
                  </button>
                ) : chats.length === 0 ? (
                  <p className="py-1 pl-5 text-sm text-muted-foreground">{t("projectsPage.noChats")}</p>
                ) : (
                  <>
                    {chats.map((chat) => {
                      const chatPinned = pinnedChatIdSet.has(chat.id);
                      const chatThreadIds = getSidebarItemThreadIds(chat);
                      const chatUnread = chatThreadIds.some((id) =>
                        unreadThreadIds.has(id),
                      );
                      return (
                      <div
                        key={chat.id}
                        className="group/chat-row flex items-center gap-3 rounded-full py-1.5 pl-5 pr-5 text-left text-sm text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.055*var(--contrast-wash-gain,1)))]"
                      >
                        <button
                          type="button"
                          onClick={() => openChat(chat, project.id)}
                          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-full text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        >
                          <span className="min-w-0 truncate">{chat.title}</span>
                          {chatUnread ? (
                            <span
                              role="img"
                              aria-label={t("projectsPage.unread")}
                              className="size-2 shrink-0 rounded-full bg-primary"
                            />
                          ) : null}
                          <span className="flex-1" />
                          {/* Same widths as a project row, so the columns line up under it, and
                              gone with it below sm, where a nested row has no room for both. */}
                          <span className="hidden w-40 shrink-0 sm:block">
                            {formatUpdated(chat.updatedAt, t, locale)}
                          </span>
                        </button>
                        <button
                          type="button"
                          aria-label={t(chatPinned ? "projectsPage.unpinChat" : "projectsPage.pinChat")}
                          onClick={(e) => {
                            e.stopPropagation();
                            togglePinChat(chat.id);
                          }}
                          className={cn(
                            "flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground focus-visible:opacity-100 group-hover/chat-row:opacity-100 pointer-coarse:opacity-100 dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
                            chatPinned ? "opacity-100" : "opacity-0",
                          )}
                        >
                          <HugeiconsIcon
                            icon={chatPinned ? PinOffIcon : PinIcon}
                            strokeWidth={1.75}
                            className="size-4"
                          />
                        </button>
                        {/* Stands in for the project row's New chat, keeping pin and menu in line. */}
                        <span className="size-7 shrink-0" />
                        <div className="relative flex w-8 shrink-0 items-center justify-end">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <button
                                type="button"
                                onClick={(e) => e.stopPropagation()}
                                aria-label={t("projectsPage.chatOptions")}
                                className="absolute right-0 flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground focus-visible:opacity-100 group-hover/chat-row:opacity-100 pointer-coarse:opacity-100 data-[state=open]:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] data-[state=open]:opacity-100 dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                              >
                                <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
                              </button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent
                              side="bottom"
                              align="end"
                              sideOffset={0}
                              onClick={(e) => e.stopPropagation()}
                              onKeyDown={(e) => e.stopPropagation()}
                              className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-52 font-heading rounded-xl border-0"
                            >
                              <DropdownMenuItem
                                onSelect={() => {
                                  setChatNameDraft(chat.title);
                                  setRenamingChat(chat);
                                }}
                              >
                                <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                                <span>{t("common.rename")}</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onSelect={() =>
                                  chatUnread
                                    ? clearThreadsUnread(chatThreadIds)
                                    : markThreadsUnread(
                                        chatThreadIds,
                                        Object.fromEntries(
                                          chatThreadIds.map((id) => [id, chat.id]),
                                        ),
                                      )
                                }
                              >
                                <HugeiconsIcon icon={chatUnread ? ViewIcon : ViewOffSlashIcon} strokeWidth={1.75} className="size-icon" />
                                <span>
                                  {t(chatUnread ? "shell.selection.markRead" : "shell.selection.markUnread")}
                                </span>
                              </DropdownMenuItem>
                              <DropdownMenuSub>
                                <DropdownMenuSubTrigger>
                                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon mr-1" />
                                  <span>{t("common.export")}</span>
                                </DropdownMenuSubTrigger>
                                <DropdownMenuSubContent className="w-52">
                                  {chatExportOptions().map(({ label, format }) => (
                                    <DropdownMenuItem
                                      key={`${chat.id}-${label}`}
                                      onSelect={(e) => {
                                        e.stopPropagation();
                                        void handleChatExport(chat, format);
                                      }}
                                    >
                                      {label}
                                    </DropdownMenuItem>
                                  ))}
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    onSelect={() =>
                                      useSettingsDialogStore
                                        .getState()
                                        .openDialog("data")
                                    }
                                  >
                                    {t("projectsPage.exportAllChats")}
                                  </DropdownMenuItem>
                                </DropdownMenuSubContent>
                              </DropdownMenuSub>
                              <OpenChatFolderItem item={chat} />
                              <DropdownMenuSeparator />
                              <DropdownMenuItem onSelect={() => void archiveChat(chat)}>
                                <HugeiconsIcon icon={Archive03Icon} strokeWidth={1.75} className="size-icon" />
                                <span>{t("projectsPage.archive")}</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                variant="destructive"
                                onSelect={() =>
                                  confirmDeleteChats
                                    ? openChatDelete(chat)
                                    : void deleteChat(chat, alwaysDeleteChatFiles)
                                }
                              >
                                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                                <span>{t("common.delete")}</span>
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </div>
                      );
                    })}
                    {staleProjectIds.has(project.id) && (
                      <button
                        type="button"
                        onClick={() => loadProjectChats(project.id, true)}
                        className="cursor-pointer self-start py-1 text-left text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                      >
                        {t("projectsPage.refreshChatsFailed")}
                      </button>
                    )}
                  </>
                )}
              </div>
            )}
            </div>
            );
          })}
          {hasMore && <div ref={sentinelRef} className="h-px w-full" />}
          </div>
        </div>
        </>
      )}

      <NewProjectDialog open={creating} onOpenChange={setCreating} />

      <SectionNameDialog
        open={sectionFor !== null}
        mode="create"
        onOpenChange={(open) => !open && setSectionFor(null)}
        onSubmit={(name) => {
          const sectionId = createCustomSection(name);
          if (sectionId && sectionFor) {
            fileProjectInSection(sectionFor, sectionId, normalizeSectionName(name));
          }
        }}
      />
      {/* Edit project (name + instructions + source folders), the sidebar's dialog. */}
      <EditProjectDialog
        project={editing}
        onOpenChange={(open) => {
          if (!open) setEditing(null);
        }}
        onDelete={(project) => openProjectDelete(project)}
      />

      <Dialog
        open={renamingChat !== null}
        onOpenChange={(open) => {
          if (!open) setRenamingChat(null);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("projectsPage.renameChat")}</DialogTitle>
          </DialogHeader>
          <Input
            value={chatNameDraft}
            onChange={(e) => setChatNameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitChatRename();
              }
            }}
            autoFocus
            maxLength={200}
            placeholder={t("projectsPage.chatName")}
            aria-label={t("projectsPage.chatName")}
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setRenamingChat(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              type="button"
              variant="dark"
              onClick={() => void commitChatRename()}
              disabled={
                !chatNameDraft.trim() ||
                chatNameDraft.trim() === renamingChat?.title
              }
            >
              {t("common.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deletingChat !== null}
        onOpenChange={(open) => {
          if (!open) setDeletingChat(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("projectsPage.deleteChat")}</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            {deleteChatText[0]}
            <em>{deletingChat?.title}</em>
            {deleteChatText[1]}
          </p>
          <DeleteChatFilesSwitch
            id="projects-delete-chat-files"
            checked={deleteFilesOnDelete}
            onCheckedChange={setDeleteFilesOnDelete}
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setDeletingChat(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void commitChatDelete()}
            >
              {deleteFilesOnDelete ? t("projectsPage.deleteAll") : t("common.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={importFile !== null} onOpenChange={(open) => { if (!open) setImportFile(null); }}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("projectsPage.importChats")}</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            {importText[0]}
            <span
              data-reload-snapshot-sensitive
              className="font-medium text-foreground"
            >
              {importFile?.name}
            </span>
            {importText[1]}
          </p>
          <Select
            value={importTargetId ?? "__recents__"}
            onValueChange={(v) => setImportTargetId(v === "__recents__" ? null : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder={t("projectsPage.selectDestination")} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__recents__">{t("projectsPage.recents")}</SelectItem>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setImportFile(null)}>{t("common.cancel")}</Button>
            <Button type="button" onClick={() => void commitImport()}>{t("projectsPage.import")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("projectsPage.deleteProject")}</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            {deleteProjectText[0]}
            <em>{deleting?.name}</em>
            {deleteProjectText[1]}
          </p>
          <DeleteChatFilesSwitch
            id="projects-delete-project-files"
            checked={deleteFilesOnDelete}
            onCheckedChange={setDeleteFilesOnDelete}
            description={
              deleting?.rootPath ??
              t("projectsPage.deleteProjectFiles")
            }
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setDeleting(null)}>
              {t("common.cancel")}
            </Button>
            <Button type="button" variant="destructive" onClick={() => void commitDelete()}>
              {deleteFilesOnDelete ? t("projectsPage.deleteAll") : t("common.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Shell>
  );
}
