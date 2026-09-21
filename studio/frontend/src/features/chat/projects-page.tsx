// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  archiveChatItem,
  deleteChatItem,
  deleteChatProject,
  notifyChatHistoryUpdated,
  renameChatItem,
  useChatProjects,
  useChatRuntimeStore,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  type ProjectRecord,
} from "@/features/chat";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildProjectsTourSteps } from "./tour";
import { EditProjectDialog } from "./components/edit-project-dialog";
import { NewProjectDialog } from "./components/new-project-dialog";
import {
  Archive03Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  Folder02Icon,
  FolderAddIcon,
  PinIcon,
  PinOffIcon,
  Search01Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowDownIcon, ChevronDownIcon, MoreHorizontalIcon } from "lucide-react";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  COMBINED_EXPORT_FORMATS_LIST,
  exportProjectConversations,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  EXPORT_FORMATS_LIST,
  type ConvExportFormat,
} from "./prompt-storage/prompt-storage-dialog";
import {
  fileImportSource,
  importConversationsFromSource,
  nativeImportSource,
  type ImportSource,
} from "./utils/chat-import";
import {
  listStoredChatThreads,
} from "./utils/chat-history-storage";
import { CHAT_HISTORY_UPDATED_EVENT } from "./api/chat-api";
import { groupThreads, type SidebarItem } from "./hooks/use-chat-sidebar-items";

type SortMode = "activity" | "name";

// Reveal this many more projects each time the user scrolls near the bottom.
const PROJECTS_PAGE_STEP = 12;
// A streaming chat fires the history event per chunk; one reload per quiet window is enough.
const PROJECT_CHATS_REFRESH_DEBOUNCE_MS = 300;
// Visible count before the fit-to-height measurement runs.
const PROJECTS_INITIAL_FALLBACK = 8;
// Approx list row height in px, used to estimate how many rows fit the page.
const PROJECTS_ROW_HEIGHT = 68;

// Updated column, matching a file-list feel: Today / Yesterday / N days ago, then a short date
// once it is over a week old.
function formatUpdated(ts: number): string {
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
  if (dayDiff <= 0) return "Today";
  if (dayDiff === 1) return "Yesterday";
  if (dayDiff < 7) return `${dayDiff} days ago`;
  return then.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

export function ProjectsPage() {
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const { projects, hasLoaded } = useChatProjects();

  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("activity");
  // Newest first, the way a file list opens.
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  // Rows that fit the page height (measured), plus any revealed via Show more.
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
  const pinnedChatIds = usePinnedChatsStore((s) => s.pinnedIds);
  const togglePinChat = usePinnedChatsStore((s) => s.togglePin);
  const pinnedChatIdSet = useMemo(() => new Set(pinnedChatIds), [pinnedChatIds]);

  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<ProjectRecord | null>(null);
  const [deleting, setDeleting] = useState<ProjectRecord | null>(null);
  const [renamingChat, setRenamingChat] = useState<SidebarItem | null>(null);
  const [chatNameDraft, setChatNameDraft] = useState("");
  const [deletingChat, setDeletingChat] = useState<SidebarItem | null>(null);

  const globalImportRef = useRef<HTMLInputElement>(null);
  const projectImportRefs = useRef<Map<string, HTMLInputElement>>(new Map());
  const [importFile, setImportFile] = useState<ImportSource | null>(null);
  // A second pick mid-import would interleave two streams into one history.
  const [importing, setImporting] = useState(false);
  // null = Recents
  const [importTargetId, setImportTargetId] = useState<string | null>(null);
  // Rows open their own chats in place, loaded the first time they are opened.
  const [openProjectIds, setOpenProjectIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // Grouped as the sidebar groups them, so a comparison is one row that opens as one.
  // "error" is a failed first load: the row says so and offers a retry, and reopening it loads
  // again rather than trusting the entry.
  const [projectChats, setProjectChats] = useState<
    Record<string, SidebarItem[] | "loading" | "error">
  >({});
  // Rows kept on screen after a reload failed: they may miss a chat or show one that is gone,
  // so the row says so, and its next open asks again.
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

  // One sequence per project: a response that a newer request overtook is dropped, so a chat
  // moved or deleted mid-flight cannot come back.
  const loadSeqRef = useRef(new Map<string, number>());
  const loadProjectChats = useCallback((projectId: string, silent = false) => {
    const seq = (loadSeqRef.current.get(projectId) ?? 0) + 1;
    loadSeqRef.current.set(projectId, seq);
    // A reload keeps the rows on screen; only a first load shows the skeleton.
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
        // A failed reload keeps rows already showing, marked stale; anything else becomes a
        // retryable error, including a first load this reload overtook while it was still
        // pending, and a folder loaded as empty, which the reload may have been about to fill.
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

  // A loaded list goes stale when chats are imported, moved or deleted. Streaming fires the
  // event per chunk, so the reload is debounced: open rows reload in place, the rest load
  // again on their next open.
  const openProjectIdsRef = useRef(openProjectIds);
  useEffect(() => {
    openProjectIdsRef.current = openProjectIds;
  }, [openProjectIds]);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const refresh = () => {
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        const open = openProjectIdsRef.current;
        // A closed project's load may still be in flight; its answer must not refill the cache.
        for (const [id, seq] of loadSeqRef.current) {
          if (!open.has(id)) loadSeqRef.current.set(id, seq + 1);
        }
        setProjectChats((prev) => {
          const kept: typeof prev = {};
          for (const id of open) if (prev[id] !== undefined) kept[id] = prev[id];
          return kept;
        });
        for (const id of open) loadProjectChats(id, true);
      }, PROJECT_CHATS_REFRESH_DEBOUNCE_MS);
    };
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, refresh);
    return () => {
      if (timer !== null) clearTimeout(timer);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, refresh);
    };
  }, [loadProjectChats]);

  async function handleImport(source: ImportSource, projectId: string | null) {
    // Counts up while it runs: a large export takes minutes of writes.
    setImporting(true);
    const toastId = toast.loading("Importing chats...");
    try {
      const { imported, failed } = await importConversationsFromSource(
        source,
        projectId,
        {
          onProgress: ({ imported: done, bytesRead, totalBytes }) => {
            const percent = totalBytes
              ? Math.min(100, Math.round((bytesRead / totalBytes) * 100))
              : 0;
            toast.loading(`Importing chats: ${done} so far (${percent}%)...`, {
              id: toastId,
            });
          },
        },
      );
      if (imported === 0 && failed === 0) {
        toast.info("No conversations found in file.", { id: toastId });
        return;
      }
      if (imported === 0) {
        // Nothing was created, so however the count is phrased this is a failure.
        toast.error("Import failed.", {
          id: toastId,
          description: `${failed} conversation${failed === 1 ? "" : "s"} could not be saved.`,
        });
        return;
      }
      const dest = projectId
        ? (projects.find((p) => p.id === projectId)?.name ?? "project")
        : "Recents";
      toast.success(
        failed > 0
          ? `Imported ${imported} conversation${imported === 1 ? "" : "s"} to ${dest}; ${failed} could not be saved.`
          : `Imported ${imported} conversation${imported === 1 ? "" : "s"} to ${dest}.`,
        { id: toastId },
      );
    } catch (error) {
      toast.error("Import failed.", {
        id: toastId,
        description: error instanceof Error ? error.message : undefined,
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
      setImportTargetId(projects[0]?.id ?? null);
      setImportFile(nativeImportSource(selected));
    } catch (error) {
      toast.error("Import failed.", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async function selectProjectImportFile(projectId: string) {
    if (importing) return;
    if (!isTauri) {
      projectImportRefs.current.get(projectId)?.click();
      return;
    }
    try {
      const selected = await pickNativeChatImport();
      if (!selected) return;
      await handleImport(nativeImportSource(selected), projectId);
    } catch (error) {
      toast.error("Import failed.", {
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
    filtered.sort((a, b) => {
      if (sortMode === "name") return a.name.localeCompare(b.name);
      // Direction belongs to the Updated column, which is the only one that carries an arrow.
      return sortDir === "asc" ? a.updatedAt - b.updatedAt : b.updatedAt - a.updatedAt;
    });
    return filtered;
  }, [projects, query, sortMode, sortDir]);
  // Default view shows as many rows as fit the page, then loads more as the user scrolls near the
  // bottom. Search always spans every project.
  const isSearching = query.trim() !== "";
  const visibleCount = baseFit + extraCount;
  const visibleProjects = isSearching
    ? sortedProjects
    : sortedProjects.slice(0, visibleCount);
  const hasMore = !isSearching && sortedProjects.length > visibleCount;

  // The list only renders once at least one project exists, so its step is dropped until then.
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

  // Estimate how many rows fit below the list's top so the first page fills the screen without
  // loading everything up front.
  useEffect(() => {
    function measure() {
      const el = listRef.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top;
      const reserve = 24; // bottom breathing room
      const fits = Math.floor(
        (window.innerHeight - top - reserve) / PROJECTS_ROW_HEIGHT,
      );
      setBaseFit(Math.max(PROJECTS_PAGE_STEP, fits));
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [hasLoaded]);

  // Infinite scroll: reveal another page-step whenever the sentinel near the list bottom scrolls into view.
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
    // Re-observe after each load so it keeps filling while the sentinel stays in view
    // (IntersectionObserver does not re-fire on a steady intersection).
  }, [hasMore, visibleCount]);

  function toggleProjectChats(projectId: string) {
    const opening = !openProjectIds.has(projectId);
    setOpenProjectIds((prev) => {
      const next = new Set(prev);
      if (opening) next.add(projectId);
      else next.delete(projectId);
      return next;
    });
    // Only an open asks: a close after a failed load must not start a request the reopen
    // then waits on.
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
    // A comparison restores from its pair id; a pane opened as a thread is half of it.
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

  async function handleProjectExport(project: ProjectRecord, fmt: ConvExportFormat) {
    try {
      const threads = await listStoredChatThreads({ projectId: project.id, includeArchived: false });
      const ids = [...new Set(threads.map((t) => t.id))];
      await exportProjectConversations(ids, fmt, project.name);
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Export failed.");
      }
    }
  }

  async function handleBulkProjectExport(
    scope: "projects" | "all",
    fmt: ConvExportFormat,
    merged: boolean,
  ) {
    try {
      let threads;
      if (scope === "projects") {
        threads = (
          await Promise.all(
            projects.map((p) =>
              listStoredChatThreads({ projectId: p.id, includeArchived: false }),
            ),
          )
        ).flat();
      } else {
        threads = await listStoredChatThreads({ includeArchived: false });
      }
      const ids = [...new Set(threads.map((t) => t.id))];
      if (ids.length === 0) { toast.info("No conversations to export."); return; }
      const ts = new Date().toISOString().slice(0, 10);
      const basename = `${scope === "all" ? "all-chats" : "all-projects"}-${ts}`;
      if (merged) {
        await exportBulkConversationsMerged(ids, fmt, basename);
      } else {
        await exportBulkConversationsSeparate(ids, fmt, basename);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Export failed.");
      }
    }
  }

  // Chat row actions, the same calls the sidebar's chat menu makes.
  async function commitChatRename() {
    const target = renamingChat;
    const name = chatNameDraft.trim();
    setRenamingChat(null);
    if (!target || !name || name === target.title) return;
    try {
      await renameChatItem(target, name);
      notifyChatHistoryUpdated();
    } catch (err) {
      toast.error("Failed to rename chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  // The open chat is the runtime's, not this page's, so its id comes from there.
  function activeThreadId(): string | undefined {
    return useChatRuntimeStore.getState().activeThreadId ?? undefined;
  }

  async function archiveChat(chat: SidebarItem) {
    try {
      await archiveChatItem(chat, activeThreadId(), () => {});
    } catch (err) {
      toast.error("Failed to archive chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function commitChatDelete() {
    const target = deletingChat;
    if (!target) return;
    setDeletingChat(null);
    try {
      await deleteChatItem(target, activeThreadId(), () => {});
    } catch (err) {
      toast.error("Failed to delete chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleChatExport(chat: SidebarItem, fmt: ConvExportFormat) {
    try {
      const ids = chat.threadIds?.length ? chat.threadIds : [chat.id];
      const safe = chat.title.replace(/[^a-z0-9_-]/gi, "_").slice(0, 40);
      await exportBulkConversationsMerged(ids, fmt, `chat-${safe}`);
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Export failed.");
      }
    }
  }

  async function commitDelete() {
    const target = deleting;
    if (!target) return;
    setDeleting(null);
    try {
      await deleteChatProject(target.id);
    } catch (err) {
      toast.error("Failed to delete project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <main className="mx-auto w-full max-w-5xl px-6 pb-10 pt-16 font-heading sm:px-10">
      <GuidedTour {...tour.tourProps} />
      {/* Global import file input */}
      <input
        ref={globalImportRef}
        type="file"
        accept=".json,.jsonl,.ndjson,.csv"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            setImportTargetId(projects[0]?.id ?? null);
            setImportFile(fileImportSource(file));
          }
          e.target.value = "";
        }}
      />
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          Projects
        </h1>
        <div className="flex items-center gap-3">
          <div className="relative">
            <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
              <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4" />
            </span>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search projects"
              className="h-9 w-52 rounded-full border-none bg-muted pl-10 pr-4 shadow-none dark:bg-card sm:w-64"
              aria-label="Search projects"
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Sort by</span>
            <Select
              value={sortMode}
              onValueChange={(v) => setSortMode(v as SortMode)}
            >
              <SelectTrigger className="h-9 w-[130px] rounded-full border-none bg-muted shadow-none dark:bg-card">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="activity">Activity</SelectItem>
                <SelectItem value="name">Name</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                data-tour="projects-io"
                title="Import / Export projects"
                className="rounded-full border-none bg-muted shadow-none dark:bg-card"
              >
                <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem onSelect={() => void selectGlobalImportFile()}>
                <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
                Import chats…
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>Export All Projects</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-52">
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      Combined
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
                      Per chat
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
                <DropdownMenuSubTrigger>Export Projects + Recents</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-52">
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
                      Combined
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
                      Per chat
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
          <Button data-tour="projects-new" onClick={() => setCreating(true)}>
            New project
          </Button>
        </div>
      </div>

      {!hasLoaded ? (
        <div className="mt-16">
          {/* The loaded header without its sort control, which has nothing to sort yet. */}
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-ui-13 font-medium text-muted-foreground">
            <span className="flex-1">Name</span>
            <span className="w-40 shrink-0">Updated</span>
            <span className="size-7 shrink-0" />
            <span className="w-8 shrink-0" />
          </div>
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="flex items-center gap-3 rounded-xl px-5 py-4"
            >
              <Skeleton className="mr-1 size-9 shrink-0 rounded-[10px]" />
              <Skeleton className="h-4 w-40 rounded-[8px]" />
              <span className="flex-1" />
              <Skeleton className="h-4 w-16 rounded-[8px]" />
              <span className="w-8 shrink-0" />
            </div>
          ))}
        </div>
      ) : visibleProjects.length === 0 ? (
        <div className="mt-16 flex flex-col items-center justify-center gap-2 text-center text-muted-foreground">
          <p className="text-sm">
            {projects.length === 0
              ? "No projects yet."
              : "No projects match your search."}
          </p>
          {projects.length === 0 && (
            <Button
              variant="outline"
              className="mt-2 border-none bg-background shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
              onClick={() => setCreating(true)}
            >
              <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className="size-icon" />
              Create your first project
            </Button>
          )}
        </div>
      ) : (
        <>
        <div className="mt-16">
          {/* Column header. Name starts at the folder icon's left edge, and the trailing
              spacers stand in for the row's pin and menu so Updated sits over its values. */}
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-ui-13 font-medium text-muted-foreground">
            <span className="flex-1">Name</span>
            {/* The column sorts the list, and the arrow says which way. */}
            <button
              type="button"
              onClick={() => {
                if (sortMode !== "activity") setSortMode("activity");
                else setSortDir((dir) => (dir === "desc" ? "asc" : "desc"));
              }}
              title={sortDir === "desc" ? "Newest first" : "Oldest first"}
              className="flex w-40 shrink-0 cursor-pointer items-center gap-1 text-left transition-colors hover:text-foreground"
            >
              Updated
              {/* Down for newest first, up for oldest, as a sorted column reads. */}
              <ArrowDownIcon
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  sortDir === "asc" && "rotate-180",
                  // The list is by name, so this column is not what orders it.
                  sortMode !== "activity" && "invisible",
                )}
              />
            </button>
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
            <input
              key={`import-${project.id}`}
              type="file"
              accept=".json,.jsonl,.ndjson,.csv"
              className="hidden"
              ref={(el) => {
                if (el) projectImportRefs.current.set(project.id, el);
                else projectImportRefs.current.delete(project.id);
              }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleImport(fileImportSource(file), project.id);
                e.target.value = "";
              }}
            />
            <div
              key={project.id}
              role="button"
              tabIndex={0}
              onClick={() => openProject(project.id)}
              onKeyDown={(e) => {
                // A key pressed on a control inside the row is that control's.
                if (e.target !== e.currentTarget) return;
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  openProject(project.id);
                }
              }}
              className="group/project-row relative flex cursor-pointer items-center gap-3 rounded-xl px-5 py-4 text-left transition-colors duration-150 hover:bg-muted/70 dark:hover:bg-white/[0.055] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <span className="mr-1 flex size-9 shrink-0 items-center justify-center rounded-[10px] bg-muted text-foreground/70 transition-colors group-hover/project-row:bg-primary/10 group-hover/project-row:text-primary">
                <HugeiconsIcon
                  icon={Folder02Icon}
                  strokeWidth={1.75}
                  className="size-5"
                />
              </span>
              {/* The disclosure belongs to the name, so it sits beside it rather than out by
                  the Updated column, where it read as another row action. */}
              <span className="flex min-w-0 flex-1 items-center gap-2">
                <span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">
                  {project.name}
                </span>
                {/* Opens the project's chats in place, without leaving the list. */}
                <button
                  type="button"
                  aria-label={chatsOpen ? "Hide chats" : "Show chats"}
                  aria-expanded={chatsOpen}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleProjectChats(project.id);
                  }}
                  className={cn(
                    "flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-black/5 hover:text-foreground focus-visible:opacity-100 group-hover/project-row:opacity-100 pointer-coarse:opacity-100 dark:hover:bg-white/10",
                    chatsOpen ? "opacity-100" : "opacity-0",
                  )}
                >
                  <ChevronDownIcon
                    strokeWidth={1.75}
                    className={cn(
                      "size-4 transition-transform",
                      !chatsOpen && "-rotate-90",
                    )}
                  />
                </button>
              </span>
              <span className="w-40 shrink-0 text-sm text-muted-foreground">
                {formatUpdated(project.updatedAt)}
              </span>
              {/* Pinning is one click here, as it is on a sidebar row. */}
              <button
                type="button"
                aria-label={pinned ? "Unpin project" : "Pin project"}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePinProject(project.id);
                }}
                // On show for every row, hover or not, so pinning is never hidden.
                className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10"
              >
                <HugeiconsIcon
                  icon={pinned ? PinOffIcon : PinIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </button>
              <div className="relative flex w-8 shrink-0 items-center justify-end">
                {/* On show beside the pin, so the row's actions read the same at rest. */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      aria-label="Project options"
                      className="absolute right-0 flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-black/5 hover:text-foreground data-[state=open]:bg-black/5 dark:hover:bg-white/10 dark:data-[state=open]:bg-white/10"
                    >
                      <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    side="bottom"
                    align="end"
                    sideOffset={0}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                    className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
                  >
                    {/* No pin item: the row's own button does it. Edit opens the sidebar's
                        dialog, which owns the name, instructions and source folders. */}
                    <DropdownMenuItem onSelect={() => setEditing(project)}>
                      <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Edit</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={(e) => {
                        e.stopPropagation();
                        void selectProjectImportFile(project.id);
                      }}
                    >
                      <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Import chats</span>
                    </DropdownMenuItem>
                    <DropdownMenuSub>
                      <DropdownMenuSubTrigger>
                        <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon mr-1" />
                        <span>Export</span>
                      </DropdownMenuSubTrigger>
                      <DropdownMenuSubContent className="w-52">
                        {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                          <DropdownMenuItem
                            key={fmt}
                            onSelect={(e) => {
                              e.stopPropagation();
                              void handleProjectExport(project, fmt);
                            }}
                          >
                            {label}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuSubContent>
                    </DropdownMenuSub>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => setDeleting(project)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Delete</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
            {chatsOpen && (
              <div className="mb-2 flex flex-col gap-0.5 pl-[76px]">
                {chats === undefined || chats === "loading" ? (
                  <Skeleton className="h-6 w-48 rounded-[8px]" />
                ) : chats === "error" ? (
                  <button
                    type="button"
                    onClick={() => loadProjectChats(project.id)}
                    className="cursor-pointer self-start py-1 text-left text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                  >
                    Could not load chats. Retry
                  </button>
                ) : chats.length === 0 ? (
                  <p className="py-1 text-sm text-muted-foreground">No chats</p>
                ) : (
                  <>
                    {chats.map((chat) => {
                      const chatPinned = pinnedChatIdSet.has(chat.id);
                      return (
                      <div
                        key={chat.id}
                        role="button"
                        tabIndex={0}
                        onClick={() => openChat(chat, project.id)}
                        onKeyDown={(e) => {
                          // A key pressed on a control inside the row is that control's.
                          if (e.target !== e.currentTarget) return;
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            openChat(chat, project.id);
                          }
                        }}
                        className="group/chat-row flex cursor-pointer items-center gap-3 rounded-xl py-1.5 pl-2 pr-5 text-left text-sm text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground dark:hover:bg-white/[0.055]"
                      >
                        <HugeiconsIcon
                          icon={MessageCircleIcon}
                          strokeWidth={1.75}
                          className="size-4 shrink-0"
                        />
                        <span className="min-w-0 flex-1 truncate">{chat.title}</span>
                        {/* Same widths as a project row, so the columns line up under it. */}
                        <span className="w-40 shrink-0">
                          {formatUpdated(chat.updatedAt)}
                        </span>
                        {/* A chat's actions belong to the row the cursor is on, so they stay
                            hover-revealed, and show outright without a cursor to hover with. */}
                        <button
                          type="button"
                          aria-label={chatPinned ? "Unpin chat" : "Pin chat"}
                          onClick={(e) => {
                            e.stopPropagation();
                            togglePinChat(chat.id);
                          }}
                          className={cn(
                            "flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-black/5 hover:text-foreground focus-visible:opacity-100 group-hover/chat-row:opacity-100 pointer-coarse:opacity-100 dark:hover:bg-white/10",
                            chatPinned ? "opacity-100" : "opacity-0",
                          )}
                        >
                          <HugeiconsIcon
                            icon={chatPinned ? PinOffIcon : PinIcon}
                            strokeWidth={1.75}
                            className="size-4"
                          />
                        </button>
                        <div className="relative flex w-8 shrink-0 items-center justify-end">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <button
                                type="button"
                                onClick={(e) => e.stopPropagation()}
                                aria-label="Chat options"
                                className="absolute right-0 flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition hover:bg-black/5 hover:text-foreground focus-visible:opacity-100 group-hover/chat-row:opacity-100 pointer-coarse:opacity-100 data-[state=open]:bg-black/5 data-[state=open]:opacity-100 dark:hover:bg-white/10 dark:data-[state=open]:bg-white/10"
                              >
                                <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
                              </button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent
                              side="bottom"
                              align="end"
                              sideOffset={0}
                              onClick={(e) => e.stopPropagation()}
                              onKeyDown={(e) => e.stopPropagation()}
                              className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
                            >
                              <DropdownMenuItem
                                onSelect={() => {
                                  setChatNameDraft(chat.title);
                                  setRenamingChat(chat);
                                }}
                              >
                                <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Rename</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem onSelect={() => void archiveChat(chat)}>
                                <HugeiconsIcon icon={Archive03Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Archive</span>
                              </DropdownMenuItem>
                              <DropdownMenuSub>
                                <DropdownMenuSubTrigger>
                                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon mr-1" />
                                  <span>Export</span>
                                </DropdownMenuSubTrigger>
                                <DropdownMenuSubContent className="w-52">
                                  {/* A comparison is two threads, so it can only take the
                                      formats that merge. */}
                                  {((chat.threadIds?.length ?? 1) > 1
                                    ? COMBINED_EXPORT_FORMATS_LIST
                                    : EXPORT_FORMATS_LIST
                                  ).map(({ fmt, label }) => (
                                    <DropdownMenuItem
                                      key={`${chat.id}-${fmt}`}
                                      onSelect={(e) => {
                                        e.stopPropagation();
                                        void handleChatExport(chat, fmt);
                                      }}
                                    >
                                      {label}
                                    </DropdownMenuItem>
                                  ))}
                                </DropdownMenuSubContent>
                              </DropdownMenuSub>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                variant="destructive"
                                onSelect={() => setDeletingChat(chat)}
                              >
                                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Delete</span>
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
                        Could not refresh chats. Retry
                      </button>
                    )}
                  </>
                )}
              </div>
            )}
            </div>
            );
          })}
          {/* Loads the next page-step when scrolled into view. */}
          {hasMore && <div ref={sentinelRef} className="h-px w-full" />}
          </div>
        </div>
        </>
      )}

      {/* Create project (name + drag-and-drop sources) */}
      <NewProjectDialog open={creating} onOpenChange={setCreating} />

      {/* Edit project (name + instructions + source folders), the sidebar's dialog. */}
      <EditProjectDialog
        project={editing}
        onOpenChange={(open) => {
          if (!open) setEditing(null);
        }}
        // Delete keeps this page's own confirmation.
        onDelete={(project) => setDeleting(project)}
      />

      {/* Rename chat */}
      <Dialog
        open={renamingChat !== null}
        onOpenChange={(open) => {
          if (!open) setRenamingChat(null);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rename chat</DialogTitle>
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
            placeholder="Chat name"
            aria-label="Chat name"
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setRenamingChat(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void commitChatRename()}
              disabled={
                !chatNameDraft.trim() ||
                chatNameDraft.trim() === renamingChat?.title
              }
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete chat */}
      <Dialog
        open={deletingChat !== null}
        onOpenChange={(open) => {
          if (!open) setDeletingChat(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete chat</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete <em>{deletingChat?.title}</em>? Its messages
            will be permanently deleted.
          </p>
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setDeletingChat(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void commitChatDelete()}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import destination picker */}
      <Dialog open={importFile !== null} onOpenChange={(open) => { if (!open) setImportFile(null); }}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Import chats</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Choose where to import{" "}
            {/* A picked local file, and the dialog portals out of any marked
                ancestor, so the name needs its own marker. */}
            <span
              data-reload-snapshot-sensitive
              className="font-medium text-foreground"
            >
              {importFile?.name}
            </span>
            :
          </p>
          <Select
            value={importTargetId ?? "__recents__"}
            onValueChange={(v) => setImportTargetId(v === "__recents__" ? null : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select destination" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__recents__">Recents</SelectItem>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setImportFile(null)}>Cancel</Button>
            <Button type="button" onClick={() => void commitImport()}>Import</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete project */}
      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete project</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete <em>{deleting?.name}</em>? Its chats will
            be permanently deleted.
          </p>
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button type="button" variant="destructive" onClick={() => void commitDelete()}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
