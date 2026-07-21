// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { isDownloadCancelled, pickNativeChatImport } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  createChatProject,
  deleteChatProject,
  renameChatProject,
  useChatProjects,
  useChatRuntimeStore,
  usePinnedProjectsStore,
  type ProjectRecord,
} from "@/features/chat";
import {
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
import { MoreHorizontalIcon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  exportProjectConversations,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  importConversationsFromFile,
  EXPORT_FORMATS_LIST,
  type ConvExportFormat,
} from "./prompt-storage/prompt-storage-dialog";
import {
  listStoredChatThreads,
} from "./utils/chat-history-storage";

type SortMode = "activity" | "name";

// Reveal this many more projects each time the user scrolls near the bottom.
const PROJECTS_PAGE_STEP = 12;
// Visible count before the fit-to-height measurement runs.
const PROJECTS_INITIAL_FALLBACK = 8;
// Approx list row height in px, used to estimate how many rows fit the page.
const PROJECTS_ROW_HEIGHT = 68;

// Modified column, matching a file-list feel: Today / Yesterday / N days ago,
// then a short date once it is over a week old.
function formatModified(ts: number): string {
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
  const navigate = useNavigate();
  const { projects, hasLoaded } = useChatProjects();

  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("activity");
  // Rows that fit the page height (measured), plus any revealed via Show more.
  const [baseFit, setBaseFit] = useState(PROJECTS_INITIAL_FALLBACK);
  const [extraCount, setExtraCount] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const togglePinProject = usePinnedProjectsStore((s) => s.togglePin);
  const pinnedProjectIdSet = useMemo(
    () => new Set(pinnedProjectIds),
    [pinnedProjectIds],
  );

  const [creating, setCreating] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [renaming, setRenaming] = useState<ProjectRecord | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleting, setDeleting] = useState<ProjectRecord | null>(null);

  const globalImportRef = useRef<HTMLInputElement>(null);
  const projectImportRefs = useRef<Map<string, HTMLInputElement>>(new Map());
  const [importFile, setImportFile] = useState<File | null>(null);
  // null = Recents
  const [importTargetId, setImportTargetId] = useState<string | null>(null);

  async function handleImport(file: File, projectId: string | null) {
    try {
      const count = await importConversationsFromFile(file, projectId);
      if (count === 0) {
        toast.info("No conversations found in file.");
      } else {
        const dest = projectId
          ? (projects.find((p) => p.id === projectId)?.name ?? "project")
          : "Recents";
        toast.success(`Imported ${count} conversation${count === 1 ? "" : "s"} to ${dest}.`);
      }
    } catch {
      toast.error("Import failed.");
    }
  }


  async function selectGlobalImportFile() {
    if (!isTauri) {
      globalImportRef.current?.click();
      return;
    }
    try {
      const selected = await pickNativeChatImport();
      if (!selected) return;
      setImportTargetId(projects[0]?.id ?? null);
      setImportFile(new File([selected.content], selected.name));
    } catch (error) {
      toast.error("Import failed.", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async function selectProjectImportFile(projectId: string) {
    if (!isTauri) {
      projectImportRefs.current.get(projectId)?.click();
      return;
    }
    try {
      const selected = await pickNativeChatImport();
      if (!selected) return;
      await handleImport(new File([selected.content], selected.name), projectId);
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
    filtered.sort((a, b) =>
      sortMode === "name"
        ? a.name.localeCompare(b.name)
        : b.updatedAt - a.updatedAt,
    );
    return filtered;
  }, [projects, query, sortMode]);
  // Default view shows as many rows as fit the page, then loads more as the
  // user scrolls near the bottom. Search always spans every project.
  const isSearching = query.trim() !== "";
  const visibleCount = baseFit + extraCount;
  const visibleProjects = isSearching
    ? sortedProjects
    : sortedProjects.slice(0, visibleCount);
  const hasMore = !isSearching && sortedProjects.length > visibleCount;

  // Estimate how many rows fit below the list's top so the first page fills the
  // screen without loading everything up front.
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

  // Infinite scroll: reveal another page-step whenever the sentinel near the
  // list bottom scrolls into view.
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
    // Re-observe after each load so it keeps filling while the sentinel stays
    // in view (IntersectionObserver does not re-fire on a steady intersection).
  }, [hasMore, visibleCount]);

  function openProject(projectId: string) {
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(projectId);
    navigate({ to: "/chat", search: { project: projectId } });
  }

  async function commitCreate() {
    const name = nameDraft.trim();
    if (!name) return;
    try {
      const project = await createChatProject(name);
      setCreating(false);
      setNameDraft("");
      openProject(project.id);
    } catch (err) {
      toast.error("Failed to create project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function commitRename() {
    const target = renaming;
    const name = renameDraft.trim();
    if (!target || !name || name === target.name) {
      setRenaming(null);
      return;
    }
    setRenaming(null);
    try {
      await renameChatProject(target.id, name);
    } catch (err) {
      toast.error("Failed to rename project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
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
      {/* Global import file input */}
      <input
        ref={globalImportRef}
        type="file"
        accept=".jsonl,.ndjson,.csv"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            setImportTargetId(projects[0]?.id ?? null);
            setImportFile(file);
          }
          e.target.value = "";
        }}
      />
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
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
                    <DropdownMenuLabel className="pb-1 pt-2 text-[11px] font-medium">
                      Combined
                    </DropdownMenuLabel>
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`ap-m-${fmt}`} onSelect={() => void handleBulkProjectExport("projects", fmt, true)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator />
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-[11px] font-medium">
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
                    <DropdownMenuLabel className="pb-1 pt-2 text-[11px] font-medium">
                      Combined
                    </DropdownMenuLabel>
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`all-m-${fmt}`} onSelect={() => void handleBulkProjectExport("all", fmt, true)}>
                        {label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator />
                  <DropdownMenuGroup>
                    <DropdownMenuLabel className="pb-1 pt-2 text-[11px] font-medium">
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
          <Button
            onClick={() => {
              setNameDraft("");
              setCreating(true);
            }}
          >
            New project
          </Button>
        </div>
      </div>

      {!hasLoaded ? (
        <div className="mt-16">
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-[13px] font-medium text-muted-foreground">
            <span className="flex-1">Name</span>
            <span className="w-40 shrink-0">Modified</span>
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
              onClick={() => {
                setNameDraft("");
                setCreating(true);
              }}
            >
              <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className="size-icon" />
              Create your first project
            </Button>
          )}
        </div>
      ) : (
        <>
        <div className="mt-16">
          {/* Column header. Name starts at the folder icon's left edge; the
              right-anchored columns keep Modified over its values. */}
          <div className="mb-1 flex items-center gap-3 px-5 pb-1 text-[13px] font-medium text-muted-foreground">
            <span className="flex-1">Name</span>
            <span className="w-40 shrink-0">Modified</span>
            <span className="w-8 shrink-0" />
          </div>
          <div ref={listRef}>
          {visibleProjects.map((project) => {
            const pinned = pinnedProjectIdSet.has(project.id);
            return (
            <div key={`wrap-${project.id}`}>
            <input
              key={`import-${project.id}`}
              type="file"
              accept=".jsonl,.ndjson,.csv"
              className="hidden"
              ref={(el) => {
                if (el) projectImportRefs.current.set(project.id, el);
                else projectImportRefs.current.delete(project.id);
              }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleImport(file, project.id);
                e.target.value = "";
              }}
            />
            <div
              key={project.id}
              role="button"
              tabIndex={0}
              onClick={() => openProject(project.id)}
              onKeyDown={(e) => {
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
              <span className="min-w-0 flex-1 truncate text-[15px] font-semibold text-foreground">
                {project.name}
              </span>
              <span className="w-40 shrink-0 text-sm text-muted-foreground">
                {formatModified(project.updatedAt)}
              </span>
              <div className="flex w-8 shrink-0 items-center justify-end">
                {/* Pin indicator and options button swap via display so they
                    never overlap: pin when idle, kebab on hover or menu open. */}
                {pinned && (
                  <span className="text-muted-foreground group-hover/project-row:hidden group-has-[[data-state=open]]/project-row:hidden">
                    <HugeiconsIcon icon={PinIcon} strokeWidth={1.75} className="size-4" />
                  </span>
                )}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      aria-label="Project options"
                      className="hidden size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10 group-hover/project-row:flex data-[state=open]:flex data-[state=open]:bg-black/5 dark:data-[state=open]:bg-white/10"
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
                      onSelect={() => togglePinProject(project.id)}
                    >
                      <HugeiconsIcon
                        icon={pinned ? PinOffIcon : PinIcon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      <span>{pinned ? "Unpin project" : "Pin project"}</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => {
                        setRenameDraft(project.name);
                        setRenaming(project);
                      }}
                    >
                      <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Rename</span>
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
                        {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
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
            </div>
            );
          })}
          {/* Loads the next page-step when scrolled into view. */}
          {hasMore && <div ref={sentinelRef} className="h-px w-full" />}
          </div>
        </div>
        </>
      )}

      {/* Create project */}
      <Dialog
        open={creating}
        onOpenChange={(open) => {
          if (!open) setCreating(false);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>New project</DialogTitle>
          </DialogHeader>
          <Input
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitCreate();
              }
            }}
            autoFocus
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setCreating(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={() => void commitCreate()} disabled={!nameDraft.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename project */}
      <Dialog
        open={renaming !== null}
        onOpenChange={(open) => {
          if (!open) setRenaming(null);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rename project</DialogTitle>
          </DialogHeader>
          <Input
            value={renameDraft}
            onChange={(e) => setRenameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitRename();
              }
            }}
            autoFocus
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void commitRename()}
              disabled={!renameDraft.trim() || renameDraft.trim() === renaming?.name}
            >
              Save
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
            <span className="font-medium text-foreground">{importFile?.name}</span>:
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
            Are you sure you want to delete <em>{deleting?.name}</em>? Chats in this
            project will be moved back to Recents.
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
