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
import { useMemo, useRef, useState } from "react";
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

function formatUpdatedAgo(ts: number): string {
  const diff = Date.now() - ts;
  if (!Number.isFinite(diff) || diff < 0) return "just now";
  const s = Math.floor(diff / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} minute${m === 1 ? "" : "s"} ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d} day${d === 1 ? "" : "s"} ago`;
  const mo = Math.floor(d / 30);
  if (mo < 12) return `${mo} month${mo === 1 ? "" : "s"} ago`;
  const y = Math.floor(mo / 12);
  return `${y} year${y === 1 ? "" : "s"} ago`;
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const { projects, hasLoaded } = useChatProjects();

  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("activity");
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

  const visibleProjects = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    const filtered = trimmed
      ? projects.filter((p) => p.name.toLowerCase().includes(trimmed))
      : projects.slice();
    filtered.sort((a, b) =>
      sortMode === "name"
        ? a.name.localeCompare(b.name)
        : b.updatedAt - a.updatedAt,
    );
    // Default view shows only the 4 most recent; search still spans all matches.
    return trimmed ? filtered : filtered.slice(0, 4);
  }, [projects, query, sortMode]);

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
    <main className="mx-auto w-full max-w-6xl px-6 py-10 font-heading sm:px-10">
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
        <div className="mt-12 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="min-h-[172px] rounded-[26px] bg-card p-6 shadow-[0_2px_12px_-4px_rgba(0,0,0,0.10)] dark:shadow-none"
            >
              <Skeleton className="size-10 rounded-[14px]" />
              <Skeleton className="mt-4 h-5 w-2/3 rounded-[8px]" />
              <Skeleton className="mt-2 h-4 w-4/5 rounded-[8px]" />
              <Skeleton className="mt-8 h-3 w-24 rounded-[8px]" />
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
        <div className="mt-12 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {visibleProjects.map((project) => (
            <div key={`wrap-${project.id}`} className="contents">
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
              className="group/project-card relative flex min-h-[172px] cursor-pointer flex-col rounded-[26px] bg-card p-6 text-left shadow-[0_2px_12px_-4px_rgba(0,0,0,0.10)] transition-colors duration-150 hover:bg-[#f2f2f2] dark:shadow-none dark:hover:bg-accent/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-[14px] bg-muted text-foreground/70 transition-colors group-hover/project-card:bg-primary/10 group-hover/project-card:text-primary">
                  <HugeiconsIcon
                    icon={Folder02Icon}
                    strokeWidth={1.75}
                    className="size-5"
                  />
                </span>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      aria-label="Project options"
                      className="-mr-1 -mt-1 inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10 focus-visible:opacity-100 group-hover/project-card:opacity-100 data-[state=open]:bg-black/5 data-[state=open]:opacity-100 dark:data-[state=open]:bg-white/10"
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
                        icon={
                          pinnedProjectIdSet.has(project.id)
                            ? PinOffIcon
                            : PinIcon
                        }
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      <span>
                        {pinnedProjectIdSet.has(project.id)
                          ? "Unpin project"
                          : "Pin project"}
                      </span>
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
              <h2 className="mt-4 truncate text-[16px] font-semibold text-foreground">
                {project.name}
              </h2>
              {project.instructions ? (
                <p className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                  {project.instructions}
                </p>
              ) : null}
              <span className="mt-auto pt-4 text-xs text-muted-foreground/80">
                Updated {formatUpdatedAgo(project.updatedAt)}
              </span>
            </div>
            </div>
          ))}
        </div>
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
