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
  DropdownMenuItem,
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
import { toast } from "@/lib/toast";
import {
  createChatProject,
  deleteChatProject,
  renameChatProject,
  useChatProjects,
  useChatRuntimeStore,
  type ProjectRecord,
} from "@/features/chat";
import {
  Delete02Icon,
  Edit03Icon,
  FolderAddIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MoreHorizontalIcon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";

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

  const [creating, setCreating] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [renaming, setRenaming] = useState<ProjectRecord | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleting, setDeleting] = useState<ProjectRecord | null>(null);

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
    return filtered;
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
    <main className="mx-auto w-full max-w-7xl px-4 py-8 font-heading sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Projects
        </h1>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Sort by</span>
            <Select
              value={sortMode}
              onValueChange={(v) => setSortMode(v as SortMode)}
            >
              <SelectTrigger className="h-9 w-[130px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="activity">Activity</SelectItem>
                <SelectItem value="name">Name</SelectItem>
              </SelectContent>
            </Select>
          </div>
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

      <div className="relative mt-6">
        <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
          <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-icon" />
        </span>
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search projects..."
          className="h-11 pl-10"
          aria-label="Search projects"
        />
      </div>

      {!hasLoaded ? (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="min-h-[160px] rounded-[14px] border border-border/70 bg-card p-5"
            >
              <Skeleton className="h-5 w-2/3 rounded-[6px]" />
              <Skeleton className="mt-3 h-4 w-full rounded-[6px]" />
              <Skeleton className="mt-2 h-4 w-4/5 rounded-[6px]" />
              <Skeleton className="mt-12 h-3 w-24 rounded-[6px]" />
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
              className="mt-2"
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
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {visibleProjects.map((project) => (
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
              className="group/project-card relative flex min-h-[160px] cursor-pointer flex-col rounded-[14px] border border-border/70 bg-card p-5 text-left transition-colors hover:border-border hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <div className="flex items-start justify-between gap-2">
                <h2 className="truncate pr-2 text-[16px] font-semibold text-foreground">
                  {project.name}
                </h2>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      onClick={(e) => e.stopPropagation()}
                      aria-label="Project options"
                      className="-mr-1 -mt-1 inline-flex size-7 shrink-0 items-center justify-center rounded-[8px] text-muted-foreground opacity-0 transition-opacity hover:bg-accent hover:text-foreground focus-visible:opacity-100 group-hover/project-card:opacity-100"
                    >
                      <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    side="bottom"
                    align="end"
                    sideOffset={4}
                    className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
                  >
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
                      variant="destructive"
                      onSelect={() => setDeleting(project)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Delete</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              {project.instructions ? (
                <p className="mt-2 line-clamp-3 text-sm text-muted-foreground">
                  {project.instructions}
                </p>
              ) : null}
              <span className="mt-auto pt-4 text-xs text-muted-foreground">
                Updated {formatUpdatedAgo(project.updatedAt)}
              </span>
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
        <DialogContent className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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
        <DialogContent className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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

      {/* Delete project */}
      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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
