// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { useChatProjectScope } from "../chat-project-scope";
import {
  updateChatProjectGoal,
  useChatProjects,
} from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";
import { projectGoalViewState } from "./project-goal-state";

export function ProjectGoalBar() {
  const projectId = useChatProjectScope();
  const { projects } = useChatProjects();
  const project = projectId
    ? projects.find((candidate) => candidate.id === projectId)
    : undefined;
  const goal = project?.goal?.trim();
  if (!project) {
    return null;
  }

  return (
    <ProjectGoalBarContent
      key={`${project.id}:${project.goalUpdatedAt ?? 0}:${goal ?? ""}`}
      project={project}
    />
  );
}

function ProjectGoalBarContent({ project }: { project: ProjectRecord }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(project.goal ?? "");
  const [saving, setSaving] = useState(false);
  const view = projectGoalViewState(project);
  const { goal, status } = view;
  const activeProject = project;
  const goalInput = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing) {
      goalInput.current?.focus();
    }
  }, [editing]);

  function reportSaveError(
    error: unknown,
    title = "Could not update the project goal",
  ): void {
    toast.error(title, {
      description:
        error instanceof Error ? error.message : "Try the change again.",
    });
  }

  async function saveGoal() {
    const next = draft.trim();
    if (!next || saving) {
      return;
    }
    setSaving(true);
    try {
      await updateChatProjectGoal(activeProject.id, {
        goal: next,
        goalStatus: "active",
      });
      setEditing(false);
    } catch (error) {
      reportSaveError(error);
    } finally {
      setSaving(false);
    }
  }

  async function setStatus(next: "active" | "paused" | "completed") {
    if (saving) {
      return;
    }
    setSaving(true);
    try {
      await updateChatProjectGoal(activeProject.id, { goalStatus: next });
    } catch (error) {
      reportSaveError(
        error,
        next === "completed"
          ? "Could not complete the project goal"
          : "Could not update the project goal",
      );
    } finally {
      setSaving(false);
    }
  }

  async function clearGoal() {
    if (saving) {
      return;
    }
    setSaving(true);
    try {
      await updateChatProjectGoal(activeProject.id, {
        goal: null,
        goalStatus: null,
      });
    } catch (error) {
      reportSaveError(error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section
      aria-label="Project goal"
      aria-busy={saving}
      className="relative z-10 mb-2 rounded-[16px] border border-border/70 bg-background/95 px-3 py-2 shadow-sm backdrop-blur dark:bg-card/95"
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            status === "active" && "bg-emerald-500",
            status === "paused" && "bg-amber-500",
            status === "completed" && "bg-muted-foreground/50",
            status === null && "bg-muted-foreground/30",
          )}
          aria-hidden="true"
        />
        <span className="shrink-0 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
          Goal
        </span>
        {editing ? (
          <textarea
            ref={goalInput}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            maxLength={12_000}
            rows={2}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                void saveGoal();
              }
              if (event.key === "Escape") {
                setDraft(goal);
                setEditing(false);
              }
            }}
            disabled={saving}
            aria-label="Project goal. Press Control or Command Enter to save."
            className="max-h-40 min-h-10 min-w-0 flex-1 resize-y rounded-lg border border-border bg-transparent px-2 py-1 text-sm outline-none focus:border-ring"
          />
        ) : (
          <span
            className={cn(
              "min-w-0 flex-1 truncate text-sm",
              status !== "active" && "text-muted-foreground",
              status === "completed" && "line-through",
            )}
            title={goal || undefined}
          >
            {view.label}
          </span>
        )}
        {status ? (
          <span className="sr-only" aria-live="polite">
            Goal status: {status}. {saving ? "Saving." : ""}
          </span>
        ) : null}
        <div className="flex shrink-0 items-center gap-1">
          {editing ? (
            <>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                disabled={!draft.trim() || saving}
                onClick={() => void saveGoal()}
              >
                Save
              </Button>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                disabled={saving}
                onClick={() => {
                  setDraft(goal);
                  setEditing(false);
                }}
              >
                Cancel
              </Button>
            </>
          ) : (
            <>
              {goal ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={saving}
                  onClick={() => setEditing(true)}
                >
                  Edit
                </Button>
              ) : (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={saving}
                  onClick={() => {
                    setDraft("");
                    setEditing(true);
                  }}
                >
                  Set goal
                </Button>
              )}
              {status === "active" ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={saving}
                  onClick={() => void setStatus("paused")}
                >
                  Pause
                </Button>
              ) : status === "paused" ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={saving}
                  onClick={() => void setStatus("active")}
                >
                  Resume
                </Button>
              ) : null}
              {goal ? (
                <>
                  {status !== "completed" ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={saving}
                      onClick={() => void setStatus("completed")}
                    >
                      Complete
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={saving}
                      onClick={() => void setStatus("active")}
                    >
                      Reopen
                    </Button>
                  )}
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    disabled={saving}
                    onClick={() => void clearGoal()}
                  >
                    Clear
                  </Button>
                </>
              ) : null}
            </>
          )}
        </div>
      </div>
      <p className="mt-1 text-ui-11 text-muted-foreground">{view.hint}</p>
    </section>
  );
}
