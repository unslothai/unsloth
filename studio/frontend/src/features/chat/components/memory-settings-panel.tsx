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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { SettingsRow, SettingsSection } from "@/features/settings";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { listChatProjects } from "../api/chat-api";
import {
  type ChatMemory,
  type MemoryScope,
  clearChatMemories,
  createChatMemory,
  deleteChatMemory,
  exportChatMemories,
  listChatMemories,
  updateChatMemory,
} from "../api/chat-memory-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ProjectRecord } from "../types";

type MemoryEditorState = {
  memory: ChatMemory | null;
  content: string;
  scope: MemoryScope;
  projectId: string;
};

export function MemorySettingsPanel() {
  const t = useT();
  const referenceMemories = useChatRuntimeStore(
    (state) => state.referenceMemories,
  );
  const autoSaveMemories = useChatRuntimeStore(
    (state) => state.autoSaveMemories,
  );
  const setReferenceMemories = useChatRuntimeStore(
    (state) => state.setReferenceMemories,
  );
  const setAutoSaveMemories = useChatRuntimeStore(
    (state) => state.setAutoSaveMemories,
  );
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
  );
  const activeProjectId = useChatRuntimeStore((state) => state.activeProjectId);

  const [scope, setScope] = useState<MemoryScope>("global");
  const [projectId, setProjectId] = useState("");
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [memories, setMemories] = useState<ChatMemory[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const [memoriesError, setMemoriesError] = useState<string | null>(null);
  const [editor, setEditor] = useState<MemoryEditorState | null>(null);
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearOpen, setClearOpen] = useState(false);
  const [clearing, setClearing] = useState(false);
  const requestVersion = useRef(0);

  useEffect(() => {
    void hydratePersistedSettings();
  }, [hydratePersistedSettings]);

  useEffect(() => {
    let cancelled = false;
    void listChatProjects({ includeArchived: true })
      .then((projectRows) => {
        if (cancelled) return;
        setProjects(projectRows);
        setProjectId((current) => {
          if (
            current &&
            projectRows.some((project) => project.id === current)
          ) {
            return current;
          }
          if (
            activeProjectId &&
            projectRows.some((project) => project.id === activeProjectId)
          ) {
            return activeProjectId;
          }
          return projectRows[0]?.id ?? "";
        });
      })
      .catch((error) => {
        if (!cancelled) {
          toast.error(t("settings.profile.memory.projectsLoadError"), {
            description: error instanceof Error ? error.message : undefined,
          });
        }
      })
      .finally(() => {
        if (!cancelled) setProjectsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProjectId, t]);

  const loadMemories = useCallback(async () => {
    if (scope === "project" && !projectId) {
      requestVersion.current += 1;
      setMemories([]);
      setMemoriesError(null);
      setMemoriesLoading(false);
      return;
    }

    const version = ++requestVersion.current;
    setMemoriesLoading(true);
    setMemoriesError(null);
    try {
      const data = await listChatMemories(
        scope,
        scope === "project" ? projectId : undefined,
      );
      if (version === requestVersion.current) setMemories(data.memories);
    } catch (error) {
      if (version === requestVersion.current) {
        setMemoriesError(error instanceof Error ? error.message : "");
        setMemories([]);
      }
    } finally {
      if (version === requestVersion.current) setMemoriesLoading(false);
    }
  }, [projectId, scope]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadMemories(), 0);
    return () => window.clearTimeout(timer);
  }, [loadMemories]);

  const projectRequired = scope === "project" && !projectId;
  const editorProjectRequired =
    editor?.scope === "project" && !editor.projectId;

  const openCreate = () => {
    setEditor({
      memory: null,
      content: "",
      scope,
      projectId,
    });
  };

  const saveMemory = async () => {
    if (!editor || !editor.content.trim() || editorProjectRequired) return;
    setSaving(true);
    try {
      if (editor.memory) {
        await updateChatMemory(editor.memory.id, {
          content: editor.content.trim(),
          scope: editor.scope,
          projectId: editor.scope === "project" ? editor.projectId : null,
        });
        toast.success(t("settings.profile.memory.updated"));
      } else {
        const result = await createChatMemory({
          content: editor.content.trim(),
          scope: editor.scope,
          projectId: editor.scope === "project" ? editor.projectId : null,
        });
        toast[result.created ? "success" : "info"](
          t(
            result.created
              ? "settings.profile.memory.saved"
              : "settings.profile.memory.duplicate",
          ),
        );
      }
      setEditor(null);
      await loadMemories();
    } catch (error) {
      toast.error(
        editor.memory
          ? t("settings.profile.memory.updateError")
          : t("settings.profile.memory.saveError"),
        { description: error instanceof Error ? error.message : undefined },
      );
    } finally {
      setSaving(false);
    }
  };

  const removeMemory = async (id: string) => {
    setDeletingId(id);
    try {
      await deleteChatMemory(id);
      await loadMemories();
      toast.success(t("settings.profile.memory.deleted"));
    } catch (error) {
      toast.error(t("settings.profile.memory.deleteError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setDeletingId(null);
    }
  };

  const exportMemories = async () => {
    try {
      const data = await exportChatMemories();
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(data, null, 2)], {
          type: "application/json",
        }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "unsloth-memories.json";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      toast.error(t("settings.profile.memory.exportError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const clearMemories = async () => {
    if (projectRequired) return;
    setClearing(true);
    try {
      await clearChatMemories(
        scope,
        scope === "project" ? projectId : undefined,
      );
      setClearOpen(false);
      await loadMemories();
      toast.success(t("settings.profile.memory.cleared"));
    } catch (error) {
      toast.error(t("settings.profile.memory.clearError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setClearing(false);
    }
  };

  const selectedProject = projects.find((project) => project.id === projectId);
  const selectedScopeLabel =
    scope === "global"
      ? t("settings.profile.memory.global")
      : (selectedProject?.name ?? t("settings.profile.memory.project"));

  return (
    <div className="mx-auto flex w-full max-w-[640px] flex-col gap-6">
      <SettingsSection
        title={t("settings.profile.memory.title")}
        description={t("settings.profile.memory.description")}
      >
        <SettingsRow
          label={t("settings.profile.memory.referenceLabel")}
          description={t("settings.profile.memory.referenceDescription")}
        >
          <Switch
            aria-label={t("settings.profile.memory.referenceLabel")}
            checked={referenceMemories}
            onCheckedChange={setReferenceMemories}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.profile.memory.autoSaveLabel")}
          description={t("settings.profile.memory.autoSaveDescription")}
        >
          <Switch
            aria-label={t("settings.profile.memory.autoSaveLabel")}
            checked={autoSaveMemories}
            onCheckedChange={setAutoSaveMemories}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.profile.memory.savedTitle")}
        description={t("settings.profile.memory.savedDescription")}
      >
        <div className="flex flex-wrap items-end justify-between gap-3 py-3">
          <div className="flex min-w-0 flex-1 flex-wrap gap-3">
            <div className="flex min-w-36 flex-1 flex-col gap-1.5">
              <Label htmlFor="memory-scope">
                {t("settings.profile.memory.scope")}
              </Label>
              <select
                id="memory-scope"
                className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                value={scope}
                onChange={(event) =>
                  setScope(event.target.value as MemoryScope)
                }
              >
                <option value="global">
                  {t("settings.profile.memory.global")}
                </option>
                <option value="project">
                  {t("settings.profile.memory.project")}
                </option>
              </select>
            </div>
            {scope === "project" ? (
              <div className="flex min-w-40 flex-1 flex-col gap-1.5">
                <Label htmlFor="memory-project">
                  {t("settings.profile.memory.project")}
                </Label>
                <select
                  id="memory-project"
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={projectId}
                  disabled={projectsLoading || projects.length === 0}
                  onChange={(event) => setProjectId(event.target.value)}
                >
                  <option value="">
                    {projectsLoading
                      ? t("common.loading")
                      : t("settings.profile.memory.selectProject")}
                  </option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                      {project.archived
                        ? ` (${t("settings.profile.memory.archived")})`
                        : ""}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}
          </div>
          <div className="flex shrink-0 gap-2">
            <Button variant="outline" onClick={() => void exportMemories()}>
              {t("settings.profile.memory.export")}
            </Button>
            <Button onClick={openCreate} disabled={projectRequired}>
              {t("settings.profile.memory.add")}
            </Button>
          </div>
        </div>

        <div className="min-h-20 py-3" aria-live="polite">
          {projectRequired ? (
            <p className="text-sm text-muted-foreground">
              {t("settings.profile.memory.projectRequired")}
            </p>
          ) : memoriesLoading ? (
            <p className="text-sm text-muted-foreground">
              {t("settings.profile.memory.loading")}
            </p>
          ) : memoriesError !== null ? (
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-sm text-destructive">
                {t("settings.profile.memory.loadError")}
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={() => void loadMemories()}
              >
                {t("settings.profile.memory.retry")}
              </Button>
            </div>
          ) : memories.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {t("settings.profile.memory.empty")}
            </p>
          ) : (
            <ul className="divide-y divide-border/60">
              {memories.map((memory) => (
                <li
                  key={memory.id}
                  className="flex flex-wrap items-center gap-x-3 gap-y-2 py-3"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm break-words">{memory.content}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {t(
                        memory.sourceType === "manual"
                          ? "settings.profile.memory.savedManually"
                          : memory.sourceType === "explicit"
                            ? "settings.profile.memory.savedExplicitly"
                            : "settings.profile.memory.savedAutomatically",
                      )}
                    </p>
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={deletingId === memory.id}
                      onClick={() =>
                        setEditor({
                          memory,
                          content: memory.content,
                          scope: memory.scope,
                          projectId: memory.projectId ?? projectId,
                        })
                      }
                    >
                      {t("settings.profile.memory.edit")}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      disabled={deletingId === memory.id}
                      onClick={() => void removeMemory(memory.id)}
                    >
                      {deletingId === memory.id
                        ? t("settings.profile.memory.deleting")
                        : t("common.delete")}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <SettingsRow
          destructive={true}
          label={t("settings.profile.memory.clearTitle")}
          description={t("settings.profile.memory.clearDescription")}
        >
          <Button
            size="sm"
            variant="outline"
            className="text-destructive hover:text-destructive"
            disabled={
              projectRequired || memoriesLoading || memories.length === 0
            }
            onClick={() => setClearOpen(true)}
          >
            {t("settings.profile.memory.clearAction")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog
        open={editor !== null}
        onOpenChange={(open) => {
          if (!open && !saving) setEditor(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {editor?.memory
                ? t("settings.profile.memory.editTitle")
                : t("settings.profile.memory.addTitle")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.profile.memory.editorDescription")}{" "}
              {t("settings.profile.memory.modelDisclosure")}
            </DialogDescription>
          </DialogHeader>
          {editor ? (
            <div className="grid gap-4">
              <div className="grid gap-1.5">
                <Label htmlFor="memory-content">
                  {t("settings.profile.memory.content")}
                </Label>
                <Textarea
                  id="memory-content"
                  className="min-h-24"
                  maxLength={300}
                  value={editor.content}
                  onChange={(event) =>
                    setEditor({ ...editor, content: event.target.value })
                  }
                />
                <p className="text-xs text-muted-foreground">
                  {t("settings.profile.memory.characterCount", {
                    count: editor.content.length,
                  })}
                </p>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="memory-editor-scope">
                  {t("settings.profile.memory.scope")}
                </Label>
                <select
                  id="memory-editor-scope"
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={editor.scope}
                  onChange={(event) =>
                    setEditor({
                      ...editor,
                      scope: event.target.value as MemoryScope,
                    })
                  }
                >
                  <option value="global">
                    {t("settings.profile.memory.global")}
                  </option>
                  <option value="project" disabled={projects.length === 0}>
                    {t("settings.profile.memory.project")}
                  </option>
                </select>
              </div>
              {editor.scope === "project" ? (
                <div className="grid gap-1.5">
                  <Label htmlFor="memory-editor-project">
                    {t("settings.profile.memory.project")}
                  </Label>
                  <select
                    id="memory-editor-project"
                    className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                    value={editor.projectId}
                    disabled={projectsLoading || projects.length === 0}
                    onChange={(event) =>
                      setEditor({ ...editor, projectId: event.target.value })
                    }
                  >
                    <option value="">
                      {t("settings.profile.memory.selectProject")}
                    </option>
                    {projects.map((project) => (
                      <option key={project.id} value={project.id}>
                        {project.name}
                        {project.archived
                          ? ` (${t("settings.profile.memory.archived")})`
                          : ""}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
            </div>
          ) : null}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={saving}
              onClick={() => setEditor(null)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              disabled={
                saving || !editor?.content.trim() || editorProjectRequired
              }
              onClick={() => void saveMemory()}
            >
              {saving ? t("common.saving") : t("common.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={clearOpen} onOpenChange={setClearOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.profile.memory.clearConfirmTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.profile.memory.clearConfirmDescription", {
                count: memories.length,
                scope: selectedScopeLabel,
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clearing}>
              {t("common.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={clearing}
              onClick={(event) => {
                event.preventDefault();
                void clearMemories();
              }}
            >
              {clearing
                ? t("settings.profile.memory.clearing")
                : t("settings.profile.memory.clearAction")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
