import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  getStoredChatProject,
  updateStoredChatProject,
} from "@/features/chat/utils/chat-history-storage";
import { toast } from "@/lib/toast";
import { DatasetScopeSelector } from "./dataset-scope-selector";

/** Rag Platform Chat.dataset_ids editor for every Session in this project. */
export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [initialIds, setInitialIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void getStoredChatProject(projectId)
      .then((project) => {
        if (cancelled) return;
        if (!project) throw new Error("Project was not found.");
        const ids = project.datasetIds ?? [];
        setSelectedIds(ids);
        setInitialIds(ids);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(
            cause instanceof Error ? cause.message : "Project failed to load.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, reloadKey]);

  const dirty = selectedIds.join("\0") !== initialIds.join("\0");

  async function save() {
    if (!dirty || saving) return;
    setSaving(true);
    try {
      const project = await updateStoredChatProject(projectId, {
        datasetIds: selectedIds,
      });
      const saved = project.datasetIds ?? [];
      setSelectedIds(saved);
      setInitialIds(saved);
      toast.success("Dataset scope saved.");
    } catch (cause) {
      toast.error("Failed to save dataset scope", {
        description: cause instanceof Error ? cause.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="mt-8 rounded-[26px] bg-muted/30 px-6 py-10 text-center text-sm text-muted-foreground">
        Loading project scope…
      </div>
    );
  }
  if (error) {
    return (
      <div className="mt-8 flex flex-col items-center gap-3 rounded-[26px] bg-muted/30 px-6 py-10 text-center">
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
        <Button
          variant="outline"
          onClick={() => setReloadKey((key) => key + 1)}
        >
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="mt-8 space-y-4 rounded-[26px] bg-muted/30 px-6 py-5">
      <DatasetScopeSelector
        selectedIds={selectedIds}
        onChange={setSelectedIds}
        disabled={saving}
      />
      <div className="flex items-center justify-between gap-4">
        <p className="text-xs text-muted-foreground">
          This scope is server-backed and follows you across browsers.
        </p>
        <Button
          type="button"
          disabled={!dirty || saving}
          onClick={() => void save()}
        >
          {saving ? "Saving…" : "Save scope"}
        </Button>
      </div>
    </div>
  );
}
