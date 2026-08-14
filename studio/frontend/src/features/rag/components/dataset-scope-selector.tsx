import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { listKnowledgeBases } from "../api/rag-api";
import type { KnowledgeBase } from "../types/rag";

export function DatasetScopeSelector({
  selectedIds,
  onChange,
  disabled = false,
}: {
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
}) {
  const [datasets, setDatasets] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void listKnowledgeBases(controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) setDatasets(items);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Datasets could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey]);

  const selected = new Set(selectedIds);
  function toggle(id: string, checked: boolean) {
    const next = new Set(selected);
    if (checked) next.add(id);
    else next.delete(id);
    onChange([...next]);
  }

  return (
    <div className="space-y-2" aria-busy={loading}>
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">Dataset scope</p>
          <p className="text-xs text-muted-foreground">
            Selected datasets are persisted on this Rag Platform project.
          </p>
        </div>
        {selectedIds.length > 0 ? (
          <span className="shrink-0 text-xs text-muted-foreground">
            {selectedIds.length} selected
          </span>
        ) : null}
      </div>
      <div className="max-h-48 overflow-y-auto rounded-xl border border-border bg-background p-2">
        {loading ? (
          <p className="px-2 py-4 text-center text-sm text-muted-foreground">
            Loading datasets…
          </p>
        ) : error ? (
          <div className="flex flex-col items-center gap-2 px-3 py-4 text-center">
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => setReloadKey((value) => value + 1)}
            >
              Retry
            </Button>
          </div>
        ) : datasets.length === 0 ? (
          <p className="px-2 py-4 text-center text-sm text-muted-foreground">
            No datasets are available. Create one in Documents first.
          </p>
        ) : (
          datasets.map((dataset) => (
            <label
              key={dataset.id}
              className="flex cursor-pointer items-center gap-3 rounded-lg px-2 py-2 hover:bg-muted/60 has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-50"
            >
              <Checkbox
                checked={selected.has(dataset.id)}
                disabled={disabled}
                onCheckedChange={(checked) =>
                  toggle(dataset.id, checked === true)
                }
                aria-label={`Use dataset ${dataset.name}`}
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">
                  {dataset.name}
                </span>
                <span className="block text-xs text-muted-foreground">
                  {dataset.documentCount ?? 0} documents
                </span>
              </span>
            </label>
          ))
        )}
      </div>
    </div>
  );
}
