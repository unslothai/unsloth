import { FileDatabaseIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { DatasetScopeSelector } from "@/features/rag/components/dataset-scope-selector";
import {
  isPlatformApiError,
  isPlatformChatPersistenceEnabled,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  getPlatformChatDatasetScope,
  updatePlatformChatDatasetScope,
} from "../api/platform-chat-adapter";

function scopeErrorMessage(cause: unknown): string {
  if (isPlatformApiError(cause)) {
    if (cause.httpStatus === 403) {
      return "You do not have permission to change this chat's sources.";
    }
    if (cause.code === "CLIENT_TIMEOUT") {
      return "Chat sources timed out. Try again.";
    }
  }
  return cause instanceof Error
    ? cause.message
    : "Chat sources could not be loaded.";
}

export function PlatformChatSourcesButton({
  projectId,
}: {
  projectId?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [initialIds, setInitialIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const saveAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void getPlatformChatDatasetScope(projectId, controller.signal)
      .then((chat) => {
        if (controller.signal.aborted) return;
        const ids = chat.datasetIds ?? [];
        setSelectedIds(ids);
        setInitialIds(ids);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(scopeErrorMessage(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, projectId, reloadKey]);

  useEffect(
    () => () => {
      saveAbortRef.current?.abort();
    },
    [],
  );

  if (!isPlatformChatPersistenceEnabled()) return null;

  const dirty = selectedIds.join("\0") !== initialIds.join("\0");

  async function save() {
    if (!dirty || saving) return;
    saveAbortRef.current?.abort();
    const controller = new AbortController();
    saveAbortRef.current = controller;
    setSaving(true);
    setError(null);
    try {
      const chat = await updatePlatformChatDatasetScope(
        projectId,
        selectedIds,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      const ids = chat.datasetIds ?? [];
      setSelectedIds(ids);
      setInitialIds(ids);
      toast.success("Chat sources saved.");
    } catch (cause) {
      if (!controller.signal.aborted) {
        const message = scopeErrorMessage(cause);
        setError(message);
        toast.error("Chat sources could not be saved", {
          description: message,
        });
      }
    } finally {
      if (!controller.signal.aborted) setSaving(false);
      if (saveAbortRef.current === controller) saveAbortRef.current = null;
    }
  }

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) saveAbortRef.current?.abort();
      }}
    >
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              className={cn(
                "relative flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:text-white",
                open && "bg-nav-surface-hover text-foreground",
              )}
              aria-label="Manage chat sources"
              aria-pressed={open}
            >
              <HugeiconsIcon
                icon={FileDatabaseIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {initialIds.length > 0 ? (
                <span
                  aria-hidden={true}
                  className="absolute right-1 top-1 size-1.5 rounded-full bg-primary ring-2 ring-background"
                />
              ) : null}
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          Chat sources
        </TooltipContent>
      </Tooltip>
      <PopoverContent align="end" sideOffset={6} className="w-80 gap-4 p-4">
        <PopoverHeader>
          <PopoverTitle>Chat sources</PopoverTitle>
          <PopoverDescription>
            Choose which document collections this chat may retrieve from.
          </PopoverDescription>
        </PopoverHeader>

        {loading ? (
          <div
            className="rounded-xl border border-border px-3 py-8 text-center text-sm text-muted-foreground"
            aria-busy="true"
          >
            Loading chat sources…
          </div>
        ) : error ? (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-border px-3 py-6 text-center">
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
        ) : (
          <>
            <DatasetScopeSelector
              selectedIds={selectedIds}
              onChange={setSelectedIds}
              disabled={saving}
            />
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground" aria-live="polite">
                {initialIds.length > 0
                  ? `${initialIds.length} source${initialIds.length === 1 ? "" : "s"} active`
                  : "No sources active"}
              </p>
              <Button
                type="button"
                size="sm"
                disabled={!dirty || saving}
                onClick={() => void save()}
              >
                {saving ? "Saving…" : "Save"}
              </Button>
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
