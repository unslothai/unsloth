// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** OpenAI shell-tool container management (OpenAI cloud, gpt-5.5 models only). */

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
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
import { Input } from "@/components/ui/input";
import { TrashIcon, PlusIcon } from "lucide-react";
import {
  createOpenAIContainer,
  deleteOpenAIContainer,
  listOpenAIContainers,
  type OpenAIContainerSummary,
} from "../api/openai-containers";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import type { ExternalProviderConfig } from "../external-providers";
import { ensureThreadRecord } from "../runtime-provider";
import { InfoHint } from "@/components/ui/info-hint";
import {
  getStoredChatThread,
  listStoredChatThreads,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import { RefreshGlyph } from "@/lib/refresh-icon";

const DEFAULT_TTL_MINUTES = 20;
const TTL_MIN = 1;
const TTL_MAX = 20; // OpenAI hard cap on expires_after.minutes
// OpenAI TTL flips at minute granularity, so 30s is enough.
const REFRESH_POLL_MS = 30_000;

function shortContainerId(id: string): string {
  if (id.length <= 18) return id;
  return `${id.slice(0, 12)}…${id.slice(-4)}`;
}

function isContainerRunning(c: OpenAIContainerSummary): boolean {
  // A missing status counts as running so older payloads do not false-positive.
  return c.status == null || c.status === "running";
}

interface OpenAICodeExecSectionProps {
  provider: ExternalProviderConfig;
  apiKey: string | null;
  activeThreadId: string | null;
  onProviderChange: (provider: ExternalProviderConfig) => void;
}

export function OpenAICodeExecSection({
  provider,
  apiKey,
  activeThreadId,
  onProviderChange,
}: OpenAICodeExecSectionProps) {

  const hasCredential = Boolean(apiKey || provider.hasApiKey);
  const [containers, setContainers] = useState<OpenAIContainerSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  // OpenAI's list can keep returning a deleted id for a while; hide it for the page's lifetime.
  const [tombstones, setTombstones] = useState<Set<string>>(() => new Set());
  // Optimistic inserts not yet seen by the eventually consistent list endpoint.
  const [pendingIds, setPendingIds] = useState<Set<string>>(() => new Set());
  // Ref mirror so refresh() does not re-bind when pending changes (it is an effect dep).
  const pendingIdsRef = useRef<Set<string>>(pendingIds);
  useEffect(() => {
    pendingIdsRef.current = pendingIds;
  }, [pendingIds]);
  const pendingRetryRef = useRef<number | null>(null);
  const [pendingDelete, setPendingDelete] =
    useState<OpenAIContainerSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [activeContainerId, setActiveContainerId] = useState<string | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    async function loadActiveContainer() {
      if (!activeThreadId) {
        setActiveContainerId(null);
        return;
      }
      const thread = await getStoredChatThread(activeThreadId).catch(
        () => undefined,
      );
      if (!cancelled) {
        setActiveContainerId(thread?.openaiCodeExecContainerId ?? null);
      }
    }
    void loadActiveContainer();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, loadActiveContainer);
    return () => {
      cancelled = true;
      window.removeEventListener(
        CHAT_HISTORY_UPDATED_EVENT,
        loadActiveContainer,
      );
    };
  }, [activeThreadId]);

  const visibleContainers = useMemo(() => {
    if (tombstones.size === 0) return containers;
    return containers.filter((c) => !tombstones.has(c.id));
  }, [containers, tombstones]);

  const sortedContainers = useMemo(
    () =>
      [...visibleContainers].sort(
        (a, b) => (b.lastActiveAt ?? 0) - (a.lastActiveAt ?? 0),
      ),
    [visibleContainers],
  );

  const firstRunningContainer = useMemo(
    () => sortedContainers.find(isContainerRunning) ?? null,
    [sortedContainers],
  );

  // Decoupled from Dexie state so the auto-bind write can propagate; expired falls back.
  const boundContainer = useMemo(
    () => sortedContainers.find((c) => c.id === activeContainerId) ?? null,
    [sortedContainers, activeContainerId],
  );
  const displayedContainerId =
    (boundContainer && isContainerRunning(boundContainer)
      ? boundContainer.id
      : firstRunningContainer?.id) ?? null;

  const refresh = useCallback(async () => {
    if (!hasCredential) return;
    setIsLoading(true);
    try {
      const list = await listOpenAIContainers({
        providerId: provider.id,

        apiKey,
        baseUrl: provider.baseUrl || null,
      });
      const serverIds = new Set(list.map((c) => c.id));
      setContainers((prev) => {
        const orphans = prev.filter(
          (c) => !serverIds.has(c.id) && pendingIdsRef.current.has(c.id),
        );
        return orphans.length > 0 ? [...orphans, ...list] : list;
      });
      setPendingIds((prev) => {
        if (prev.size === 0) return prev;
        const next = new Set(prev);
        let changed = false;
        for (const id of serverIds) {
          if (next.delete(id)) changed = true;
        }
        return changed ? next : prev;
      });
    } catch (err) {
      toast.error(
        `Failed to list containers: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    } finally {
      setIsLoading(false);
    }
  }, [apiKey, hasCredential, provider.baseUrl, provider.id]);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        void refresh();
      }
    }, REFRESH_POLL_MS);
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
      if (pendingRetryRef.current != null) {
        window.clearTimeout(pendingRetryRef.current);
        pendingRetryRef.current = null;
      }
    };
  }, [refresh]);

  // Auto-bind an unbound thread to the newest container; nothing is created at OpenAI here.
  useEffect(() => {
    if (
      !activeThreadId ||
      activeContainerId ||
      visibleContainers.length === 0
    ) {
      return;
    }
    const sorted = [...visibleContainers].sort(
      (a, b) => (b.lastActiveAt ?? 0) - (a.lastActiveAt ?? 0),
    );
    const candidate = sorted[0];
    if (!candidate) return;
    void (async () => {
      try {
        await ensureThreadRecord({
          threadId: activeThreadId,
          modelType: "base",
        });
        await updateStoredChatThread(activeThreadId, {
          openaiCodeExecContainerId: candidate.id,
        });
      } catch {
        // Best-effort; the chat-adapter will inherit/create on send.
      }
    })();
  }, [activeThreadId, activeContainerId, visibleContainers]);

  const ttlValue = provider.openaiContainerTtlMinutes ?? DEFAULT_TTL_MINUTES;

  const onTtlChange = (raw: string) => {
    const n = parseInt(raw, 10);
    if (Number.isNaN(n)) return;
    const clamped = Math.min(Math.max(n, TTL_MIN), TTL_MAX);
    onProviderChange({ ...provider, openaiContainerTtlMinutes: clamped });
  };

  const onPick = async (value: string) => {
    if (!activeThreadId || !value) return;
    try {
      await ensureThreadRecord({ threadId: activeThreadId, modelType: "base" });
      const updated = await updateStoredChatThread(activeThreadId, {
        openaiCodeExecContainerId: value,
      });
      if (!updated) {
        toast.error("Could not update thread.");
      }
    } catch (err) {
      toast.error(
        `Could not update thread: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    }
  };

  const onCreate = async () => {
    if (!hasCredential) return;
    const name = createName.trim();
    if (!name) {
      toast.error("Container name is required");
      return;
    }
    const ttlMinutes =
      provider.openaiContainerTtlMinutes ?? DEFAULT_TTL_MINUTES;
    setCreating(true);
    try {
      const created = await createOpenAIContainer(
        { providerId: provider.id, apiKey, baseUrl: provider.baseUrl || null },
        { name, ttlMinutes },
      );
      toast.success(`Created container ${name}`);
      setCreateName("");
      setCreateOpen(false);
      setContainers((prev) =>
        prev.some((c) => c.id === created.id) ? prev : [created, ...prev],
      );
      setPendingIds((prev) => {
        if (prev.has(created.id)) return prev;
        const next = new Set(prev);
        next.add(created.id);
        return next;
      });
      if (pendingRetryRef.current != null) {
        window.clearTimeout(pendingRetryRef.current);
      }
      pendingRetryRef.current = window.setTimeout(() => {
        pendingRetryRef.current = null;
        void refresh();
      }, 5000);
      if (activeThreadId) {
        try {
          await ensureThreadRecord({
            threadId: activeThreadId,
            modelType: "base",
          });
          await updateStoredChatThread(activeThreadId, {
            openaiCodeExecContainerId: created.id,
          });
        } catch {
          /* best-effort; toast above already confirmed creation */
        }
      }
    } catch (err) {
      toast.error(
        `Create failed: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    } finally {
      setCreating(false);
      // Refresh even on failure: the create may have succeeded with the response lost.
      await refresh();
    }
  };

  const confirmDelete = async () => {
    if (!hasCredential || !pendingDelete) return;
    const { id, name } = pendingDelete;
    setDeleting(true);
    try {
      await deleteOpenAIContainer(
        { providerId: provider.id, apiKey, baseUrl: provider.baseUrl || null },
        id,
      );
      setTombstones((prev) => {
        if (prev.has(id)) return prev;
        const next = new Set(prev);
        next.add(id);
        return next;
      });
      const affected = (
        await listStoredChatThreads({ includeArchived: true })
      ).filter((t) => t.openaiCodeExecContainerId === id);
      await Promise.all(
        affected.map((t) =>
          updateStoredChatThread(t.id, { openaiCodeExecContainerId: null }),
        ),
      );
      toast.success(`Deleted container ${name || id}`);
    } catch (err) {
      toast.error(
        `Delete failed: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    } finally {
      setDeleting(false);
      setPendingDelete(null);
      await refresh();
    }
  };

  const displayActiveId = displayedContainerId;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <label
            htmlFor="openai-container-ttl"
            className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg"
          >
            Idle timeout
          </label>
          <InfoHint>
            Minutes a newly-created container stays alive between calls.
            OpenAI caps this at 20.
          </InfoHint>
        </div>
        <Input
          id="openai-container-ttl"
          type="number"
          min={TTL_MIN}
          max={TTL_MAX}
          value={ttlValue}
          onChange={(e) => onTtlChange(e.target.value)}
          className="h-8 w-[calc(72px*var(--ui-space-scale,1))] pl-3 text-sm tabular-nums"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-ui-11 uppercase tracking-wider text-muted-foreground">
            Containers
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="-mr-1 h-6 w-6 p-0 text-muted-foreground"
            onClick={() => void refresh()}
            disabled={isLoading || !hasCredential}
            aria-label="Refresh container list"
          >
            <RefreshGlyph
              className={`size-3.5 ${isLoading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>
        {sortedContainers.length === 0 ? (
          <div className="flex h-9 w-full items-center rounded-md border border-dashed border-border/60 bg-muted/20 px-2 text-xs text-muted-foreground">
            None yet - one will be created on first send.
          </div>
        ) : (
          <ul className="flex max-h-52 flex-col gap-1 overflow-auto">
            {sortedContainers.map((c) => {
              const running = isContainerRunning(c);
              const isActive = running && c.id === displayActiveId;
              const isPending = pendingIds.has(c.id);
              const ttlMinutes = c.expiresAfterMinutes ?? DEFAULT_TTL_MINUTES;
              const canActivate =
                activeThreadId != null && !isActive && running;
              const statusLabel = !running ? (c.status ?? "expired") : null;
              return (
                <li
                  key={c.id}
                  className={`flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs transition-colors ${
                    isActive
                      ? "border-ring-strong bg-primary/5"
                      : "border-border/60 hover:bg-muted/40"
                  } ${canActivate ? "cursor-pointer" : ""} ${
                    running ? "" : "opacity-60"
                  }`}
                  onClick={() => {
                    if (canActivate) void onPick(c.id);
                  }}
                  onKeyDown={(e) => {
                    if (!canActivate) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      void onPick(c.id);
                    }
                  }}
                  tabIndex={canActivate ? 0 : undefined}
                  role={canActivate ? "button" : undefined}
                  aria-pressed={isActive}
                  title={
                    canActivate
                      ? "Use this container for the active thread"
                      : !running
                        ? `Container is ${statusLabel}`
                        : undefined
                  }
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 truncate font-medium">
                        {c.name ?? "(unnamed)"}
                      </span>
                      {isPending ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-ui-9 font-medium uppercase tracking-wider text-muted-foreground">
                          Creating
                        </span>
                      ) : isActive ? (
                        <span className="shrink-0 rounded-sm bg-primary/15 px-1 py-px text-ui-9 font-medium uppercase tracking-wider text-primary">
                          Active
                        </span>
                      ) : statusLabel ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-ui-9 font-medium uppercase tracking-wider text-muted-foreground">
                          {statusLabel}
                        </span>
                      ) : null}
                    </div>
                    <div
                      className="flex min-w-0 items-center gap-1.5 text-muted-foreground"
                      title={c.id}
                    >
                      <span className="min-w-0 truncate font-mono text-ui-11">
                        {shortContainerId(c.id)}
                      </span>
                      <span className="shrink-0 text-ui-10 uppercase tracking-wider">
                        · {ttlMinutes}m
                      </span>
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 shrink-0 p-0 text-muted-foreground hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      setPendingDelete(c);
                    }}
                    aria-label={`Delete container ${c.name ?? c.id}`}
                  >
                    <TrashIcon className="size-3.5" />
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {createOpen ? (
        <div className="flex items-center gap-1 rounded-md border border-border/60 bg-muted/20 px-1.5 py-1">
          <Input
            autoFocus
            placeholder="Name"
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                if (createName.trim() && !creating && hasCredential) {
                  void onCreate();
                }
              } else if (e.key === "Escape") {
                e.preventDefault();
                setCreateOpen(false);
                setCreateName("");
              }
            }}
            className="h-7 min-w-0 flex-1 border-0 bg-transparent px-1.5 text-xs shadow-none focus-visible:ring-0"
          />
          <Button
            size="sm"
            variant="ghost"
            className="h-7 shrink-0 px-2 text-xs"
            onClick={() => {
              setCreateOpen(false);
              setCreateName("");
            }}
            disabled={creating}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            className="h-7 shrink-0 px-3 text-xs"
            onClick={() => void onCreate()}
            disabled={creating || !createName.trim() || !hasCredential}
          >
            {creating ? "Creating…" : "Create"}
          </Button>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="h-8"
          onClick={() => setCreateOpen(true)}
          disabled={!hasCredential}
        >
          <PlusIcon className="size-3.5 mr-1" />
          New container
        </Button>
      )}

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(nextOpen) => {
          if (!nextOpen && deleting) return;
          if (!nextOpen) setPendingDelete(null);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete{" "}
              <span className="font-mono">
                {pendingDelete?.name ?? "container"}
              </span>
              ?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Threads using this container will fall back to auto-create on
              their next turn. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(e) => {
                e.preventDefault();
                void confirmDelete();
              }}
            >
              {deleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
