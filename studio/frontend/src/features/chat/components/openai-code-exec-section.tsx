// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Settings-sheet section for OpenAI shell-tool container management. Renders
 * only when the active provider is OpenAI cloud (api.openai.com) and the model
 * is gpt-5.5 / gpt-5.5-pro (the only families wired to the shell tool today).
 *
 * Controls:
 *   1. Default container idle-timeout (minutes). Persisted on the provider
 *      record; pre-fills the create dialog and feeds the chat-adapter's
 *      lazy-create path on a thread's first turn.
 *   2. Container picker for the active thread (an existing OpenAI container or
 *      auto-create per thread).
 *   3. Create-new-container inline form, with per-row refresh + delete.
 *
 * Persistence: TTL -> ExternalProviderConfig.openaiContainerTtlMinutes;
 * active container -> ThreadRecord.openaiCodeExecContainerId. No global stores;
 * the list is fetched on open / refresh and held in component state.
 */

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
import { TrashIcon, RefreshCwIcon, PlusIcon } from "lucide-react";
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

const DEFAULT_TTL_MINUTES = 20;
const TTL_MIN = 1;
const TTL_MAX = 20; // OpenAI hard cap on expires_after.minutes
// Re-fetch cadence while the section is mounted. OpenAI's TTL flips at minute
// granularity, so 30s drops an expired container's ACTIVE pill within half a
// minute without hammering /v1/containers.
const REFRESH_POLL_MS = 30_000;

function shortContainerId(id: string): string {
  // Mid-truncate keeps the "cntr_" prefix readable and still surfaces the
  // tail digits users sometimes copy off OpenAI's dashboard.
  if (id.length <= 18) return id;
  return `${id.slice(0, 12)}…${id.slice(-4)}`;
}

function isContainerRunning(c: OpenAIContainerSummary): boolean {
  // OpenAI reports `status: "running"` while idle TTL is valid and "expired"
  // afterward. Treat a missing status as running so older payloads without the
  // field don't false-positive.
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
  const [containers, setContainers] = useState<OpenAIContainerSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  // Ids deleted this session. A tombstoned id stays hidden for the page's
  // lifetime: OpenAI's /containers list can keep returning a freshly-deleted
  // id for an undocumented, variable time, and auto-reshowing it confuses more
  // than it helps. A page refresh resets the tombstone.
  const [tombstones, setTombstones] = useState<Set<string>>(() => new Set());
  // Ids optimistically inserted after a create but not yet confirmed by a
  // /v1/containers list. That endpoint is eventually consistent (a new
  // container can be absent for several seconds), so we render the row at once
  // with a "Creating" pill and drop it once a refresh sees the id.
  const [pendingIds, setPendingIds] = useState<Set<string>>(() => new Set());
  // Ref mirror so `refresh()` reads the current pending set without re-binding
  // when it changes (the callback is a useEffect dep).
  const pendingIdsRef = useRef<Set<string>>(pendingIds);
  useEffect(() => {
    pendingIdsRef.current = pendingIds;
  }, [pendingIds]);
  // One-shot follow-up refresh after a create, for when the server list lags
  // the create response by a few seconds. Tracked so we clear it on unmount.
  const pendingRetryRef = useRef<number | null>(null);
  // Target row for the delete-confirmation dialog. Held in state (not
  // window.confirm) so the dialog sits inside the settings sheet.
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

  // Hide just-deleted containers even if OpenAI's list still returns them.
  // Single chokepoint: every downstream view (sorted picker, auto-bind
  // candidate, all-containers list) derives from visibleContainers.
  const visibleContainers = useMemo(() => {
    if (tombstones.size === 0) return containers;
    return containers.filter((c) => !tombstones.has(c.id));
  }, [containers, tombstones]);

  // Newest-first by lastActiveAt so the dropdown default (auto-bind target)
  // shows up first.
  const sortedContainers = useMemo(
    () =>
      [...visibleContainers].sort(
        (a, b) => (b.lastActiveAt ?? 0) - (a.lastActiveAt ?? 0),
      ),
    [visibleContainers],
  );

  // First running container by lastActiveAt: the auto-bind target and what we
  // surface visually before Dexie catches up.
  const firstRunningContainer = useMemo(
    () => sortedContainers.find(isContainerRunning) ?? null,
    [sortedContainers],
  );

  // What the picker treats as "active" now. Decoupled from `activeContainerId`
  // (Dexie state) so the user sees the most-recent running container while the
  // auto-bind effect's async write propagates. If the Dexie-bound container
  // expired, fall back to the first running candidate; the stale-bind sweeper
  // clears Dexie shortly after.
  const boundContainer = useMemo(
    () => sortedContainers.find((c) => c.id === activeContainerId) ?? null,
    [sortedContainers, activeContainerId],
  );
  const displayedContainerId =
    (boundContainer && isContainerRunning(boundContainer)
      ? boundContainer.id
      : firstRunningContainer?.id) ?? null;

  const refresh = useCallback(async () => {
    if (!apiKey) return;
    setIsLoading(true);
    try {
      const list = await listOpenAIContainers({
        apiKey,
        baseUrl: provider.baseUrl || null,
      });
      const serverIds = new Set(list.map((c) => c.id));
      setContainers((prev) => {
        // Preserve optimistic inserts the server hasn't acknowledged yet so
        // they don't disappear on the reconciling refresh.
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
  }, [apiKey, provider.baseUrl]);

  // Fetch on mount (or provider change), then poll on a low cadence so an
  // expired container's ACTIVE pill clears without a manual refresh. Also
  // re-fetch when the tab regains visibility (sheet left open while idle).
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

  // Auto-bind the active thread to the most-recently-active container when the
  // thread has none and at least one container exists on the account. Sorting
  // by `lastActiveAt` matches what feels "most recent" to the user.
  //
  // `ensureThreadRecord` eagerly materializes the thread row so the bind lands
  // before the first message. This creates nothing at OpenAI (only a local
  // ThreadRecord), so a fresh OpenAI container is still not created until first
  // send.
  //
  // If no containers exist yet, this short-circuits: the picker shows an
  // empty-state hint and the chat-adapter mints the first container on send.
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
    // value is always a container id now; "Auto-create per thread" was removed
    // in favour of defaulting to the most-recently-active container. The
    // chat-adapter still handles the no-containers case (lazy-create on send).
    //
    // ensureThreadRecord eagerly materializes the thread row (modelType "base":
    // settings sheet is single-thread only) so the update lands before send.
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
    if (!apiKey) return;
    const name = createName.trim();
    if (!name) {
      toast.error("Container name is required");
      return;
    }
    // TTL inherits from the section-level "Idle timeout" control (no
    // per-container override). Read at submit time so a last-second TTL change
    // applies.
    const ttlMinutes =
      provider.openaiContainerTtlMinutes ?? DEFAULT_TTL_MINUTES;
    setCreating(true);
    try {
      const created = await createOpenAIContainer(
        { apiKey, baseUrl: provider.baseUrl || null },
        { name, ttlMinutes },
      );
      toast.success(`Created container ${name}`);
      setCreateName("");
      setCreateOpen(false);
      // Optimistic insert + "Creating" pill. The /v1/containers list is
      // eventually consistent and can omit the new container for seconds;
      // without this the row only appears on the next poll or manual refresh.
      setContainers((prev) =>
        prev.some((c) => c.id === created.id) ? prev : [created, ...prev],
      );
      setPendingIds((prev) => {
        if (prev.has(created.id)) return prev;
        const next = new Set(prev);
        next.add(created.id);
        return next;
      });
      // Follow-up refresh ~5s later to reconcile the optimistic row once
      // /v1/containers catches up. One shot; the poll covers a longer tail.
      if (pendingRetryRef.current != null) {
        window.clearTimeout(pendingRetryRef.current);
      }
      pendingRetryRef.current = window.setTimeout(() => {
        pendingRetryRef.current = null;
        void refresh();
      }, 5000);
      // Auto-bind the new container to the active thread. ensureThreadRecord
      // first so the bind lands even if no message has been sent yet.
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
      // Refresh even on failure: the request may have partially succeeded
      // (container created, response lost); a re-fetch keeps the picker in sync.
      await refresh();
    }
  };

  const confirmDelete = async () => {
    if (!apiKey || !pendingDelete) return;
    const { id, name } = pendingDelete;
    setDeleting(true);
    try {
      await deleteOpenAIContainer(
        { apiKey, baseUrl: provider.baseUrl || null },
        id,
      );
      // Tombstone the id so the picker hides it at once even if OpenAI's list
      // keeps returning it for a while.
      setTombstones((prev) => {
        if (prev.has(id)) return prev;
        const next = new Set(prev);
        next.add(id);
        return next;
      });
      // Clear any thread bindings pointing at the now-deleted id.
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
      // Always refresh so a stale list entry (deleted elsewhere or expired) is
      // purged even when the delete call errored.
      await refresh();
    }
  };

  const displayActiveId = displayedContainerId;

  return (
    <div className="flex flex-col gap-3 pt-1">
      {/* TTL */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <label
            htmlFor="openai-container-ttl"
            className="min-w-0 text-[0.8125rem] font-medium leading-[1.25] tracking-nav text-nav-fg"
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
          className="h-8 w-[72px] pl-3 text-sm tabular-nums"
        />
      </div>

      {/* Container list. Clicking a row binds it to the active thread; the
          ACTIVE pill marks which one (no separate picker). */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.6875rem] uppercase tracking-wider text-muted-foreground">
            Containers
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="-mr-1 h-6 w-6 p-0 text-muted-foreground"
            onClick={() => void refresh()}
            disabled={isLoading || !apiKey}
            aria-label="Refresh container list"
          >
            <RefreshCwIcon
              className={`size-3.5 ${isLoading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>
        {sortedContainers.length === 0 ? (
          // Quiet placeholder with the same muted border as row cards so an
          // empty section doesn't look like an active control. The first
          // container is lazy-created on first send and appears after refresh.
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
                  {/* min-w-0 + truncate keeps long container ids from
                      spilling under the trash button on narrow sheets. */}
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 truncate font-medium">
                        {c.name ?? "(unnamed)"}
                      </span>
                      {isPending ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-[0.5625rem] font-medium uppercase tracking-wider text-muted-foreground">
                          Creating
                        </span>
                      ) : isActive ? (
                        <span className="shrink-0 rounded-sm bg-primary/15 px-1 py-px text-[0.5625rem] font-medium uppercase tracking-wider text-primary">
                          Active
                        </span>
                      ) : statusLabel ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-[0.5625rem] font-medium uppercase tracking-wider text-muted-foreground">
                          {statusLabel}
                        </span>
                      ) : null}
                    </div>
                    <div
                      className="flex min-w-0 items-center gap-1.5 text-muted-foreground"
                      title={c.id}
                    >
                      <span className="min-w-0 truncate font-mono text-[0.6875rem]">
                        {shortContainerId(c.id)}
                      </span>
                      <span className="shrink-0 text-[0.625rem] uppercase tracking-wider">
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

      {/* Create new: inline single-row edit echoing a container card. TTL is
          inherited from the top "Idle timeout" control (no per-container
          override), keeping the form light. */}
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
                if (createName.trim() && !creating && apiKey) {
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
            disabled={creating || !createName.trim() || !apiKey}
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
          disabled={!apiKey}
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
