// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Settings-sheet section for OpenAI shell-tool container management.
 * Renders only when:
 *   - active provider is OpenAI cloud (api.openai.com base URL), AND
 *   - the active model is gpt-5.5 or gpt-5.5-pro (the only families
 *     where the shell tool is wired through today).
 *
 * Surfaces three controls:
 *   1. Default container idle-timeout (minutes). Persists on the
 *      provider record; pre-fills the create dialog and is used by
 *      the chat-adapter's lazy-create path on the first turn of a
 *      thread.
 *   2. Container picker for the *active thread* — pick any of the
 *      user's existing OpenAI containers, or "Auto-create per thread"
 *      (default; lets the auto-create path manage it).
 *   3. Create-new-container inline form. Refresh + delete actions
 *      per row.
 *
 * State persistence:
 *   - TTL → ExternalProviderConfig.openaiContainerTtlMinutes
 *   - Active container for this thread → ThreadRecord.openaiCodeExecContainerId
 *
 * No new global stores — list is fetched on open / refresh and held
 * in component state.
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
import { db } from "../db";
import type { ExternalProviderConfig } from "../external-providers";
import { useLiveQuery } from "../db";
import { ensureThreadRecord } from "../runtime-provider";
import { InfoHint } from "../chat-settings-sheet";

const AUTO_OPTION_VALUE = "__auto__";
const DEFAULT_TTL_MINUTES = 20;
const TTL_MIN = 1;
const TTL_MAX = 20; // OpenAI hard cap on expires_after.minutes
// Cadence for re-fetching the container list while the section is
// mounted. OpenAI's container TTL flips at minute granularity, so 30s
// is fast enough that an expired container loses its ACTIVE pill within
// half a minute without hammering /v1/containers.
const REFRESH_POLL_MS = 30_000;

function ageLabel(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "";
  const ageSec = Math.max(0, Math.floor(Date.now() / 1000) - epochSeconds);
  if (ageSec < 60) return `${ageSec}s ago`;
  const ageMin = Math.floor(ageSec / 60);
  if (ageMin < 60) return `${ageMin}m ago`;
  const ageHr = Math.floor(ageMin / 60);
  if (ageHr < 48) return `${ageHr}h ago`;
  const ageDay = Math.floor(ageHr / 24);
  return `${ageDay}d ago`;
}

function shortContainerId(id: string): string {
  // Mid-truncate keeps the "cntr_" prefix readable and still surfaces the
  // tail digits users sometimes copy off OpenAI's dashboard.
  if (id.length <= 18) return id;
  return `${id.slice(0, 12)}…${id.slice(-4)}`;
}

function isContainerRunning(c: OpenAIContainerSummary): boolean {
  // OpenAI's containers API reports `status: "running"` while idle TTL is
  // valid and `status: "expired"` once the idle window has passed. Treat
  // a missing status as running so we don't false-positive on any older
  // payloads that didn't include the field.
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
  // Ids that have been deleted in this session. Once tombstoned, an id
  // stays hidden from the picker for the lifetime of the page — OpenAI's
  // /containers list can keep returning a freshly-deleted id for an
  // undocumented and variable amount of time, and an automatic re-show
  // creates more confusion than it solves. Refreshing the page resets
  // the tombstone naturally.
  const [tombstones, setTombstones] = useState<Set<string>>(() => new Set());
  // Ids optimistically inserted after a successful create but not yet
  // confirmed by a /v1/containers list response. OpenAI's list endpoint
  // is eventually consistent — a freshly-created container can be absent
  // for several seconds. We render the row immediately with a "Creating"
  // pill, then drop it from this set once a refresh sees the id.
  const [pendingIds, setPendingIds] = useState<Set<string>>(() => new Set());
  // Ref mirror so `refresh()` can read the current pending set without
  // re-binding when it changes (the callback is in a useEffect dep).
  const pendingIdsRef = useRef<Set<string>>(pendingIds);
  useEffect(() => {
    pendingIdsRef.current = pendingIds;
  }, [pendingIds]);
  // One-shot follow-up refresh scheduled after a create, to catch the
  // common case where the server list lags the create response by a few
  // seconds. Tracked so we can clear it on unmount.
  const pendingRetryRef = useRef<number | null>(null);
  // Target row for the destructive confirmation dialog. Held in state
  // (rather than blocking with window.confirm) so the dialog sits inside
  // the settings sheet instead of a native browser alert.
  const [pendingDelete, setPendingDelete] =
    useState<OpenAIContainerSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  const thread = useLiveQuery(
    async () => (activeThreadId ? db.threads.get(activeThreadId) : undefined),
    [activeThreadId],
  );
  const activeContainerId = thread?.openaiCodeExecContainerId ?? null;

  // Hide just-deleted containers even if OpenAI's list still returns them.
  // This is the single chokepoint — every downstream view (sorted picker,
  // auto-bind candidate, all-containers list) derives from visibleContainers.
  const visibleContainers = useMemo(() => {
    if (tombstones.size === 0) return containers;
    return containers.filter((c) => !tombstones.has(c.id));
  }, [containers, tombstones]);

  // Containers sorted newest-first by lastActiveAt so the dropdown's
  // default (auto-bind target) shows up first.
  const sortedContainers = useMemo(
    () =>
      [...visibleContainers].sort(
        (a, b) => (b.lastActiveAt ?? 0) - (a.lastActiveAt ?? 0),
      ),
    [visibleContainers],
  );

  // First running container by lastActiveAt — the auto-bind target and
  // also what we surface visually before Dexie catches up.
  const firstRunningContainer = useMemo(
    () => sortedContainers.find(isContainerRunning) ?? null,
    [sortedContainers],
  );

  // What the picker should treat as "active" right now. We decouple
  // this from `activeContainerId` (Dexie state) so the user immediately
  // sees the most-recent running container while the auto-bind effect's
  // async write propagates. If the Dexie-bound container has since
  // expired, fall back to the first running candidate — the stale-bind
  // sweeper below will clear Dexie shortly after.
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
        // Preserve optimistic inserts the server hasn't acknowledged
        // yet so they don't disappear on the reconciling refresh.
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

  // Fetch once when the section mounts (or provider changes), then
  // poll on a low cadence so an expired container's ACTIVE pill clears
  // without the user clicking the refresh button. Also re-fetch when
  // the tab regains visibility — covers the common case of leaving the
  // sheet open across a long idle period.
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

  // Auto-bind the active thread to the most-recently-active container
  // whenever the thread has none set and at least one container exists
  // on the user's OpenAI account. Sorting by `lastActiveAt` matches
  // what feels "most recent" from the user's perspective.
  //
  // We eagerly materialize the thread row via `ensureThreadRecord` so
  // the bind actually lands in Dexie before the user has sent a first
  // message. This does NOT create anything at OpenAI — only a local
  // ThreadRecord — so it does not bypass the user's expectation that
  // a fresh OpenAI container is not created until first send.
  //
  // If `containers` is empty (no OpenAI containers exist yet), this
  // effect short-circuits: the picker renders an empty-state hint and
  // the chat-adapter's lazy-create path will mint the first container
  // on first send.
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
        await db.threads.update(activeThreadId, {
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
    // value is always a container id now — the "Auto-create per thread"
    // option has been removed in favour of always defaulting to the
    // most-recently-active container. The chat-adapter still handles
    // the no-containers-exist case (lazy-create on first send).
    //
    // ensureThreadRecord materializes the thread row eagerly (modelType
    // "base" — settings sheet is single-thread-mode only) so the update
    // actually lands when the user hasn't sent a message yet.
    try {
      await ensureThreadRecord({ threadId: activeThreadId, modelType: "base" });
      const affected = await db.threads.update(activeThreadId, {
        openaiCodeExecContainerId: value,
      });
      if (affected === 0) {
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
    // TTL inherits from the section-level "Idle timeout" control —
    // there is no per-container override on the form. Read it at
    // submit time so a last-second change to the TTL row applies.
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
      // Optimistic insert + "Creating" pill. OpenAI's /v1/containers
      // list endpoint is eventually consistent and can omit the new
      // container for several seconds — without this, the row only
      // shows up on the next 30s poll or a manual refresh.
      setContainers((prev) =>
        prev.some((c) => c.id === created.id) ? prev : [created, ...prev],
      );
      setPendingIds((prev) => {
        if (prev.has(created.id)) return prev;
        const next = new Set(prev);
        next.add(created.id);
        return next;
      });
      // Follow-up refresh ~5s later to reconcile the optimistic row
      // with the server's view once /v1/containers catches up. One
      // shot; the regular poll covers any longer tail.
      if (pendingRetryRef.current != null) {
        window.clearTimeout(pendingRetryRef.current);
      }
      pendingRetryRef.current = window.setTimeout(() => {
        pendingRetryRef.current = null;
        void refresh();
      }, 5000);
      // Auto-bind the just-created container to the active thread.
      // ensureThreadRecord first so the bind lands even when the user
      // creates a container before sending the first message — without
      // it, db.threads.update silently affects 0 rows and the chat
      // adapter falls back to cross-thread inheritance / lazy-create,
      // which can pick a stale container that fails with "container
      // does not exist" on the first turn.
      if (activeThreadId) {
        try {
          await ensureThreadRecord({
            threadId: activeThreadId,
            modelType: "base",
          });
          await db.threads.update(activeThreadId, {
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
      // server-side (created container, lost response), and a re-fetch
      // keeps the picker in sync with OpenAI's actual state.
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
      // Tombstone the id so the picker hides it immediately even if
      // OpenAI's list keeps returning it for a while.
      setTombstones((prev) => {
        if (prev.has(id)) return prev;
        const next = new Set(prev);
        next.add(id);
        return next;
      });
      // Clear any thread bindings pointing at the now-deleted id.
      const affected = await db.threads
        .filter((t) => t.openaiCodeExecContainerId === id)
        .toArray();
      await Promise.all(
        affected.map((t) =>
          db.threads.update(t.id, { openaiCodeExecContainerId: null }),
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
      // Always refresh so a stale list entry (e.g. container deleted
      // elsewhere, or already expired) is purged from the UI even when
      // the delete call itself errored.
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
            className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg"
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
          className="h-8 w-14 px-2 text-center text-sm tabular-nums"
        />
      </div>

      {/* Single container list. Clicking a row binds it to the active
          thread and the ACTIVE pill marks which one — no separate
          picker needed. */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
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
          // Quiet placeholder with the same muted border as row cards
          // so an empty section doesn't masquerade as an active control.
          // The first container is minted by the chat-adapter on first
          // send (lazy-create) and appears here after the next refresh.
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
                      ? "border-primary/30 bg-primary/5"
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
                  {/* min-w-0 + truncate keeps long OpenAI container ids
                      from spilling under the trash button on narrow
                      settings sheets. */}
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 truncate font-medium">
                        {c.name ?? "(unnamed)"}
                      </span>
                      {isPending ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
                          Creating
                        </span>
                      ) : isActive ? (
                        <span className="shrink-0 rounded-sm bg-primary/15 px-1 py-px text-[9px] font-medium uppercase tracking-wider text-primary">
                          Active
                        </span>
                      ) : statusLabel ? (
                        <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
                          {statusLabel}
                        </span>
                      ) : null}
                    </div>
                    <div
                      className="flex min-w-0 items-center gap-1.5 text-muted-foreground"
                      title={c.id}
                    >
                      <span className="min-w-0 truncate font-mono text-[11px]">
                        {shortContainerId(c.id)}
                      </span>
                      <span className="shrink-0 text-[10px] uppercase tracking-wider">
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

      {/* Create new — inline single-row edit that visually echoes a
          container card. TTL is inherited from the section's top
          "Idle timeout" control (no per-container override), which
          keeps the form light and avoids a duplicated input. */}
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
