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

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
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

const AUTO_OPTION_VALUE = "__auto__";
const DEFAULT_TTL_MINUTES = 20;
const TTL_MIN = 1;
const TTL_MAX = 10080; // one week — matches backend bound

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
  const [createTtl, setCreateTtl] = useState<number>(
    provider.openaiContainerTtlMinutes ?? DEFAULT_TTL_MINUTES,
  );

  const thread = useLiveQuery(
    async () => (activeThreadId ? db.threads.get(activeThreadId) : undefined),
    [activeThreadId],
  );
  const activeContainerId = thread?.openaiCodeExecContainerId ?? null;

  // Containers sorted newest-first by lastActiveAt so the dropdown's
  // default (auto-bind target) shows up first.
  const sortedContainers = useMemo(
    () =>
      [...containers].sort(
        (a, b) => (b.lastActiveAt ?? 0) - (a.lastActiveAt ?? 0),
      ),
    [containers],
  );

  // What the dropdown should display right now. We decouple this from
  // `activeContainerId` (which is whatever is in Dexie) so the user
  // immediately sees the most-recent container by name when there is
  // no thread binding yet, rather than a "Selecting most recent…"
  // placeholder while the auto-bind effect's async write propagates
  // back through useLiveQuery. The auto-bind effect still writes the
  // bind to Dexie so the chat adapter sees it on send.
  const displayedContainerId =
    activeContainerId ?? sortedContainers[0]?.id ?? null;

  const refresh = useCallback(async () => {
    if (!apiKey) return;
    setIsLoading(true);
    try {
      const list = await listOpenAIContainers({
        apiKey,
        baseUrl: provider.baseUrl || null,
      });
      setContainers(list);
    } catch (err) {
      toast.error(
        `Failed to list containers: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    } finally {
      setIsLoading(false);
    }
  }, [apiKey, provider.baseUrl]);

  // Fetch once when the section mounts (or provider changes).
  useEffect(() => {
    void refresh();
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
    if (!activeThreadId || activeContainerId || containers.length === 0) {
      return;
    }
    const sorted = [...containers].sort(
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
  }, [activeThreadId, activeContainerId, containers]);

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
    setCreating(true);
    try {
      const created = await createOpenAIContainer(
        { apiKey, baseUrl: provider.baseUrl || null },
        { name, ttlMinutes: createTtl },
      );
      toast.success(`Created container ${name}`);
      setCreateName("");
      setCreateOpen(false);
      await refresh();
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
    }
  };

  const onDelete = async (id: string, name: string | null | undefined) => {
    if (!apiKey) return;
    if (
      !window.confirm(
        `Delete container ${name || id}? Threads using it will fall back to auto-create on their next turn.`,
      )
    ) {
      return;
    }
    try {
      await deleteOpenAIContainer(
        { apiKey, baseUrl: provider.baseUrl || null },
        id,
      );
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
      await refresh();
    } catch (err) {
      toast.error(
        `Delete failed: ${err instanceof Error ? err.message : "Unknown"}`,
      );
    }
  };

  return (
    <div className="flex flex-col gap-3 pt-1">
      {/* TTL */}
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor="openai-container-ttl"
          className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg"
        >
          New-container idle timeout (min)
        </label>
        <Input
          id="openai-container-ttl"
          type="number"
          min={TTL_MIN}
          max={TTL_MAX}
          value={ttlValue}
          onChange={(e) => onTtlChange(e.target.value)}
          className="h-8 w-24 text-sm"
        />
      </div>

      {/* Active container picker — visually emphasized so it reads as
          the primary control vs. the static list below. Accent
          background + ring outline distinguish it from the plain
          bordered list items beneath. */}
      <div className="flex flex-col gap-1.5 rounded-md border border-primary/30 bg-primary/5 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-semibold leading-[1.25] tracking-nav text-primary">
            Active for this thread
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => void refresh()}
            disabled={isLoading || !apiKey}
            aria-label="Refresh container list"
          >
            <RefreshCwIcon
              className={`size-3.5 ${isLoading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>
        {/* When no containers exist yet, render a disabled placeholder
            instead of the picker. The first one is created by the
            chat-adapter on first send (lazy-create) and will appear
            here after the next refresh. */}
        {sortedContainers.length === 0 ? (
          <div className="h-9 w-full rounded-md border border-primary/40 bg-background px-2 flex items-center text-sm text-muted-foreground">
            (none yet — will be created on first send)
          </div>
        ) : (
          <select
            value={displayedContainerId ?? sortedContainers[0].id}
            onChange={(e) => onPick(e.target.value)}
            disabled={!activeThreadId}
            className="h-9 w-full rounded-md border border-primary/40 bg-background px-2 text-sm font-medium shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          >
            {sortedContainers.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name ?? "(unnamed)"} · {c.id.slice(0, 14)}…
                {c.lastActiveAt ? ` · active ${ageLabel(c.lastActiveAt)}` : ""}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Container list with delete actions — labeled and visually
          quieter so it's clearly the "all containers, manage them"
          area rather than the active selector above. */}
      <div className="flex flex-col gap-1.5">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
          All containers
        </span>
        {isLoading && containers.length === 0 ? (
          <Skeleton className="h-16 w-full" />
        ) : containers.length > 0 ? (
          <ul className="flex flex-col gap-1 max-h-44 overflow-auto">
            {containers.map((c) => {
              const isActive = c.id === activeContainerId;
              return (
                <li
                  key={c.id}
                  className={`flex items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-xs ${
                    isActive
                      ? "border-primary/30 bg-primary/5"
                      : "border-border/60"
                  }`}
                >
                  <div className="flex min-w-0 flex-col">
                    <span className="truncate font-medium">
                      {c.name ?? "(unnamed)"}
                      {isActive ? (
                        <span className="ml-1.5 text-[10px] font-normal uppercase tracking-wider text-primary">
                          · active
                        </span>
                      ) : null}
                    </span>
                    <span className="text-muted-foreground">
                      {c.id} · TTL{" "}
                      {c.expiresAfterMinutes ?? DEFAULT_TTL_MINUTES}m
                    </span>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 text-destructive"
                    onClick={() => void onDelete(c.id, c.name)}
                    aria-label={`Delete container ${c.name ?? c.id}`}
                  >
                    <TrashIcon className="size-3.5" />
                  </Button>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-xs text-muted-foreground">
            No saved containers yet. Use auto-create or create a named
            one below.
          </p>
        )}
      </div>

      {/* Create new */}
      {createOpen ? (
        <div className="flex flex-col gap-2 rounded-md border border-border/60 p-2">
          <Input
            placeholder="Container name (e.g. data-analysis)"
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            className="h-8 text-sm"
          />
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={TTL_MIN}
              max={TTL_MAX}
              value={createTtl}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (!Number.isNaN(n))
                  setCreateTtl(Math.min(Math.max(n, TTL_MIN), TTL_MAX));
              }}
              className="h-8 w-24 text-sm"
              aria-label="Idle timeout in minutes"
            />
            <span className="text-xs text-muted-foreground">min idle</span>
            <div className="flex-1" />
            <Button
              size="sm"
              variant="ghost"
              className="h-7"
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
              className="h-7"
              onClick={() => void onCreate()}
              disabled={creating || !createName.trim() || !apiKey}
            >
              Create
            </Button>
          </div>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="h-8"
          onClick={() => {
            setCreateTtl(ttlValue);
            setCreateOpen(true);
          }}
          disabled={!apiKey}
        >
          <PlusIcon className="size-3.5 mr-1" />
          New container
        </Button>
      )}
    </div>
  );
}
