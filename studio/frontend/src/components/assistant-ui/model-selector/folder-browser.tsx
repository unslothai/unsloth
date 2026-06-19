// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  type BrowseFoldersResponse,
  browseFolders,
} from "@/features/chat/api/chat-api";
import { cn } from "@/lib/utils";
import { ArrowUp02Icon, Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface FolderBrowserProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called with the absolute path the user confirmed. */
  onSelect: (path: string) => void;
  /** Optional initial directory. Defaults to the user's home on the server. */
  initialPath?: string;
}

function splitBreadcrumb(path: string): { label: string; value: string }[] {
  if (!path) return [];
  // Distinguish path styles BEFORE normalizing separators. On POSIX
  // backslashes are valid filename characters, so we cannot blindly
  // rewrite ``\`` -> ``/`` -- doing so would mangle directory names
  // like ``my\backup`` into ``my/backup`` and produce breadcrumb
  // values that 404 on the server. Only Windows-style absolute paths
  // (drive letter, or UNC ``\\server\share``) get the conversion.
  const isWindowsDrive = /^[A-Za-z]:[\\/]/.test(path) || /^[A-Za-z]:$/.test(path);
  const isUnc = /^\\\\/.test(path);
  const isWindows = isWindowsDrive || isUnc;
  const normalized = isWindows ? path.replace(/\\/g, "/") : path;
  const segments = normalized.split("/");
  const parts: { label: string; value: string }[] = [];

  // POSIX absolute path: leading empty segment from split("/")
  if (segments[0] === "") {
    parts.push({ label: "/", value: "/" });
    let cur = "";
    for (const seg of segments.slice(1)) {
      if (!seg) continue;
      cur = `${cur}/${seg}`;
      parts.push({ label: seg, value: cur });
    }
    return parts;
  }

  // Windows-ish drive path (C:, D:): first segment is the drive. Use
  // ``C:/`` (drive-absolute) as the crumb value so clicking the drive
  // root navigates to the root of the drive rather than the
  // drive-relative current working directory on that drive (``C:``
  // alone resolves to ``CWD-on-C``, not ``C:\``).
  if (/^[A-Za-z]:$/.test(segments[0])) {
    const driveRoot = `${segments[0]}/`;
    let cur = driveRoot;
    parts.push({ label: segments[0], value: driveRoot });
    for (const seg of segments.slice(1)) {
      if (!seg) continue;
      cur = cur.endsWith("/") ? `${cur}${seg}` : `${cur}/${seg}`;
      parts.push({ label: seg, value: cur });
    }
    return parts;
  }

  // Fallback: relative / UNC-ish. Render as-is as a single crumb.
  return [{ label: path, value: path }];
}

export function FolderBrowser({
  open,
  onOpenChange,
  onSelect,
  initialPath,
}: FolderBrowserProps) {
  const [data, setData] = useState<BrowseFoldersResponse | null>(null);
  const [path, setPath] = useState<string | undefined>(initialPath);
  const [showHidden, setShowHidden] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const navigate = useCallback(
    (
      target: string | undefined,
      hidden: boolean,
      opts?: { fallbackOnError?: boolean },
    ) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setLoading(true);
      setError(null);
      // Forward the signal so cancelled navigation actually cancels the
      // backend enumeration instead of just discarding the response.
      browseFolders(target, hidden, ctrl.signal)
        .then((res) => {
          if (ctrl.signal.aborted) return;
          setData(res);
          setPath(res.current);
        })
        .catch((err) => {
          if (ctrl.signal.aborted) return;
          // Surface the error, but if the very first request (typically
          // a typo'd or denylisted ``initialPath``) fails AND the
          // browser is empty (no ``data`` to render against), fall
          // back to the user's HOME so the modal is navigable instead
          // of an irrecoverable dead end.
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          if (opts?.fallbackOnError && target !== undefined) {
            // Re-issue without a target -> backend defaults to HOME.
            // Don't recurse if HOME itself fails (paranoia: shouldn't
            // happen since the sandbox allowlist always includes HOME).
            queueMicrotask(() => navigate(undefined, hidden));
          }
        })
        .finally(() => {
          if (!ctrl.signal.aborted) setLoading(false);
        });
    },
    [],
  );

  // Fetch when the dialog opens.  Only re-run when the dialog transitions
  // closed -> open; subsequent navigation is driven by `navigate()` so we
  // don't want `path` in the dependency list here.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!open) return;
    // ``fallbackOnError``: if the user-supplied ``initialPath`` is bad
    // (typo, denylisted, deleted) we recover into HOME instead of
    // showing an empty modal with no breadcrumbs/entries.
    navigate(initialPath, showHidden, { fallbackOnError: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const handleConfirm = useCallback(() => {
    if (!path) return;
    onSelect(path);
    onOpenChange(false);
  }, [onSelect, onOpenChange, path]);

  const crumbs = useMemo(
    () => (data?.current ? splitBreadcrumb(data.current) : []),
    [data?.current],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-md p-0 gap-0"
        overlayClassName="bg-black/20 backdrop-blur-none"
        data-testid="folder-browser-dialog"
      >
        <DialogHeader className="px-4 pt-4 pb-2">
          <DialogTitle className="text-sm font-medium">
            Browse for folder
          </DialogTitle>
        </DialogHeader>

        {/* Breadcrumb */}
        <div className="flex flex-wrap items-center gap-0.5 border-t border-border/50 px-4 py-2 font-mono text-[11px] text-muted-foreground">
          {crumbs.length === 0 ? (
            <span className="text-muted-foreground/60">(loading…)</span>
          ) : (
            crumbs.map((c, i) => (
              <span key={c.value} className="flex items-center gap-0.5">
                <button
                  type="button"
                  className="rounded px-1 py-0.5 hover:bg-accent hover:text-foreground"
                  onClick={() => navigate(c.value, showHidden)}
                  disabled={loading}
                >
                  {c.label}
                </button>
                {i < crumbs.length - 1 && (
                  <span className="text-muted-foreground/40">/</span>
                )}
              </span>
            ))
          )}
        </div>

        {/* Suggestions (quick-pick chips) */}
        {data?.suggestions && data.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-1 border-t border-border/50 px-4 py-2">
            {data.suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => navigate(s, showHidden)}
                disabled={loading}
                className="rounded-full border border-border/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
                title={s}
              >
                {s.length > 36 ? `…${s.slice(-33)}` : s}
              </button>
            ))}
          </div>
        )}

        {/* Entry list */}
        <div className="max-h-64 min-h-24 overflow-y-auto border-t border-border/50">
          {error && (
            <div className="px-4 py-3 text-xs text-destructive">{error}</div>
          )}
          {!error && loading && (
            <div className="flex items-center gap-2 px-4 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Loading…</span>
            </div>
          )}
          {!error && !loading && data && (
            <>
              {/* Up row */}
              {data.parent !== null && (
                <button
                  type="button"
                  onClick={() => navigate(data.parent ?? undefined, showHidden)}
                  className="flex w-full items-center gap-2 px-4 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={ArrowUp02Icon}
                    className="size-3 shrink-0"
                  />
                  <span className="font-mono">..</span>
                </button>
              )}
              {data.entries.length === 0 && !(data.model_files_here && data.model_files_here > 0) && (
                <div className="px-4 py-3 text-xs text-muted-foreground/60">
                  (empty directory)
                </div>
              )}
              {data.model_files_here !== undefined && data.model_files_here > 0 && (
                <div className="border-t border-border/30 px-4 py-1.5 text-[10px] text-foreground/70">
                  {data.model_files_here} model file{data.model_files_here === 1 ? "" : "s"} in this folder. Click "Use this folder" to scan it.
                </div>
              )}
              {data.truncated === true && (
                <div className="border-t border-border/30 px-4 py-1.5 text-[10px] text-muted-foreground/70">
                  Showing first {data.entries.length} entries. Narrow the path
                  to see more.
                </div>
              )}
              {data.entries.map((e) => (
                <button
                  type="button"
                  key={e.name}
                  onClick={() => {
                    const sep = data.current.endsWith("/") ? "" : "/";
                    navigate(`${data.current}${sep}${e.name}`, showHidden);
                  }}
                  className={cn(
                    "flex w-full items-center gap-2 px-4 py-1.5 text-left text-xs transition-colors hover:bg-accent hover:text-foreground",
                    e.hidden && "text-muted-foreground/60",
                  )}
                >
                  <HugeiconsIcon
                    icon={Folder02Icon}
                    className={cn(
                      "size-3 shrink-0",
                      e.has_models
                        ? "text-foreground"
                        : "text-muted-foreground/50",
                    )}
                  />
                  <span className="truncate font-mono">{e.name}</span>
                  {e.has_models && (
                    <span className="ml-auto shrink-0 rounded-full border border-border/50 px-1.5 py-0 text-[9px] uppercase tracking-wider text-muted-foreground">
                      models
                    </span>
                  )}
                </button>
              ))}
            </>
          )}
        </div>

        {/* Footer */}
        <DialogFooter className="flex items-center justify-between gap-2 border-t border-border/50 px-4 py-2">
          <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-muted-foreground">
            <input
              type="checkbox"
              checked={showHidden}
              onChange={(e) => {
                const next = e.target.checked;
                setShowHidden(next);
                navigate(path, next);
              }}
              className="size-3"
            />
            Show hidden
          </label>
          <div className="flex gap-2">
            <DialogClose asChild={true}>
              <button
                type="button"
                className="h-7 rounded border border-border/50 px-2.5 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                Cancel
              </button>
            </DialogClose>
            <button
              type="button"
              onClick={handleConfirm}
              disabled={!path || loading || !!error}
              className="h-7 rounded bg-foreground px-2.5 text-[11px] font-medium text-background transition-colors hover:bg-foreground/90 disabled:opacity-40"
            >
              Use this folder
            </button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
