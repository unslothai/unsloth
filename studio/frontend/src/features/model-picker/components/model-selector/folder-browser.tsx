// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { type BrowseFoldersResponse, browseFolders } from "@/features/chat";
import { ChevronUpStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Folder02Icon } from "@hugeicons/core-free-icons";
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
  // Detect path style BEFORE normalizing: on POSIX, `\` is a valid filename
  // char, so blindly rewriting `\` -> `/` mangles names like `my\backup` into
  // 404ing breadcrumbs. Only Windows-style paths (drive letter, or UNC) convert.
  const isWindowsDrive =
    /^[A-Za-z]:[\\/]/.test(path) || /^[A-Za-z]:$/.test(path);
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

  // Windows drive path (C:, D:): first segment is the drive. Use `C:/` as the
  // crumb value so clicking the drive root navigates to the drive root, not the
  // drive-relative CWD (`C:` alone resolves to CWD-on-C, not `C:\`).
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

  function navigate(
    target: string | undefined,
    hidden: boolean,
    opts?: { fallbackOnError?: boolean },
  ) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    // Forward the signal so cancelled navigation aborts the backend
    // enumeration, not just the response.
    browseFolders(target, hidden, ctrl.signal)
      .then((res) => {
        if (ctrl.signal.aborted) return;
        setData(res);
        setPath(res.current);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        // Surface the error; if the first request (e.g. a bad initialPath)
        // fails, fall back to HOME so the modal stays navigable.
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        if (opts?.fallbackOnError && target !== undefined) {
          // Re-issue without a target -> backend defaults to HOME.
          // Don't recurse if HOME itself fails (allowlist always has HOME).
          queueMicrotask(() => navigate(undefined, hidden));
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
  }

  // Fetch only on closed -> open; later navigation is driven by `navigate()`,
  // so `path` is deliberately kept out of the dependency list.
  useEffect(() => {
    if (!open) return;
    // fallbackOnError: recover into HOME if initialPath is bad, rather than
    // showing an empty modal.
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
    [data],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="corner-squircle dialog-soft-surface sm:max-w-md p-0 gap-0 [&_[data-slot=dialog-close]]:top-4"
        overlayClassName="bg-black/20 backdrop-blur-none"
        data-testid="folder-browser-dialog"
      >
        <DialogHeader className="px-6 pt-6 pb-2">
          <DialogTitle>Select folder to detect models</DialogTitle>
        </DialogHeader>

        {/* Breadcrumb */}
        <div className="flex flex-wrap items-center gap-0.5 border-t border-border/50 px-6 py-2 font-mono text-[11px] text-muted-foreground">
          {crumbs.length === 0 ? (
            <span className="text-muted-foreground/60">(loading…)</span>
          ) : (
            crumbs.map((c, i) => (
              <span key={c.value} className="flex items-center gap-0.5">
                <button
                  type="button"
                  className="rounded px-1 py-0.5 hover:bg-muted hover:text-foreground"
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
          <div className="flex flex-wrap gap-1 border-t border-border/50 px-6 py-2">
            {data.suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => navigate(s, showHidden)}
                disabled={loading}
                className="rounded-full border border-border/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40"
                title={s}
              >
                {s.length > 36 ? `…${s.slice(-33)}` : s}
              </button>
            ))}
          </div>
        )}

        {/* Entry list. Keep the list mounted while a refetch is in flight (e.g.
        toggling Show hidden) and just dim it, so the dialog doesn't collapse and
        flash. The full-height spinner only shows on the first load, when there
        is no data yet. */}
        <div className="max-h-64 min-h-24 overflow-y-auto border-t border-border/50">
          {error && (
            <div className="px-6 py-3 text-xs text-destructive">{error}</div>
          )}
          {!error && !data && loading && (
            <div className="flex items-center gap-2 px-6 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Loading…</span>
            </div>
          )}
          {!error && data && (
            <div
              className={cn(
                "transition-opacity duration-150",
                loading && "pointer-events-none opacity-50",
              )}
            >
              {/* Up row */}
              {data.parent !== null && (
                <button
                  type="button"
                  onClick={() => navigate(data.parent ?? undefined, showHidden)}
                  className="flex w-full items-center gap-2 px-6 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={ChevronUpStandardIcon}
                    className="size-3 shrink-0"
                  />
                  <span className="font-mono">..</span>
                </button>
              )}
              {data.entries.length === 0 &&
                !(data.model_files_here && data.model_files_here > 0) && (
                  <div className="px-6 py-3 text-xs text-muted-foreground/60">
                    (empty directory)
                  </div>
                )}
              {data.model_files_here !== undefined &&
                data.model_files_here > 0 && (
                  <div className="border-t border-border/30 px-6 py-1.5 text-[10px] text-foreground/70">
                    {data.model_files_here} model file
                    {data.model_files_here === 1 ? "" : "s"} in this folder.
                    Click "Use this folder" to scan it.
                  </div>
                )}
              {data.truncated === true && (
                <div className="border-t border-border/30 px-6 py-1.5 text-[10px] text-muted-foreground/70">
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
                    "flex w-full items-center gap-2 px-6 py-1.5 text-left text-xs transition-colors hover:bg-muted hover:text-foreground",
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
            </div>
          )}
        </div>

        {/* Footer */}
        <DialogFooter className="flex items-center justify-between gap-2 border-t border-border/50 px-6 py-3">
          <label
            htmlFor="folder-browser-show-hidden"
            className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground"
          >
            <Checkbox
              id="folder-browser-show-hidden"
              className="rounded-full"
              checked={showHidden}
              onCheckedChange={(checked) => {
                const next = checked === true;
                setShowHidden(next);
                navigate(path, next);
              }}
            />
            Show hidden
          </label>
          <div className="flex gap-2">
            <DialogClose asChild={true}>
              <Button type="button" variant="ghost">
                Cancel
              </Button>
            </DialogClose>
            <Button
              type="button"
              onClick={handleConfirm}
              disabled={!path || loading || !!error}
            >
              Use this folder
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
