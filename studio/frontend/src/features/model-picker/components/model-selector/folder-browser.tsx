// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  type BrowseFoldersResponse,
  browseFolders,
  createBrowseFolder,
} from "@/features/chat";
import { useT } from "@/i18n";
import { ChevronUpStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Folder02Icon, FolderAddIcon } from "@/lib/folder-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  filterEntries,
  isPathLike,
  joinChild,
  resolveTypedPath,
  splitBreadcrumb,
  splitTypedPath,
} from "./folder-browser-path";

export interface FolderBrowserProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (path: string) => void;
  /** Optional initial directory. Defaults to the user's home on the server. */
  initialPath?: string;
  title?: string;
  description?: string;
  confirmLabel?: string;
  showModelHints?: boolean;
}

function errorStatus(error: unknown): number | undefined {
  return typeof error === "object" && error !== null && "status" in error
    ? Number((error as { status: unknown }).status)
    : undefined;
}

export function FolderBrowser({
  open,
  onOpenChange,
  onSelect,
  initialPath,
  title = "Select folder to detect models",
  description,
  confirmLabel = "Use this folder",
  showModelHints = true,
}: FolderBrowserProps) {
  const t = useT();
  const listId = useId();
  const [data, setData] = useState<BrowseFoldersResponse | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [newName, setNewName] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const nameRef = useRef<HTMLInputElement | null>(null);
  const naming = newName !== null;
  // Focused here, not with autoFocus: the dialog's focus trap takes focus back from a mounting input.
  useEffect(() => {
    if (naming) nameRef.current?.focus();
  }, [naming]);

  function navigate(
    target: string | undefined,
    hidden: boolean,
    opts?: { fallbackOnError?: boolean; typed?: boolean; filter?: string },
  ) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    browseFolders(target, hidden, ctrl.signal, showModelHints)
      .then((res) => {
        if (ctrl.signal.aborted) return;
        setData(res);
        setQuery(opts?.filter ?? "");
        setActive(0);
        setNewName(null);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        // A half-typed last name is not a folder yet: open its parent and narrow to it.
        const split =
          opts?.typed && target && errorStatus(err) === 404
            ? splitTypedPath(target)
            : null;
        if (split) {
          queueMicrotask(() =>
            navigate(split.parent, hidden, { filter: split.leaf }),
          );
          return;
        }
        // If the first request fails, fall back to HOME so the modal stays navigable.
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        if (opts?.fallbackOnError && target !== undefined) {
          // No target means HOME; do not recurse if HOME itself fails.
          queueMicrotask(() => navigate(undefined, hidden));
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
  }

  // Fetch only on closed -> open; the open folder is deliberately not a dependency.
  // biome-ignore lint/correctness/useExhaustiveDependencies: only reopening resets navigation to the initial path
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setNewName(null);
    navigate(initialPath, showHidden, { fallbackOnError: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const current = data?.current;
  const handleConfirm = useCallback(() => {
    if (!current) return;
    onSelect(current);
    onOpenChange(false);
  }, [onSelect, onOpenChange, current]);

  const crumbs = useMemo(
    () => (current ? splitBreadcrumb(current) : []),
    [current],
  );
  const typedPath = isPathLike(query);
  const visible = useMemo(
    () =>
      data
        ? typedPath
          ? data.entries
          : filterEntries(data.entries, query)
        : [],
    [data, query, typedPath],
  );
  const activeIndex = Math.min(active, Math.max(visible.length - 1, 0));
  const optionId = (index: number) => `${listId}-option-${index}`;

  // biome-ignore lint/correctness/useExhaustiveDependencies: optionId only depends on listId
  useEffect(() => {
    document
      .getElementById(optionId(activeIndex))
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, listId, visible]);

  function openEntry(name: string) {
    if (current) navigate(joinChild(current, name), showHidden);
  }

  function onInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive(
        Math.min(
          Math.max(activeIndex + step, 0),
          Math.max(visible.length - 1, 0),
        ),
      );
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (loading || !current) return;
      if (event.metaKey || event.ctrlKey) {
        handleConfirm();
      } else if (typedPath) {
        navigate(resolveTypedPath(current, query), showHidden, { typed: true });
      } else if (visible[activeIndex]) {
        openEntry(visible[activeIndex].name);
      }
      return;
    }
    if (event.key === "Backspace" && query === "" && data?.parent && !loading) {
      event.preventDefault();
      navigate(data.parent, showHidden);
    }
  }

  async function createFolder() {
    const name = (newName ?? "").trim();
    if (!name || !current || creating) return;
    setCreating(true);
    setError(null);
    try {
      const created = await createBrowseFolder(current, name);
      setData(created);
      setQuery("");
      setActive(0);
      setNewName(null);
      inputRef.current?.focus();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="corner-squircle dialog-soft-surface sm:max-w-md p-0 gap-0 [&_[data-slot=dialog-close]]:top-4"
        overlayClassName="bg-black/20 backdrop-blur-none"
        data-testid="folder-browser-dialog"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          inputRef.current?.focus();
        }}
        // Escape leaves the name row first; the dialog's own listener sits above any key handler in here.
        onEscapeKeyDown={(event) => {
          if (!naming) return;
          event.preventDefault();
          setNewName(null);
          inputRef.current?.focus();
        }}
      >
        <DialogHeader className="px-6 pt-6 pb-2">
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription className="sr-only">
            {description ?? title}. Type to filter the folders or enter a path,
            use the arrow keys to move, Enter to open a folder and Backspace to
            go up.
          </DialogDescription>
        </DialogHeader>

        <div
          className="flex min-h-9 flex-wrap items-center gap-0.5 border-t border-border px-6 py-2 font-mono text-ui-11 text-muted-foreground"
          data-testid="folder-browser-crumbs"
        >
          {crumbs.length === 0 ? (
            loading ? (
              <Spinner className="size-3 text-muted-foreground" />
            ) : (
              <button
                type="button"
                className="rounded px-1 py-0.5 hover:bg-muted hover:text-foreground"
                onClick={() => navigate(undefined, showHidden)}
              >
                Go to home folder
              </button>
            )
          ) : (
            crumbs.map((c, i) => (
              <span key={c.value} className="flex items-center gap-0.5">
                <button
                  type="button"
                  className="rounded px-1 py-0.5 hover:bg-muted hover:text-foreground"
                  onClick={() => navigate(c.value, showHidden)}
                  disabled={loading}
                  aria-current={
                    i === crumbs.length - 1 ? "location" : undefined
                  }
                >
                  {c.label}
                </button>
                {i < crumbs.length - 1 && c.label !== "/" && (
                  <span className="text-muted-foreground/80">/</span>
                )}
              </span>
            ))
          )}
        </div>

        {data?.suggestions && data.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-1 border-t border-border px-6 py-2">
            {data.suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => navigate(s, showHidden)}
                disabled={loading}
                className="rounded-full border border-border px-2 py-0.5 font-mono text-ui-10 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40"
                title={s}
              >
                {s.length > 36 ? `…${s.slice(-33)}` : s}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 border-t border-border px-6 py-2">
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-label="Filter folders or type a path"
            aria-expanded={true}
            aria-controls={listId}
            aria-activedescendant={
              visible[activeIndex] ? optionId(activeIndex) : undefined
            }
            aria-autocomplete="list"
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
              setError(null);
            }}
            onKeyDown={onInputKeyDown}
            placeholder="Filter, or type a path like ~/models"
            className="h-7 min-w-0 flex-1 bg-transparent font-mono text-xs outline-none placeholder:font-sans placeholder:text-muted-foreground"
            data-testid="folder-browser-input"
          />
          {typedPath && (
            <span className="shrink-0 text-ui-10 text-muted-foreground">
              Enter to open
            </span>
          )}
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="shrink-0 rounded-full"
            disabled={!current || loading}
            aria-label="New folder"
            title="New folder"
            onClick={() => {
              if (naming) nameRef.current?.focus();
              else setNewName("");
            }}
          >
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-4"
            />
          </Button>
        </div>

        <div className="h-64 overflow-y-auto border-t border-border">
          {error && (
            <div
              role="alert"
              className="px-6 py-2 text-xs text-destructive"
              data-testid="folder-browser-error"
            >
              {error}
            </div>
          )}
          {newName !== null && (
            <form
              className="flex items-center gap-2 border-b border-border px-6 py-1.5"
              onSubmit={(event) => {
                event.preventDefault();
                void createFolder();
              }}
            >
              <HugeiconsIcon
                icon={FolderAddIcon}
                className="size-3 shrink-0 text-muted-foreground"
              />
              <input
                ref={nameRef}
                type="text"
                aria-label="New folder name"
                autoComplete="off"
                spellCheck={false}
                maxLength={255}
                value={newName}
                disabled={creating}
                onChange={(event) => setNewName(event.target.value)}
                placeholder="Folder name"
                className="h-6 min-w-0 flex-1 bg-transparent font-mono text-xs outline-none placeholder:font-sans placeholder:text-muted-foreground"
                data-testid="folder-browser-new-name"
              />
              <Button
                type="submit"
                variant="ghost"
                size="sm"
                className="h-6 shrink-0 px-2 text-xs"
                disabled={creating || !newName.trim()}
              >
                {creating ? "Creating…" : "Create"}
              </Button>
            </form>
          )}
          {!data && loading && (
            <div className="flex items-center gap-2 px-6 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                {t("common.loading")}
              </span>
            </div>
          )}
          {data && (
            <div
              className={cn(
                "transition-opacity duration-150",
                loading && "pointer-events-none opacity-50",
              )}
            >
              {data.parent !== null && query === "" && (
                <button
                  type="button"
                  tabIndex={-1}
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
              {visible.length === 0 &&
                (!showModelHints ||
                  !(data.model_files_here && data.model_files_here > 0)) && (
                  <div className="px-6 py-3 text-xs text-muted-foreground/80">
                    {data.entries.length === 0
                      ? "No folders here."
                      : `No folder matches "${query.trim()}".`}
                  </div>
                )}
              {showModelHints &&
                data.model_files_here !== undefined &&
                data.model_files_here > 0 && (
                  <div className="border-t border-border px-6 py-1.5 text-ui-10 text-muted-foreground">
                    {data.model_files_here} model file
                    {data.model_files_here === 1 ? "" : "s"} in this folder.
                    Click "{confirmLabel}" to scan it.
                  </div>
                )}
              {data.truncated === true && (
                <div className="border-t border-border px-6 py-1.5 text-ui-10 text-muted-foreground/80">
                  Showing the first {data.entries.length} folders. Type a path
                  to go straight to one.
                </div>
              )}
              <div
                id={listId}
                role="listbox"
                aria-label="Folders"
                tabIndex={-1}
              >
                {visible.map((e, index) => (
                  <button
                    type="button"
                    role="option"
                    id={optionId(index)}
                    aria-selected={index === activeIndex}
                    tabIndex={-1}
                    key={e.name}
                    onClick={() => openEntry(e.name)}
                    onMouseMove={() => {
                      if (index !== activeIndex) setActive(index);
                    }}
                    className={cn(
                      "flex w-full items-center gap-2 px-6 py-1.5 text-left text-xs transition-colors",
                      index === activeIndex && "bg-muted text-foreground",
                      e.hidden && "text-muted-foreground/80",
                    )}
                  >
                    <HugeiconsIcon
                      icon={Folder02Icon}
                      className={cn(
                        "size-3 shrink-0",
                        e.has_models
                          ? "text-foreground"
                          : "text-muted-foreground/80",
                      )}
                    />
                    <span className="truncate font-mono">{e.name}</span>
                    {showModelHints && e.has_models && (
                      <span className="ml-auto shrink-0 rounded-full border border-border px-1.5 py-0 text-ui-9 uppercase tracking-wider text-muted-foreground">
                        models
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="flex items-center justify-between gap-2 border-t border-border px-6 py-3">
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
                navigate(current, next, { filter: typedPath ? "" : query });
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
              disabled={!current || loading}
              title={current}
            >
              {confirmLabel}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
