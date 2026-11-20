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
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Folder02Icon, FolderAddIcon } from "@/lib/folder-icons";
import {
  Clock01Icon,
  HardDriveIcon,
  Home09Icon,
  PackageIcon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement, HugeiconsIcon } from "@hugeicons/react";
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
  collapseCrumbs,
  filterEntries,
  folderName,
  isPathLike,
  isWindowsPath,
  joinChild,
  readRecentFolders,
  rememberRecentFolder,
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
  acceptUnbrowsable?: boolean;
}

const PLACE_ICONS: Record<string, IconSvgElement> = {
  home: Home09Icon,
  user: Folder02Icon,
  drive: HardDriveIcon,
  models: PackageIcon,
};

const SIDE_ROW_CLASS =
  "flex w-full min-w-0 items-center gap-2 rounded-lg px-2 py-1.5 text-left text-ui-13 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50 data-[active=true]:bg-muted data-[active=true]:text-foreground";

function recentStore(): Storage | undefined {
  try {
    return typeof window === "undefined" ? undefined : window.localStorage;
  } catch {
    return undefined;
  }
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
  acceptUnbrowsable = false,
}: FolderBrowserProps) {
  const t = useT();
  const listId = useId();
  const [data, setData] = useState<BrowseFoldersResponse | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unbrowsable, setUnbrowsable] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [newName, setNewName] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [recents, setRecents] = useState<string[]>(() =>
    readRecentFolders(recentStore()),
  );
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
    setUnbrowsable(null);
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
        if (
          acceptUnbrowsable &&
          opts?.typed &&
          target &&
          errorStatus(err) === 403
        ) {
          setUnbrowsable(target);
        }
        if (opts?.fallbackOnError && target !== undefined) {
          // No target means HOME; do not recurse if HOME itself fails.
          queueMicrotask(() => navigate(undefined, hidden));
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: only reopening resets navigation to the initial path
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setNewName(null);
    setRecents(readRecentFolders(recentStore()));
    navigate(initialPath, showHidden, { fallbackOnError: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const current = data?.current;
  const handleConfirm = useCallback(() => {
    if (!current) return;
    setRecents(rememberRecentFolder(recentStore(), current));
    onSelect(current);
    onOpenChange(false);
  }, [onSelect, onOpenChange, current]);

  const places = useMemo(
    () =>
      data?.places ??
      (data?.suggestions ?? []).map((path) => ({
        path,
        label: folderName(path),
        kind: "models" as const,
      })),
    [data],
  );

  const crumbs = useMemo(
    () => collapseCrumbs(current ? splitBreadcrumb(current) : []),
    [current],
  );
  const typedPath = isPathLike(query, isWindowsPath(current));
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
  const {
    attach: attachList,
    onScroll: onListScroll,
    className: listFadeClass,
  } = useScrollFades();
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

  // Handled keys stop here: React bubbles portal events, and a parent dialog saves on Cmd/Ctrl+Enter.
  function onInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
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
      event.stopPropagation();
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
      event.stopPropagation();
      navigate(data.parent, showHidden);
    }
  }

  async function createFolder() {
    const name = (newName ?? "").trim();
    if (!name || !current || creating) return;
    setCreating(true);
    setError(null);
    // Shares navigation's controller: a later navigation wins over this response.
    abortRef.current?.abort();
    setLoading(false);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const created = await createBrowseFolder(current, name);
      if (ctrl.signal.aborted) return;
      setData(created);
      setQuery("");
      setActive(0);
      setNewName(null);
      inputRef.current?.focus();
    } catch (err) {
      if (!ctrl.signal.aborted) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setCreating(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="corner-squircle dialog-soft-surface gap-0 p-0 sm:max-w-2xl **:data-[slot=dialog-close]:top-4"
        overlayClassName="bg-black/20 backdrop-blur-none"
        data-testid="folder-browser-dialog"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          inputRef.current?.focus();
        }}
        onEscapeKeyDown={(event) => {
          if (!naming) return;
          event.preventDefault();
          setNewName(null);
          inputRef.current?.focus();
        }}
      >
        <DialogHeader className="px-6 pt-6 pb-4">
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription className="sr-only">
            {description ?? title}. Type to filter the folders or enter a path,
            use the arrow keys to move, Enter to open a folder and Backspace to
            go up.
          </DialogDescription>
        </DialogHeader>

        <div className="flex h-[calc(27rem*var(--ui-space-scale,1))] min-h-0 border-t border-border">
          <nav
            aria-label="Places"
            className="hidden w-44 shrink-0 flex-col gap-3 overflow-y-auto border-r border-border p-2 sm:flex"
            data-testid="folder-browser-places"
          >
            <div className="flex flex-col gap-0.5">
              {places.map((place) => (
                <button
                  key={place.path}
                  type="button"
                  title={place.path}
                  disabled={loading}
                  data-active={place.path === current}
                  onClick={() => navigate(place.path, showHidden)}
                  className={SIDE_ROW_CLASS}
                >
                  <HugeiconsIcon
                    icon={PLACE_ICONS[place.kind] ?? Folder02Icon}
                    strokeWidth={1.75}
                    className="size-4 shrink-0"
                  />
                  <span className="truncate">{place.label}</span>
                </button>
              ))}
            </div>
            {recents.length > 0 && (
              <div className="flex flex-col gap-0.5">
                <p className="px-2 pb-0.5 text-ui-11 font-medium text-muted-foreground/80">
                  Recent
                </p>
                {recents.map((path) => (
                  <button
                    key={path}
                    type="button"
                    title={path}
                    disabled={loading}
                    data-active={path === current}
                    onClick={() => navigate(path, showHidden)}
                    className={SIDE_ROW_CLASS}
                  >
                    <HugeiconsIcon
                      icon={Clock01Icon}
                      strokeWidth={1.75}
                      className="size-4 shrink-0"
                    />
                    <span className="truncate">{folderName(path)}</span>
                  </button>
                ))}
              </div>
            )}
          </nav>

          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div className="flex h-11 items-center gap-1 px-3">
              <div
                className="flex min-w-0 flex-1 items-center gap-0.5 overflow-hidden text-ui-13 text-muted-foreground"
                data-testid="folder-browser-crumbs"
                title={current}
              >
                {!current ? (
                  loading ? (
                    <Spinner className="size-3 text-muted-foreground" />
                  ) : (
                    <button
                      type="button"
                      className="rounded-md px-1.5 py-0.5 hover:bg-muted hover:text-foreground"
                      onClick={() => navigate(undefined, showHidden)}
                    >
                      Go to home folder
                    </button>
                  )
                ) : (
                  <>
                    {crumbs.hidden.length > 0 && (
                      <>
                        <button
                          type="button"
                          className="shrink-0 rounded-md px-1.5 py-0.5 hover:bg-muted hover:text-foreground"
                          disabled={loading}
                          aria-label="Earlier folders"
                          title={crumbs.hidden[crumbs.hidden.length - 1].value}
                          onClick={() =>
                            navigate(
                              crumbs.hidden[crumbs.hidden.length - 1].value,
                              showHidden,
                            )
                          }
                        >
                          …
                        </button>
                        <HugeiconsIcon
                          icon={ChevronRightStandardIcon}
                          className="size-3 shrink-0 text-muted-foreground/60"
                        />
                      </>
                    )}
                    {crumbs.shown.map((c, i) => {
                      const last = i === crumbs.shown.length - 1;
                      return (
                        <span
                          key={c.value}
                          className={cn(
                            "flex items-center gap-0.5",
                            last ? "min-w-0" : "shrink-0",
                          )}
                        >
                          <button
                            type="button"
                            className={cn(
                              "max-w-40 truncate rounded-md px-1.5 py-0.5 hover:bg-muted hover:text-foreground",
                              last && "font-medium text-foreground",
                            )}
                            onClick={() => navigate(c.value, showHidden)}
                            disabled={loading}
                            aria-current={last ? "location" : undefined}
                          >
                            {c.label}
                          </button>
                          {!last && (
                            <HugeiconsIcon
                              icon={ChevronRightStandardIcon}
                              className="size-3 shrink-0 text-muted-foreground/60"
                            />
                          )}
                        </span>
                      );
                    })}
                  </>
                )}
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="shrink-0 gap-1.5 rounded-full px-2.5 text-ui-13 text-muted-foreground"
                disabled={!current || loading}
                aria-label="New folder"
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
                <span className="hidden sm:inline">New folder</span>
              </Button>
            </div>

            {places.length > 0 && (
              <div className="flex gap-1 overflow-x-auto px-4 pb-2 sm:hidden">
                {places.map((place) => (
                  <button
                    key={place.path}
                    type="button"
                    onClick={() => navigate(place.path, showHidden)}
                    disabled={loading}
                    className="shrink-0 rounded-full border border-border px-2.5 py-0.5 text-ui-12 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40"
                    title={place.path}
                  >
                    {place.label}
                  </button>
                ))}
              </div>
            )}

            <div className="px-4 pb-2">
              <div className="flex h-9 items-center gap-2 rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255/calc(0.06*var(--contrast-wash-gain,1)))]">
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
                  className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                  data-testid="folder-browser-input"
                />
                {typedPath && (
                  <span className="shrink-0 text-ui-11 text-muted-foreground">
                    Enter to open
                  </span>
                )}
              </div>
            </div>

            <div
              ref={attachList}
              onScroll={onListScroll}
              className={cn(
                "model-list-scroll min-h-0 flex-1 overflow-y-auto px-2 py-1.5",
                listFadeClass,
              )}
            >
              <div>
                {error && (
                  <div
                    role="alert"
                    className="px-2.5 py-2 text-ui-13 text-destructive"
                    data-testid="folder-browser-error"
                  >
                    {error}
                    {unbrowsable && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="mt-1.5 flex h-7 rounded-full px-2.5 text-ui-13 text-foreground"
                        onClick={() => {
                          onSelect(unbrowsable);
                          onOpenChange(false);
                        }}
                        data-testid="folder-browser-add-unbrowsable"
                      >
                        Add {unbrowsable} anyway
                      </Button>
                    )}
                  </div>
                )}
                {newName !== null && (
                  <form
                    className="mb-1 flex items-center gap-2 rounded-lg bg-muted/60 px-2.5 py-1"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void createFolder();
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") event.stopPropagation();
                    }}
                  >
                    <HugeiconsIcon
                      icon={FolderAddIcon}
                      strokeWidth={1.75}
                      className="size-4 shrink-0 text-muted-foreground"
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
                      className="h-7 min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                      data-testid="folder-browser-new-name"
                    />
                    <Button
                      type="submit"
                      variant="ghost"
                      size="sm"
                      className="h-7 shrink-0 rounded-full px-2.5 text-ui-13"
                      disabled={creating || !newName.trim()}
                    >
                      {creating ? "Creating…" : "Create"}
                    </Button>
                  </form>
                )}
                {!data && loading && (
                  <div className="flex items-center gap-2 px-2.5 py-2">
                    <Spinner className="size-3 text-muted-foreground" />
                    <span className="text-ui-13 text-muted-foreground">
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
                    {visible.length === 0 &&
                      (!showModelHints ||
                        !(
                          data.model_files_here && data.model_files_here > 0
                        )) && (
                        <div className="px-2.5 py-6 text-center text-ui-13 text-muted-foreground">
                          {data.entries.length === 0
                            ? "No folders in here."
                            : `No folder matches "${query.trim()}".`}
                        </div>
                      )}
                    {showModelHints &&
                      data.model_files_here !== undefined &&
                      data.model_files_here > 0 && (
                        <div className="px-2.5 py-1.5 text-ui-12 text-muted-foreground">
                          {data.model_files_here} model file
                          {data.model_files_here === 1 ? "" : "s"} in this
                          folder. Click "{confirmLabel}" to scan it.
                        </div>
                      )}
                    {data.truncated === true && (
                      <div className="px-2.5 py-1.5 text-ui-12 text-muted-foreground/80">
                        Showing the first {data.entries.length} folders. Type a
                        path to go straight to one.
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
                            "group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-sm transition-colors",
                            index === activeIndex && "bg-muted text-foreground",
                            e.hidden && "text-muted-foreground",
                          )}
                        >
                          <HugeiconsIcon
                            icon={Folder02Icon}
                            strokeWidth={1.75}
                            className={cn(
                              "size-4 shrink-0",
                              e.has_models
                                ? "text-foreground"
                                : "text-muted-foreground",
                            )}
                          />
                          <span className="min-w-0 flex-1 truncate">
                            {e.name}
                          </span>
                          {showModelHints && e.has_models && (
                            <span className="shrink-0 rounded-full border border-border px-1.5 py-0 text-ui-10 text-muted-foreground">
                              Models
                            </span>
                          )}
                          <HugeiconsIcon
                            icon={ChevronRightStandardIcon}
                            className={cn(
                              "size-3.5 shrink-0 text-muted-foreground/70",
                              index !== activeIndex && "invisible",
                            )}
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border py-3 pr-4 pl-5">
              <label
                htmlFor="folder-browser-show-hidden"
                className="flex shrink-0 cursor-pointer items-center gap-2 text-ui-13 text-muted-foreground"
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
              <div className="flex min-w-0 items-center gap-2">
                <DialogClose asChild={true}>
                  <Button type="button" variant="ghost">
                    Cancel
                  </Button>
                </DialogClose>
                <Button
                  type="button"
                  variant="dark"
                  onClick={handleConfirm}
                  disabled={!current || loading}
                  title={current}
                >
                  {confirmLabel}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
