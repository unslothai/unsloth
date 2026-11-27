// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Folder02Icon, FolderOpenIcon } from "@/lib/folder-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  File01Icon,
  PlusSignIcon,
  Refresh01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ProjectChangeStatus,
  type ProjectChanges,
  type ProjectFileSearch,
  getProjectChanges,
  listProjectFiles,
  searchProjectFiles,
} from "../api/chat-api";
import { useChatArtifactsStore } from "../artifacts/store";
import {
  CHANGE_STATUS_LABELS,
  type ChangeTone,
  type DirState,
  addToChatText,
  changeStatusTone,
  closeSubtree,
  dirStateAt,
  flattenFileTree,
  formatFileSize,
  pathsToLoad,
  splitProjectPath,
  toggleOpenPath,
} from "../utils/project-files";

type Loadable<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; message: string };

const SEARCH_DEBOUNCE_MS = 200;

const TONE_CLASS: Record<ChangeTone, string> = {
  amber: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  green: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  red: "bg-red-500/10 text-red-700 dark:text-red-400",
  neutral: "bg-muted text-muted-foreground",
};

const ROW_CLASS =
  "group flex min-w-0 items-center gap-1 rounded-lg pr-1 transition-colors hover:bg-muted focus-within:bg-muted";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function IconButton({
  label,
  icon,
  onClick,
  className,
}: {
  label: string;
  icon: IconSvgElement;
  onClick: () => void;
  className?: string;
}): ReactElement {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      className={cn(
        "flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
    </button>
  );
}

function RowActions({
  path,
  file,
  onCopy,
  onAdd,
}: {
  path: string;
  file: boolean;
  onCopy: (path: string) => void;
  onAdd: (path: string) => void;
}): ReactElement {
  return (
    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
      {file && (
        <IconButton
          label="Add to chat"
          icon={PlusSignIcon}
          onClick={() => onAdd(path)}
        />
      )}
      <IconButton
        label="Copy path"
        icon={Copy01Icon}
        onClick={() => onCopy(path)}
      />
    </div>
  );
}

function StatusRow({
  depth = 0,
  children,
  tone = "muted",
}: {
  depth?: number;
  children: ReactNode;
  tone?: "muted" | "error";
}): ReactElement {
  return (
    <div
      className={cn(
        "flex items-center gap-2 py-1.5 pr-2.5 text-ui-12",
        tone === "error" ? "text-destructive" : "text-muted-foreground/80",
      )}
      style={{
        paddingLeft: `calc((${depth} * 1rem + 2.25rem) * var(--ui-space-scale, 1))`,
      }}
    >
      {children}
    </div>
  );
}

function RetryButton({ onClick }: { onClick: () => void }): ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-full px-2 py-0.5 text-ui-12 font-medium text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      Retry
    </button>
  );
}

function PathLabel({ path }: { path: string }): ReactElement {
  const { dir, name } = splitProjectPath(path);
  return (
    <span className="min-w-0 flex-1 truncate" title={path}>
      <span className="text-foreground">{name}</span>
      {dir && <span className="ml-2 text-muted-foreground">{dir}</span>}
    </span>
  );
}

function FlatFileRow({
  path,
  leading,
  onCopy,
  onAdd,
}: {
  path: string;
  leading?: ReactNode;
  onCopy: (path: string) => void;
  onAdd: (path: string) => void;
}): ReactElement {
  return (
    <div className={ROW_CLASS}>
      <div className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-1.5 text-sm">
        {leading}
        <HugeiconsIcon
          icon={File01Icon}
          strokeWidth={1.75}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <PathLabel path={path} />
      </div>
      <RowActions path={path} file={true} onCopy={onCopy} onAdd={onAdd} />
    </div>
  );
}

function ChangeStatusLabel({
  status,
}: {
  status: ProjectChangeStatus;
}): ReactElement {
  return (
    <span
      className={cn(
        "w-[calc(5.5rem*var(--ui-space-scale,1))] shrink-0 rounded-full px-1.5 py-0.5 text-center text-ui-10 font-medium",
        TONE_CLASS[changeStatusTone(status)],
      )}
    >
      {CHANGE_STATUS_LABELS[status] ?? status}
    </span>
  );
}

export function ProjectFilesPanel({
  projectId,
}: {
  projectId: string;
}): ReactElement {
  const [dirs, setDirs] = useState<Record<string, DirState>>({});
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const [mode, setMode] = useState<"all" | "changes">("all");
  const [changes, setChanges] = useState<Loadable<ProjectChanges>>({
    status: "loading",
  });
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState<Loadable<ProjectFileSearch> | null>(
    null,
  );
  const [searchNonce, setSearchNonce] = useState(0);
  const dirControllers = useRef(new Map<string, AbortController>());
  const changesController = useRef<AbortController | null>(null);

  const loadDir = useCallback(
    (path: string) => {
      dirControllers.current.get(path)?.abort();
      const controller = new AbortController();
      dirControllers.current.set(path, controller);
      setDirs((prev) => ({ ...prev, [path]: { status: "loading" } }));
      listProjectFiles(projectId, path, false, controller.signal)
        .then(
          (listing) => {
            if (controller.signal.aborted) return;
            setDirs((prev) => ({
              ...prev,
              [path]: { status: "ready", listing },
            }));
          },
          (error) => {
            if (controller.signal.aborted) return;
            setDirs((prev) => ({
              ...prev,
              [path]: { status: "error", message: errorMessage(error) },
            }));
          },
        )
        .finally(() => {
          if (dirControllers.current.get(path) === controller) {
            dirControllers.current.delete(path);
          }
        });
    },
    [projectId],
  );

  const loadChanges = useCallback(() => {
    changesController.current?.abort();
    const controller = new AbortController();
    changesController.current = controller;
    setChanges((prev) =>
      prev.status === "ready" ? prev : { status: "loading" },
    );
    getProjectChanges(projectId, controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setChanges({ status: "ready", data });
      },
      (error) => {
        if (!controller.signal.aborted) {
          setChanges({ status: "error", message: errorMessage(error) });
        }
      },
    );
  }, [projectId]);

  useEffect(() => {
    for (const path of pathsToLoad(dirs, open)) loadDir(path);
  }, [dirs, open, loadDir]);

  useEffect(() => {
    loadChanges();
  }, [loadChanges]);

  useEffect(() => {
    const controllers = dirControllers.current;
    return () => {
      for (const controller of controllers.values()) controller.abort();
      controllers.clear();
      changesController.current?.abort();
    };
  }, []);

  const trimmed = query.trim();
  // biome-ignore lint/correctness/useExhaustiveDependencies: searchNonce is the refetch trigger for Refresh and Retry.
  useEffect(() => {
    if (!trimmed) {
      setSearch(null);
      return;
    }
    const controller = new AbortController();
    setSearch((prev) =>
      prev?.status === "ready" ? prev : { status: "loading" },
    );
    const timer = window.setTimeout(() => {
      searchProjectFiles(projectId, trimmed, controller.signal).then(
        (data) => {
          if (!controller.signal.aborted) setSearch({ status: "ready", data });
        },
        (error) => {
          if (!controller.signal.aborted) {
            setSearch({ status: "error", message: errorMessage(error) });
          }
        },
      );
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [projectId, trimmed, searchNonce]);

  const refresh = useCallback(() => {
    for (const controller of dirControllers.current.values()) {
      controller.abort();
    }
    dirControllers.current.clear();
    setDirs({});
    loadChanges();
    setSearchNonce((n) => n + 1);
  }, [loadChanges]);

  const toggleDir = useCallback((path: string) => {
    setOpen((prev) =>
      prev.has(path) ? closeSubtree(prev, path) : toggleOpenPath(prev, path),
    );
  }, []);

  const copyPath = useCallback((path: string) => {
    void copyToClipboard(path).then((copied) => {
      if (copied) toast.success("Path copied");
      else toast.error("Could not copy the path");
    });
  }, []);

  const addToChat = useCallback((path: string) => {
    useChatArtifactsStore.getState().stageFixPrompt(addToChatText(path));
    toast.success("Added to the message");
  }, []);

  const rows = useMemo(() => flattenFileTree(dirs, open), [dirs, open]);
  const rootState = dirs[""];
  const rootListing = rootState?.status === "ready" ? rootState.listing : null;
  const repo = changes.status === "ready" && changes.data.repo;
  const changeCount = repo ? changes.data.changes.length : 0;
  const view = repo && mode === "changes" ? "changes" : "all";

  let body: ReactNode;
  if (trimmed) {
    if (!search || search.status === "loading") {
      body = (
        <StatusRow>
          <Spinner className="size-3" />
          Searching…
        </StatusRow>
      );
    } else if (search.status === "error") {
      body = (
        <StatusRow tone="error">
          {search.message}
          <RetryButton onClick={() => setSearchNonce((n) => n + 1)} />
        </StatusRow>
      );
    } else {
      body = (
        <>
          {search.data.results.length === 0 ? (
            <StatusRow>No files match</StatusRow>
          ) : (
            search.data.results.map((result) => (
              <FlatFileRow
                key={result.path}
                path={result.path}
                onCopy={copyPath}
                onAdd={addToChat}
              />
            ))
          )}
          {search.data.truncated && (
            <StatusRow>
              The folder is large, so only part of it was searched.
            </StatusRow>
          )}
        </>
      );
    }
  } else if (view === "changes" && changes.status === "ready") {
    const list = changes.data.changes;
    body = (
      <>
        {list.length === 0 ? (
          <StatusRow>No changes</StatusRow>
        ) : (
          list.map((change) => (
            <FlatFileRow
              key={`${change.status}:${change.path}`}
              path={change.path}
              leading={<ChangeStatusLabel status={change.status} />}
              onCopy={copyPath}
              onAdd={addToChat}
            />
          ))
        )}
        {changes.data.truncated && (
          <StatusRow>Showing the first {list.length} changes</StatusRow>
        )}
      </>
    );
  } else {
    body = rows.map((row) => {
      const key = `${row.type}:${row.path}`;
      if (row.type === "loading") {
        return row.depth === 0 ? (
          <StatusRow key={key}>
            <Spinner className="size-3" />
            Loading…
          </StatusRow>
        ) : null;
      }
      if (row.type === "empty") {
        return (
          <StatusRow key={key} depth={row.depth}>
            Empty
          </StatusRow>
        );
      }
      if (row.type === "truncated") {
        return (
          <StatusRow key={key} depth={row.depth}>
            Showing the first {row.count} items
          </StatusRow>
        );
      }
      if (row.type === "error") {
        return (
          <StatusRow key={key} depth={row.depth} tone="error">
            {row.message}
            <RetryButton onClick={() => loadDir(row.path)} />
          </StatusRow>
        );
      }
      const { entry, path, depth } = row;
      const indent = {
        paddingLeft: `calc((${depth} * 1rem + 0.625rem) * var(--ui-space-scale, 1))`,
      };
      if (entry.kind === "dir") {
        const loading = row.open && dirStateAt(dirs, path)?.status === "loading";
        return (
          <div key={key} className={ROW_CLASS}>
            <button
              type="button"
              aria-expanded={row.open}
              onClick={() => toggleDir(path)}
              className={cn(
                "flex min-w-0 flex-1 items-center gap-2 rounded-lg py-1.5 pr-2.5 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                entry.hidden && "text-muted-foreground",
              )}
              style={indent}
            >
              <HugeiconsIcon
                icon={ChevronRightStandardIcon}
                className={cn(
                  "size-3.5 shrink-0 text-muted-foreground/70 transition-transform",
                  row.open && "rotate-90",
                )}
              />
              <HugeiconsIcon
                icon={row.open ? FolderOpenIcon : Folder02Icon}
                strokeWidth={1.75}
                className="size-4 shrink-0 text-muted-foreground"
              />
              <span className="min-w-0 flex-1 truncate">{entry.name}</span>
              {loading && <Spinner className="size-3 text-muted-foreground" />}
            </button>
            <RowActions
              path={path}
              file={false}
              onCopy={copyPath}
              onAdd={addToChat}
            />
          </div>
        );
      }
      return (
        <div key={key} className={ROW_CLASS}>
          <div
            className={cn(
              "flex min-w-0 flex-1 items-center gap-2 py-1.5 pr-2.5 text-sm",
              entry.hidden && "text-muted-foreground",
            )}
            style={indent}
          >
            <span className="size-3.5 shrink-0" />
            <HugeiconsIcon
              icon={File01Icon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
            <span className="min-w-0 flex-1 truncate" title={path}>
              {entry.name}
            </span>
            <span className="shrink-0 text-ui-11 text-muted-foreground">
              {formatFileSize(entry.size)}
            </span>
          </div>
          <RowActions
            path={path}
            file={true}
            onCopy={copyPath}
            onAdd={addToChat}
          />
        </div>
      );
    });
  }

  return (
    <div className="mt-8 rounded-2xl bg-muted/30 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex h-9 min-w-0 flex-1 basis-48 items-center gap-2 rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255/calc(0.06*var(--contrast-wash-gain,1)))]">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-4 shrink-0 text-muted-foreground"
          />
          <input
            type="text"
            aria-label="Search files"
            placeholder="Search files"
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape" && query) {
                event.preventDefault();
                setQuery("");
              }
            }}
            className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
        </div>
        {repo && (
          <div className="flex shrink-0 items-center gap-0.5 rounded-full bg-muted p-0.5">
            {(["all", "changes"] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={view === value}
                data-active={view === value}
                onClick={() => setMode(value)}
                className="flex h-8 items-center gap-1.5 rounded-full px-3 text-ui-13 font-medium transition-colors data-[active=false]:text-muted-foreground data-[active=false]:hover:text-foreground data-[active=true]:bg-background data-[active=true]:text-foreground data-[active=true]:shadow-sm"
              >
                {value === "all" ? "All files" : "Changes"}
                {value === "changes" && changeCount > 0 && (
                  <span className="rounded-full bg-wash-10 px-1.5 text-ui-11 tabular-nums">
                    {changes.status === "ready" && changes.data.truncated
                      ? `${changeCount}+`
                      : changeCount}
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
        <IconButton
          label="Refresh"
          icon={Refresh01Icon}
          onClick={refresh}
          className="size-9 rounded-full"
        />
      </div>
      {rootListing && (
        <p
          className="mt-3 truncate px-2.5 text-ui-12 text-muted-foreground"
          title={rootListing.root}
        >
          {rootListing.root}
          {!rootListing.chosen && (
            <span className="text-muted-foreground/70"> · Managed folder</span>
          )}
        </p>
      )}
      <div className="mt-2 flex flex-col">{body}</div>
    </div>
  );
}
