// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  ChevronRight,
  File,
  FileCode,
  FileImage,
  Folder,
  RefreshCw,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

type Entry = { name: string; path: string; isDirectory: boolean; size: number };
type Listing = { entries: Entry[]; truncated: boolean };
type Preview =
  | { kind: "text"; content: string; truncated: boolean }
  | { kind: "image"; mimeType: string; data: string }
  | { kind: "unsupported"; message: string };

async function request<T>(
  endpoint: string,
  session: string,
  path: string,
  signal: AbortSignal,
  q = "",
): Promise<T> {
  const params = new URLSearchParams({ session, path, q });
  const response = await authFetch(
    `/api/workspace-files/${endpoint}?${params}`,
    { signal },
  );
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string" ? body.detail : "Could not load files",
    );
  return body;
}

function FileGlyph({ entry }: { entry: Entry }) {
  const Icon = entry.isDirectory
    ? Folder
    : /\.(png|jpe?g|gif|webp|bmp)$/i.test(entry.name)
      ? FileImage
      : /\.(py|js|jsx|ts|tsx|cpp|c|h|rs|json|html|css|sh)$/i.test(entry.name)
        ? FileCode
        : File;
  return <Icon className="size-4 shrink-0 text-muted-foreground" />;
}

function Directory({
  session,
  path,
  revision,
  selected,
  onSelect,
  expanded,
  onToggle,
}: {
  session: string;
  path: string;
  revision: number;
  selected: string | undefined;
  onSelect: (entry: Entry) => void;
  expanded: Set<string>;
  onToggle: (path: string) => void;
}) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    request<Listing>("files", session, path, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setListing(result);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : "Could not load folder",
          );
      });
    return () => controller.abort();
  }, [session, path, revision]);
  if (error)
    return (
      <p role="alert" className="p-2 text-xs text-destructive">
        {error}
      </p>
    );
  if (!listing)
    return (
      <div className="p-2">
        <Spinner />
      </div>
    );
  return (
    <ul className={path ? "ml-3 border-l pl-2" : ""}>
      {listing.entries.map((entry) => (
        <li key={entry.path}>
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent aria-[current=true]:bg-accent"
            title={entry.path}
            aria-expanded={
              entry.isDirectory ? expanded.has(entry.path) : undefined
            }
            aria-current={
              !entry.isDirectory && selected === entry.path ? "true" : undefined
            }
            onClick={() =>
              entry.isDirectory ? onToggle(entry.path) : onSelect(entry)
            }
          >
            {entry.isDirectory ? (
              <ChevronRight
                className={`size-3 shrink-0 transition-transform ${expanded.has(entry.path) ? "rotate-90" : ""}`}
              />
            ) : (
              <span className="w-3 shrink-0" />
            )}
            <FileGlyph entry={entry} />
            <span className="truncate">{entry.name}</span>
          </button>
          {entry.isDirectory && expanded.has(entry.path) && (
            <Directory
              session={session}
              path={entry.path}
              revision={revision}
              selected={selected}
              onSelect={onSelect}
              expanded={expanded}
              onToggle={onToggle}
            />
          )}
        </li>
      ))}
      {!listing.entries.length && (
        <li className="p-2 text-xs text-muted-foreground">
          {path ? "Empty folder" : "No files in this workspace yet"}
        </li>
      )}
      {listing.truncated && (
        <li className="p-2 text-xs text-muted-foreground">
          Folder limit reached, some entries are omitted.
        </li>
      )}
    </ul>
  );
}

function SearchResults({
  session,
  query,
  selected,
  onSelect,
}: {
  session: string;
  query: string;
  selected: string | undefined;
  onSelect: (entry: Entry) => void;
}) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      request<Listing>("files", session, "", controller.signal, query)
        .then((result) => {
          if (!controller.signal.aborted) setListing(result);
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setError(
              reason instanceof Error ? reason.message : "Search failed",
            );
        });
    }, 250);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [session, query]);
  if (error)
    return (
      <p role="alert" className="p-2 text-xs text-destructive">
        {error}
      </p>
    );
  if (!listing)
    return (
      <div className="p-2">
        <Spinner />
      </div>
    );
  return (
    <ul>
      {listing.entries.map((entry) => (
        <li key={entry.path}>
          <button
            type="button"
            onClick={() => onSelect(entry)}
            title={entry.path}
            aria-current={selected === entry.path ? "true" : undefined}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent aria-[current=true]:bg-accent"
          >
            <FileGlyph entry={entry} />
            <span className="truncate">{entry.path}</span>
          </button>
        </li>
      ))}
      {!listing.entries.length && (
        <li className="p-2 text-xs text-muted-foreground">No matching files</li>
      )}
      {listing.truncated && (
        <li className="p-2 text-xs text-muted-foreground">
          Search limit reached, results may be incomplete.
        </li>
      )}
    </ul>
  );
}

function FilePreview({ session, entry }: { session: string; entry: Entry }) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    request<Preview>("preview", session, entry.path, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setPreview(result);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : "Preview failed");
      });
    return () => controller.abort();
  }, [session, entry.path]);
  if (error)
    return (
      <p role="alert" className="text-sm text-destructive">
        {error}
      </p>
    );
  if (!preview) return <Spinner />;
  if (preview.kind === "text")
    return (
      <>
        {preview.truncated && (
          <p className="mb-2 text-xs text-muted-foreground">
            Preview limited to the first 256 KB.
          </p>
        )}
        <pre className="font-mono text-xs leading-relaxed">
          <code>{preview.content}</code>
        </pre>
      </>
    );
  if (preview.kind === "image")
    return (
      <img
        alt={entry.name}
        className="max-h-full max-w-full object-contain"
        onError={() => setError("Could not decode this image")}
        src={`data:${preview.mimeType};base64,${preview.data}`}
      />
    );
  return <p className="text-sm text-muted-foreground">{preview.message}</p>;
}

/** Key by sandbox session so a late response or selection cannot cross chat workspaces. */
export function WorkspaceFilesPanel({
  session,
  onClose,
}: { session: string; onClose: () => void }) {
  const [selected, setSelected] = useState<Entry | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [revision, setRevision] = useState(0);
  const toggle = (path: string) =>
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  return (
    <aside
      aria-label="Workspace files"
      className="flex h-full min-h-0 flex-col border-l bg-background"
    >
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <Folder className="size-4" />
        <h2 className="flex-1 text-sm font-medium">Files</h2>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Refresh files"
          onClick={() => setRevision((value) => value + 1)}
        >
          <RefreshCw className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close files"
          onClick={onClose}
        >
          <X className="size-4" />
        </Button>
      </header>
      <div className="p-3">
        <Input
          aria-label="Filter files"
          placeholder="Filter files…"
          value={query}
          maxLength={256}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <nav
        aria-label="Workspace folders"
        className={`${selected ? "h-2/5 shrink-0" : "flex-1"} min-h-0 overflow-auto px-2 pb-2`}
      >
        {query.trim() ? (
          <SearchResults
            key={`${session}:${query}:${revision}`}
            session={session}
            query={query.trim()}
            selected={selected?.path}
            onSelect={setSelected}
          />
        ) : (
          <Directory
            key={`${session}:${revision}`}
            session={session}
            path=""
            revision={revision}
            selected={selected?.path}
            onSelect={setSelected}
            expanded={expanded}
            onToggle={toggle}
          />
        )}
      </nav>
      {selected ? (
        <section
          aria-label="File preview"
          className="flex min-h-0 flex-1 flex-col border-t"
        >
          <div className="flex items-center gap-2 border-b px-3 py-2">
            <span
              title={selected.path}
              className="min-w-0 flex-1 truncate text-xs font-medium"
            >
              {selected.path}
            </span>
            <span className="text-xs text-muted-foreground">Read only</span>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Close preview"
              onClick={() => setSelected(null)}
            >
              <X className="size-3" />
            </Button>
          </div>
          <div className="min-h-0 flex-1 overflow-auto p-3">
            <FilePreview
              key={`${session}:${selected.path}:${revision}`}
              session={session}
              entry={selected}
            />
          </div>
        </section>
      ) : (
        <p className="border-t px-3 py-2 text-xs text-muted-foreground">
          Select a file to preview it.
        </p>
      )}
    </aside>
  );
}
