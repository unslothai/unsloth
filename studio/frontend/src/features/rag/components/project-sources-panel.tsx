// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/lib/toast";
import { FolderAddIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { SearchIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  announceProjectSourcesUpdated,
  invalidateProjectSources,
  listProjectDocuments,
  subscribeProjectSourcesUpdated,
} from "../api/rag-api";
import {
  SOURCE_SORT_LABELS,
  type SourceSortMode,
  filterSources,
  hasHiddenSources,
  isBulkRemovable,
  sortSources,
  visibleSources,
} from "../lib/source-list";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { LinkedFoldersManager } from "./linked-folders-manager";
import { useRagDocuments } from "./use-rag-documents";

const SORT_MODES: SourceSortMode[] = ["uploaded", "name", "size"];

/** Project "Sources" tab: documents indexed for retrieval in every chat that
 * belongs to the project. */
export function ProjectSourcesPanel({ projectId }: { projectId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lister = useCallback(
    () => listProjectDocuments(projectId),
    [projectId],
  );
  const { documents, loading, uploading, refresh, upload, remove } =
    useRagDocuments({ type: "project", projectId }, lister);

  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SourceSortMode>("uploaded");
  const [expanded, setExpanded] = useState(false);
  // "Select all" captures the ids it covered at the moment it was pressed.
  // Deriving the set live from what is currently visible would silently rewrite
  // what Bulk remove deletes: clearing a narrow search would extend the
  // selection to up to 25 unrelated sources, and expanding the collapsed list
  // would add every newly revealed one. Storing ids costs a prune (below) but
  // guarantees the user only deletes what was highlighted when they chose.
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [confirmingBulkRemove, setConfirmingBulkRemove] = useState(false);
  const [removing, setRemoving] = useState(false);
  // Mirrored in a ref so the drop handler reads the live value rather than the
  // one captured when the callback was created.
  const removingRef = useRef(false);
  removingRef.current = removing;
  // The project on screen now, read by the bulk loop, which outlives a
  // navigation. Deletes are keyed by document id alone, so a batch started here
  // keeps deleting this project's sources after the panel has been reused for
  // another one; its reconciliation must not then publish into that one.
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;

  // Invalidate the sources probe before each mutation so a chat sent mid-upload
  // cannot cache "no sources" for the probe's TTL, and announce after it, which
  // is the half other instances and other tabs listen for. Announcing before
  // would refetch and resurrect the row this panel has already dropped.
  const handleFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      // Not during a bulk remove: ingestion dedups by content hash, so
      // re-uploading a file that matches a document still queued for deletion
      // resolves to that same id, and the loop then deletes it. The upload
      // would report success with nothing left behind.
      if (removingRef.current) {
        toast.info("Finish removing sources before adding more");
        return;
      }
      invalidateProjectSources(projectId);
      await upload(files);
      announceProjectSourcesUpdated(projectId);
      // An upload outlives a navigation the same way a bulk remove does, and the
      // announce above already refreshes whoever is showing this project.
      if (projectIdRef.current !== projectId) return;
      // Fetch the server rows for what was just uploaded. Job completion only
      // patches status/progress, so without this an upload keeps the optimistic
      // row's missing createdAt and sizeBytes: sorting would then rank it last
      // and, past the collapse limit, hide the file the user just added. The
      // indexing poll covers slower uploads, but one that finishes inside the
      // first interval would never be reconciled.
      await refresh({ quiet: true });
    },
    [projectId, upload, refresh],
  );

  const handleRemove = useCallback(
    async (documentId: string) => {
      invalidateProjectSources(projectId);
      await remove(documentId);
      announceProjectSourcesUpdated(projectId);
    },
    [projectId, remove],
  );
  const handleLinkedSourcesChanged = useCallback(() => {
    announceProjectSourcesUpdated(projectId);
    void refresh({ quiet: true });
  }, [projectId, refresh]);

  // External mutators (sidebar/thread saves, deletes elsewhere) announce when
  // they are done; refresh the mounted list so a source saved from a chat shows
  // up here without a remount. The list only polls while a row it already knows
  // is indexing, so nothing else would ever fetch it.
  useEffect(
    () =>
      subscribeProjectSourcesUpdated(projectId, () => {
        void refresh({ quiet: true });
      }),
    [projectId, refresh],
  );

  // Switching projects must not carry a selection (or a search) onto a list of
  // different documents.
  useEffect(() => {
    setSelectedIds(new Set());
    setQuery("");
    setExpanded(false);
  }, [projectId]);

  const searching = query.trim() !== "";
  const matched = useMemo(
    () => sortSources(filterSources(documents, query), sortMode),
    [documents, query, sortMode],
  );
  const shown = useMemo(
    () => visibleSources(matched, { expanded, searching }),
    [matched, expanded, searching],
  );
  const hiddenCount = hasHiddenSources(matched.length, { expanded, searching })
    ? matched.length - shown.length
    : 0;

  // Everything the current search matches, not just the rendered slice.
  // Collapsing is a display cap: "Select all" over 27 sources means all 27, not
  // the 25 that happen to be on screen.
  const selectable = useMemo(() => matched.filter(isBulkRemovable), [matched]);
  const selectAll = selectedIds.size > 0;

  const selectVisible = useCallback(() => {
    setSelectedIds(new Set(selectable.map((doc) => doc.id)));
  }, [selectable]);
  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // Drop ids that left the list (deleted elsewhere, unlinked, or started
  // indexing) so the count never promises more than the action can deliver.
  // Narrowing the search does not deselect: the highlight is hidden with the
  // chip and comes back when the query is cleared.
  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const live = new Set(documents.filter(isBulkRemovable).map((d) => d.id));
      const next = new Set([...prev].filter((id) => live.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [documents]);

  const selectedCount = selectedIds.size;

  async function handleBulkRemove() {
    const ids = [...selectedIds];
    if (ids.length === 0) return;
    // The project this batch is for. Deletes are keyed by document id, so the
    // loop finishes against this project wherever the user navigates; only the
    // parts that write to the panel are abandoned below.
    const batchProjectId = projectId;
    setRemoving(true);
    invalidateProjectSources(batchProjectId);
    let failures = 0;
    try {
      // Sequential: the backend deletes rewrite the same scope, and one at a
      // time keeps a partial failure attributable to a single document.
      for (const id of ids) {
        if (!(await remove(id))) failures += 1;
      }
    } finally {
      invalidateProjectSources(batchProjectId);
      announceProjectSourcesUpdated(batchProjectId);
      setRemoving(false);
    }
    // The panel is not keyed by project, so it is reused across a navigation and
    // this continuation can resume against a different one. The hook refuses to
    // publish another scope's rows, so what is left here is the panel's own
    // state: a selection and a failure toast that belong to a project the user
    // is no longer looking at, and a refetch of a list nothing is showing. The
    // announce above has already told the project that actually changed.
    if (projectIdRef.current !== batchProjectId) return;
    // Clear the selection either way: with no per-file toggle the user cannot
    // curate a retry set, so leaving rows selected would only misreport what is
    // about to be deleted.
    clearSelection();
    if (failures > 0) {
      toast.error(
        failures === 1
          ? "1 source could not be removed"
          : `${failures} sources could not be removed`,
      );
    }
    // Reconcile after every batch, not only after a failure. A poll or an
    // external refresh overlapping the loop can read a document before its
    // DELETE lands and re-insert the optimistically removed row afterwards, so
    // an all-successful batch can still leave a deleted source on screen.
    await refresh({ quiet: true });
  }

  const empty = documents.length === 0;

  return (
    <div
      className="mt-8"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        void handleFiles(Array.from(e.dataTransfer.files ?? []));
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple={true}
        accept={RAG_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          void handleFiles(files);
        }}
      />
      <div className="mb-4 rounded-[22px] bg-muted/30 px-5 py-4">
        <LinkedFoldersManager
          scope={{ type: "project", id: projectId }}
          compact={true}
          onSourcesChanged={handleLinkedSourcesChanged}
        />
      </div>
      {empty ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-[26px] bg-muted/30 px-6 py-16 text-center">
          <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-6"
            />
          </span>
          <div className="space-y-1">
            <p className="text-ui-15 font-semibold text-foreground">
              Give this project context
            </p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Upload PDFs, docs, or text. Every chat in this project can use
              them.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            className="mt-1 border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
            disabled={uploading || loading}
            onClick={() => fileInputRef.current?.click()}
          >
            Add sources
          </Button>
          <p className="text-ui-11 text-muted-foreground">Or drop files here</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4 rounded-[26px] bg-muted/30 px-6 py-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {documents.length === 1
                ? "1 source"
                : `${documents.length} sources`}
            </p>
            <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
              <div className="relative min-w-[10rem] flex-1 sm:max-w-[16rem]">
                <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search sources"
                  aria-label="Search sources by name"
                  className="h-8 border-none bg-background pl-8 text-sm shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
                />
              </div>
              <Select
                value={sortMode}
                onValueChange={(value) => setSortMode(value as SourceSortMode)}
              >
                <SelectTrigger
                  size="sm"
                  aria-label="Sort sources"
                  className="h-8 w-[9.5rem] border-none bg-background text-sm shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SORT_MODES.map((mode) => (
                    <SelectItem key={mode} value={mode}>
                      {SOURCE_SORT_LABELS[mode]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="border-none bg-background text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] hover:bg-background/80 dark:bg-card dark:shadow-none dark:hover:bg-accent/50"
                disabled={uploading || removing}
                onClick={() => fileInputRef.current?.click()}
              >
                Add sources
              </Button>
            </div>
          </div>

          {selectedCount > 0 ? (
            <div className="flex flex-wrap items-center gap-2 rounded-[18px] bg-background/80 px-3 py-2 dark:bg-card/80">
              <span className="text-sm text-muted-foreground">
                {selectedCount === 1
                  ? "1 item selected"
                  : `${selectedCount} items selected`}
              </span>
              <Button
                type="button"
                size="sm"
                variant="destructive"
                disabled={removing}
                className="ml-auto"
                onClick={() => setConfirmingBulkRemove(true)}
              >
                {removing ? "Removing..." : "Bulk remove"}
              </Button>
            </div>
          ) : null}

          {selectable.length > 0 ? (
            <div className="-mb-1">
              <button
                type="button"
                disabled={removing}
                onClick={selectAll ? clearSelection : selectVisible}
                className="text-ui-11 text-muted-foreground underline-offset-2 hover:text-foreground hover:underline disabled:opacity-50"
              >
                {/* Covers every match, not just the rendered slice; the header
                  already states the total. */}
                {selectAll ? "Deselect all" : "Select all"}
              </button>
            </div>
          ) : null}

          {matched.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No sources match &quot;{query.trim()}&quot;.
            </p>
          ) : (
            <div className="flex flex-row flex-wrap items-center gap-1.5">
              {shown.map((doc) => {
                const removable = isBulkRemovable(doc);
                return (
                  <DocumentStatusChip
                    key={doc.id}
                    filename={doc.filename}
                    status={doc.status}
                    progress={doc.progress}
                    error={doc.error}
                    selected={selectedIds.has(doc.id)}
                    // A concurrent single delete during a batch would race the
                    // loop and make it report a false failure on the 404.
                    onRemove={
                      removable && !removing
                        ? () => void handleRemove(doc.id)
                        : undefined
                    }
                  />
                );
              })}
            </div>
          )}

          {hiddenCount > 0 ? (
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="self-center text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            >
              Show all {matched.length} sources
            </button>
          ) : expanded && !searching ? (
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="self-center text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            >
              Show fewer
            </button>
          ) : null}
        </div>
      )}

      <AlertDialog
        open={confirmingBulkRemove}
        onOpenChange={(open) => {
          if (!open) setConfirmingBulkRemove(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Remove{" "}
              {selectedCount === 1 ? "source" : `${selectedCount} sources`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {selectedCount === 1
                ? "The file and its indexed content are removed from this project."
                : "The files and their indexed content are removed from this project."}{" "}
              This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmingBulkRemove(false);
                void handleBulkRemove();
              }}
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
