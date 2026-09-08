// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import {
  Cancel01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChevronDown,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

// One batch of images shown at a time in the labeling grid; larger sets page with < >.
const PAGE_SIZE = 24;

import {
  type DiffusionDatasetImageRecord,
  deleteDiffusionDatasetImage,
  diffusionDatasetImageUrl,
  fetchGalleryObjectUrl,
  imageRecordsOnly,
  listDiffusionDatasetImages,
  setDiffusionDatasetCaption,
} from "../api";

// One tile: an auth-fetched thumbnail (object URL, revoked on unmount) plus a caption Textarea
// saved on blur. Uncaptioned tiles get a highlighted ring so a user labeling a small set sees
// what still needs a caption.
function LabelTile({
  dataset,
  record,
  onSaved,
  onDeleted,
}: {
  dataset: string;
  record: DiffusionDatasetImageRecord;
  onSaved: (rec: DiffusionDatasetImageRecord) => void;
  onDeleted: (filename: string) => void;
}) {
  const [thumb, setThumb] = useState<string | null>(null);
  const [caption, setCaption] = useState(record.caption ?? "");
  const [saving, setSaving] = useState(false);
  const [savedTick, setSavedTick] = useState(false);
  const [deleting, setDeleting] = useState(false);
  // The last caption we persisted, so blur only writes when the text actually changed.
  const persisted = useRef(record.caption ?? "");

  // Load the thumbnail once; revoke the object URL on unmount to avoid a leak.
  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    fetchGalleryObjectUrl(diffusionDatasetImageUrl(dataset, record.filename, 256))
      // The fetch returns the blob's size alongside the URL for the gallery's byte budget; a dataset
      // thumbnail only needs the URL.
      .then(({ url: u }) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        url = u;
        setThumb(u);
      })
      .catch(() => {
        /* a missing thumbnail just leaves the placeholder */
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [dataset, record.filename]);

  const save = useCallback(async () => {
    const next = caption.trim();
    if (next === persisted.current.trim()) return;
    setSaving(true);
    try {
      const updated = await setDiffusionDatasetCaption(dataset, record.filename, next);
      persisted.current = updated.caption ?? "";
      setCaption(updated.caption ?? "");
      onSaved(updated);
      setSavedTick(true);
      window.setTimeout(() => setSavedTick(false), 1500);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save caption");
    } finally {
      setSaving(false);
    }
  }, [caption, dataset, record.filename, onSaved]);

  const remove = useCallback(async () => {
    setDeleting(true);
    try {
      await deleteDiffusionDatasetImage(dataset, record.filename);
      onDeleted(record.filename);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to delete image");
      setDeleting(false);
    }
  }, [dataset, record.filename, onDeleted]);

  const uncaptioned = caption.trim().length === 0;

  return (
    // Named group: a bare one would also match the scroller and reveal every tile's Remove.
    <div className="group/tile flex flex-col overflow-hidden rounded-[10px] border border-border transition-colors hover:border-foreground/20">
      <div className="relative aspect-square w-full overflow-hidden bg-muted">
        {thumb ? (
          <img src={thumb} alt={record.filename} className="size-full object-cover" />
        ) : (
          <div className="flex size-full items-center justify-center">
            <Spinner className="size-4 text-muted-foreground" />
          </div>
        )}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              onClick={remove}
              disabled={deleting}
              aria-label="Remove image"
              className="absolute right-1.5 top-1.5 flex size-6 items-center justify-center rounded-full bg-background/85 text-muted-foreground opacity-0 shadow-sm backdrop-blur-sm transition-opacity hover:bg-background hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/tile:opacity-100"
            >
              {deleting ? (
                <Spinner className="size-3" />
              ) : (
                <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>Remove this image from the dataset</TooltipContent>
        </Tooltip>
      </div>
      <div className="flex min-w-0 flex-col gap-1.5 p-2">
        {/* Textarea's md:text-sm outsizes a tile, so pin the size at both widths. */}
        <Textarea
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          onBlur={() => void save()}
          rows={2}
          spellCheck={false}
          placeholder="Describe this image..."
          className="min-h-[2.75rem] rounded-sm px-2 py-1.5 text-ui-11! leading-snug md:text-ui-11!"
          aria-label={`Caption for ${record.filename}`}
        />
        {/* No leading-none: truncate's overflow would crop the descenders. */}
        <div className="flex min-w-0 items-center gap-1.5 text-ui-10 text-muted-foreground">
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <span className="min-w-0 flex-1 truncate">{record.filename}</span>
            </TooltipTrigger>
            <TooltipContent>{record.filename}</TooltipContent>
          </Tooltip>
          {saving ? (
            <span className="shrink-0 whitespace-nowrap">Saving</span>
          ) : savedTick ? (
            <span className="shrink-0 whitespace-nowrap text-emerald-600">Saved</span>
          ) : uncaptioned ? (
            <span className="shrink-0 whitespace-nowrap text-amber-600">No caption</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// A responsive grid over a dataset folder's images with per-image caption editing. Fetches the
// list on open and whenever `refreshKey` changes.
export function DatasetLabelingGrid({
  dataset,
  refreshKey = 0,
  onCountsChanged,
}: {
  dataset: string;
  refreshKey?: number;
  // Fired after a caption save or delete so the parent can refresh dataset counts.
  onCountsChanged?: () => void;
}) {
  const [records, setRecords] = useState<DiffusionDatasetImageRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setRecords(null);
    setError(null);
    setPage(0); // a new dataset (or refresh) always starts at the first batch
    listDiffusionDatasetImages(dataset)
      .then((r) => {
        // clips list alongside images but have no thumbnail endpoint; the grid is image-only.
        if (!cancelled) setRecords(imageRecordsOnly(r.images));
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to list images");
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, refreshKey]);

  const onSaved = useCallback(
    (rec: DiffusionDatasetImageRecord) => {
      setRecords((cur) =>
        cur ? cur.map((r) => (r.filename === rec.filename ? rec : r)) : cur,
      );
      onCountsChanged?.();
    },
    [onCountsChanged],
  );

  const onDeleted = useCallback(
    (filename: string) => {
      setRecords((cur) => (cur ? cur.filter((r) => r.filename !== filename) : cur));
      onCountsChanged?.();
    },
    [onCountsChanged],
  );

  if (error) {
    return <p className="text-ui-11 text-destructive">{error}</p>;
  }
  if (records === null) {
    return (
      <div className="flex items-center gap-2 py-4 text-ui-11 text-muted-foreground">
        <Spinner className="size-3.5" /> Loading images...
      </div>
    );
  }
  if (records.length === 0) {
    return <p className="text-ui-11 text-muted-foreground">This dataset has no images yet.</p>;
  }

  const uncaptioned = records.filter((r) => !r.caption || r.caption.trim() === "").length;
  const total = records.length;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * PAGE_SIZE;
  const pageRecords = records.slice(start, start + PAGE_SIZE);

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center justify-between gap-2 text-ui-11 text-muted-foreground">
        <span className="min-w-0 truncate">
          {total} image{total === 1 ? "" : "s"}
          {uncaptioned > 0 ? ` · ${uncaptioned} without a caption` : " · all captioned"}
        </span>
        {pageCount > 1 && (
          <div className="flex shrink-0 items-center gap-1">
            <span className="tabular-nums">
              {start + 1}-{Math.min(start + PAGE_SIZE, total)} of {total}
            </span>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-6"
              disabled={clampedPage === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              aria-label="Previous images"
            >
              <ChevronLeftIcon className="size-3.5" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-6"
              disabled={clampedPage >= pageCount - 1}
              onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
              aria-label="Next images"
            >
              <ChevronRightIcon className="size-3.5" />
            </Button>
          </div>
        )}
      </div>
      {/* Two columns at any width: the column is fixed, so viewport breakpoints do not apply. */}
      {/* auto-rows-min: past max-h the height is definite and auto rows split it, flattening every
          tile until its thumbnail collapses. */}
      <div className="hover-scrollbar grid max-h-[420px] auto-rows-min grid-cols-2 gap-2.5 overflow-y-auto pb-0.5 pr-1">
        {pageRecords.map((r) => (
          <LabelTile
            key={r.filename}
            dataset={dataset}
            record={r}
            onSaved={onSaved}
            onDeleted={onDeleted}
          />
        ))}
      </div>
    </div>
  );
}

// A tiny standalone control used by the panel to keep grid-refresh wiring in one place.
export function LabelingGridToggle({
  count,
  open,
  onToggle,
}: {
  count: number;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      // ghost fills on aria-expanded; the chevron shows the state, so drop the fill.
      className="h-7 w-fit gap-1 px-1.5 text-ui-11 font-medium text-muted-foreground aria-expanded:bg-transparent hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"
      onClick={onToggle}
      aria-expanded={open}
    >
      {open ? "Hide captions" : `Review captions (${count} image${count === 1 ? "" : "s"})`}
      <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
    </Button>
  );
}
