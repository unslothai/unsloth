// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { toast } from "@/lib/toast";

import {
  deleteDiffusionDatasetImage,
  diffusionDatasetImageUrl,
  fetchGalleryObjectUrl,
  imageRecordsOnly,
  listDiffusionDatasetImages,
} from "../api";

// Rows of 5 in the settings column, so this fills two and a bit before "+N more".
const MAX_TILES = 12;

// A single thumbnail tile: auth-fetches its object URL and revokes it on unmount. The tile opens
// the labeling grid; its own corner button removes the image.
function ShowcaseTile({
  dataset,
  filename,
  onBrowse,
  onRemove,
}: {
  dataset: string;
  filename: string;
  onBrowse: () => void;
  onRemove: (filename: string) => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let obj: string | null = null;
    let cancelled = false;
    fetchGalleryObjectUrl(diffusionDatasetImageUrl(dataset, filename, 256))
      // The fetch returns the blob's size alongside the URL for the gallery's byte budget; a single
      // tile only needs the URL.
      .then(({ url: u }) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        obj = u;
        setUrl(u);
      })
      .catch(() => {
        /* a missing thumbnail just leaves the placeholder */
      });
    return () => {
      cancelled = true;
      if (obj) URL.revokeObjectURL(obj);
    };
  }, [dataset, filename]);

  const remove = useCallback(async () => {
    setDeleting(true);
    try {
      await deleteDiffusionDatasetImage(dataset, filename);
      onRemove(filename);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to delete image");
      setDeleting(false);
    }
  }, [dataset, filename, onRemove]);

  return (
    <div className="group/thumb relative size-14 shrink-0">
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            onClick={onBrowse}
            className="size-full overflow-hidden rounded-[8px] bg-muted outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {url ? (
              <img
                src={url}
                alt={filename}
                className="size-full object-cover transition-opacity group-hover/thumb:opacity-70"
              />
            ) : (
              <span className="flex size-full items-center justify-center">
                <Spinner className="size-3.5 text-muted-foreground" />
              </span>
            )}
          </button>
        </TooltipTrigger>
        <TooltipContent>Browse and caption these images</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            onClick={remove}
            disabled={deleting}
            aria-label={`Remove ${filename}`}
            className="absolute right-0.5 top-0.5 flex size-5 items-center justify-center rounded-full bg-background/90 text-muted-foreground opacity-0 shadow-sm ring-1 ring-border backdrop-blur-sm transition-opacity hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/thumb:opacity-100"
          >
            {deleting ? (
              <Spinner className="size-2.5" />
            ) : (
              <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
            )}
          </button>
        </TooltipTrigger>
        <TooltipContent>Remove this image from the dataset</TooltipContent>
      </Tooltip>
    </div>
  );
}

// A compact preview of a dataset's images: up to MAX_TILES sampled thumbnails plus a "+N more"
// tile that opens the full labeling grid. Refreshes on selection or `refreshKey` change.
export function DatasetShowcase({
  dataset,
  imageCount,
  refreshKey = 0,
  onBrowse,
  onChanged,
}: {
  dataset: string;
  imageCount: number;
  refreshKey?: number;
  onBrowse: () => void;
  // Fired after a delete so the parent can refresh its dataset counts.
  onChanged?: () => void;
}) {
  const [names, setNames] = useState<string[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    setNames(null);
    listDiffusionDatasetImages(dataset)
      .then((r) => {
        if (cancelled) return;
        // Sample up to MAX_TILES evenly across the folder so the strip represents the whole set, not
        // just the first few files. Clips have no thumbnail endpoint, so the strip shows images only.
        const all = imageRecordsOnly(r.images).map((im) => im.filename);
        if (all.length <= MAX_TILES) {
          setNames(all);
          return;
        }
        const stride = all.length / MAX_TILES;
        const picked: string[] = [];
        for (let i = 0; i < MAX_TILES; i++) picked.push(all[Math.floor(i * stride)]);
        setNames(picked);
      })
      .catch(() => {
        if (!cancelled) setNames([]);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, refreshKey]);

  const onRemoved = useCallback(
    (filename: string) => {
      setNames((cur) => (cur ? cur.filter((n) => n !== filename) : cur));
      onChanged?.();
    },
    [onChanged],
  );

  if (names !== null && names.length === 0) return null;

  const remaining = imageCount - (names?.length ?? 0);

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-1.5">
      {/* Wraps onto more rows rather than scrolling sideways. */}
      <div className="flex flex-wrap items-center gap-2.5">
        {names === null ? (
          <div className="flex h-14 items-center gap-2 px-2 text-ui-11 text-muted-foreground">
            <Spinner className="size-3.5" /> Loading preview...
          </div>
        ) : (
          <>
            {names.map((n) => (
              <ShowcaseTile
                key={n}
                dataset={dataset}
                filename={n}
                onBrowse={onBrowse}
                onRemove={onRemoved}
              />
            ))}
            {remaining > 0 && (
              <button
                type="button"
                onClick={onBrowse}
                className="flex size-14 shrink-0 flex-col items-center justify-center rounded-[8px] bg-muted text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
              >
                <span className="text-sm font-medium">+{remaining}</span>
                <span className="text-ui-9">more</span>
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
