// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import {
  diffusionDatasetImageUrl,
  fetchGalleryObjectUrl,
  listDiffusionDatasetImages,
} from "../api";

const MAX_TILES = 8;

// A single thumbnail tile: auth-fetches its object URL and revokes it on unmount.
function ShowcaseTile({ dataset, filename }: { dataset: string; filename: string }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let obj: string | null = null;
    let cancelled = false;
    fetchGalleryObjectUrl(diffusionDatasetImageUrl(dataset, filename, 256))
      // The fetch returns the blob's size alongside the URL for the gallery's byte budget; a single tile only needs the URL.
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

  return (
    <div className="size-14 shrink-0 overflow-hidden rounded-[8px] bg-muted">
      {url ? (
        <img src={url} alt={filename} className="size-full object-cover" />
      ) : (
        <div className="flex size-full items-center justify-center">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}

// A compact preview strip of a dataset's images: up to 8 sampled thumbnails plus a "+N more" tile that opens the full
// labeling grid. Refreshes on selection or `refreshKey` change; the whole strip is a button to Review captions.
export function DatasetShowcase({
  dataset,
  imageCount,
  refreshKey = 0,
  onBrowse,
}: {
  dataset: string;
  imageCount: number;
  refreshKey?: number;
  onBrowse: () => void;
}) {
  const [names, setNames] = useState<string[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    setNames(null);
    listDiffusionDatasetImages(dataset)
      .then((r) => {
        if (cancelled) return;
        // Sample up to MAX_TILES evenly across the folder so the strip represents the whole set, not just the first few files.
        const all = r.images.map((im) => im.filename);
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

  if (names !== null && names.length === 0) return null;

  const remaining = imageCount - (names?.length ?? 0);

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={onBrowse}
          className="hover-scrollbar flex w-full items-center gap-1.5 overflow-x-auto rounded-lg border border-border bg-muted/20 p-1.5 text-left transition-colors hover:border-foreground/20"
        >
          {names === null ? (
            <div className="flex h-14 items-center gap-2 px-2 text-ui-11 text-muted-foreground">
              <Spinner className="size-3.5" /> Loading preview...
            </div>
          ) : (
            <>
              {names.map((n) => (
                <ShowcaseTile key={n} dataset={dataset} filename={n} />
              ))}
              {remaining > 0 && (
                <div className="flex size-14 shrink-0 flex-col items-center justify-center rounded-md bg-muted text-muted-foreground">
                  <span className="text-sm font-medium">+{remaining}</span>
                  <span className="text-ui-9">more</span>
                </div>
              )}
            </>
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>Browse and caption these images</TooltipContent>
    </Tooltip>
  );
}
