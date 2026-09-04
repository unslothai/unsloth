// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

import {
  type DiffusionDatasetExample,
  type DiffusionDatasetImportResult,
  importDiffusionDatasetExample,
} from "../api";

// Best-effort preview thumbnails from the public HF datasets-server, cached per repo at module
// level so re-renders do not refetch. A repo the server cannot serve resolves to an empty list
// and the card renders without previews.
const _previewCache = new Map<string, Promise<string[]>>();

async function fetchPreviews(repo: string): Promise<string[]> {
  const cached = _previewCache.get(repo);
  if (cached) return cached;
  const p = (async () => {
    try {
      const res = await fetch(
        `https://datasets-server.huggingface.co/first-rows?dataset=${encodeURIComponent(
          repo,
        )}&config=default&split=train`,
      );
      if (!res.ok) return [];
      const data = (await res.json()) as {
        features?: { name: string; type?: { _type?: string } }[];
        rows?: { row: Record<string, unknown> }[];
      };
      const imageCol = data.features?.find((f) => f.type?._type === "Image")?.name;
      if (!imageCol || !data.rows) return [];
      const urls: string[] = [];
      for (const r of data.rows) {
        const cell = r.row[imageCol] as { src?: string } | undefined;
        if (cell?.src) urls.push(cell.src);
        if (urls.length >= 3) break;
      }
      return urls;
    } catch {
      return [];
    }
  })();
  _previewCache.set(repo, p);
  return p;
}

// "Dog (DreamBooth subject)" -> "Dog": one-line rows drop the parenthetical.
export function shortExampleLabel(label: string): string {
  return label.replace(/\s*\(.*$/, "");
}

function ExamplePreviews({ repo }: { repo: string }) {
  const [urls, setUrls] = useState<string[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    void fetchPreviews(repo).then((u) => {
      if (!cancelled) setUrls(u);
    });
    return () => {
      cancelled = true;
    };
  }, [repo]);

  if (!urls || urls.length === 0) return null;
  return (
    <div className="flex gap-1">
      {urls.map((u) => (
        <div key={u} className="h-9 w-12 shrink-0 overflow-hidden rounded-[8px] bg-muted">
          <img src={u} alt="" loading="lazy" className="size-full object-cover" />
        </div>
      ))}
    </div>
  );
}

// One-click example-dataset importers. Each card shows the license before import plus preview
// thumbnails; on success the parent refreshes its dataset list and selects the folder. One card
// per row: the config column is narrow, so a two-column grid wrapped titles one word per line.
export function ExampleDatasetCards({
  examples,
  busyId,
  onImport,
  className,
}: {
  examples: DiffusionDatasetExample[];
  busyId: string | null;
  onImport: (ex: DiffusionDatasetExample) => void;
  /** Spacing above, which the caller sets: what sits above this block varies. */
  className?: string;
}) {
  if (examples.length === 0) return null;

  return (
    <div className={cn("grid gap-2", className)}>
      <span className="text-ui-11 font-medium text-muted-foreground">
        Or start from an example dataset
      </span>
      <div className="grid gap-2">
        {examples.map((ex) => (
          <div
            key={ex.id}
            className="flex min-w-0 flex-col gap-2.5 rounded-lg border border-border px-4 py-3"
          >
            <div className="flex min-w-0 items-center gap-1.5">
              {/* Both are truncated, so the full text lives in a tooltip. */}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <span className="min-w-0 flex-1 truncate text-xs font-medium">
                    {shortExampleLabel(ex.label)}
                  </span>
                </TooltipTrigger>
                <TooltipContent>{ex.label}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <span className="max-w-[110px] shrink truncate rounded-full bg-secondary px-2 py-0.5 text-ui-10 font-normal text-secondary-foreground">
                    {ex.license}
                  </span>
                </TooltipTrigger>
                <TooltipContent>{ex.license}</TooltipContent>
              </Tooltip>
            </div>
            <p className="line-clamp-2 text-ui-11 leading-snug text-muted-foreground">
              {ex.description}
            </p>
            {/* The card's action, so: bottom right, on the thumbnail row. */}
            <div className="flex min-w-0 items-end justify-between gap-3">
              <ExamplePreviews repo={ex.repo} />
              <Button
                type="button"
                size="sm"
                variant="outline"
                // ml-auto keeps it right-aligned even when the previews are missing.
                className="ml-auto h-7 shrink-0 px-3 text-xs"
                onClick={() => onImport(ex)}
                disabled={busyId !== null}
              >
                {busyId === ex.id ? "Importing..." : "Import"}
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Shared import helper so the panel's dropdown and the cards import identically.
export async function runExampleImport(
  ex: DiffusionDatasetExample,
): Promise<DiffusionDatasetImportResult> {
  const res = await importDiffusionDatasetExample(ex.id);
  toast.success(
    res.imported > 0
      ? `Imported ${res.image_count} images into "${res.name}"`
      : `"${res.name}" already imported (${res.image_count} images)`,
  );
  return res;
}
