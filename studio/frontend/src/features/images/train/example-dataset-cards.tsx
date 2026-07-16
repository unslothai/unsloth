// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

import {
  type DiffusionDatasetExample,
  type DiffusionDatasetImportResult,
  importDiffusionDatasetExample,
} from "../api";

// Best-effort preview thumbnails from the public HF datasets-server. Cached per repo (module-level)
// so re-renders/re-mounts don't refetch. A repo the server can't serve (e.g. diffusers/dog-example)
// resolves to an empty list and the card renders without previews -- the import still works.
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
        <div key={u} className="size-10 shrink-0 overflow-hidden rounded-md bg-muted">
          <img src={u} alt="" loading="lazy" className="size-full object-cover" />
        </div>
      ))}
    </div>
  );
}

// One-click example-dataset importers. Each card shows the license (terms before import) plus a
// few preview thumbnails. On success the parent refreshes its dataset list and selects the
// imported folder (and can seed the trigger prompt from suggested_trigger).
//
// Layout: one card per row (the config column is narrow, so a two-column grid wrapped titles
// one word per line and let the long license text overrun into the next card).
export function ExampleDatasetCards({
  examples,
  busyId,
  onImport,
}: {
  examples: DiffusionDatasetExample[];
  busyId: string | null;
  onImport: (ex: DiffusionDatasetExample) => void;
}) {
  if (examples.length === 0) return null;

  return (
    <div className="grid gap-2">
      <span className="text-[11px] font-medium text-muted-foreground">
        Or start from an example dataset
      </span>
      <div className="grid gap-2">
        {examples.map((ex) => (
          <div
            key={ex.id}
            className="flex min-w-0 flex-col gap-2 rounded-lg border border-border p-2.5"
          >
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="min-w-0 flex-1 truncate text-xs font-medium">{ex.label}</span>
                  <span
                    className="max-w-[110px] shrink truncate rounded-full bg-secondary px-2 py-0.5 text-[10px] font-normal text-secondary-foreground"
                    title={ex.license}
                  >
                    {ex.license}
                  </span>
                </div>
                <p className="line-clamp-2 text-[11px] leading-snug text-muted-foreground">
                  {ex.description}
                </p>
              </div>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="h-7 shrink-0 self-center px-3 text-xs"
                onClick={() => onImport(ex)}
                disabled={busyId !== null}
              >
                {busyId === ex.id ? "Importing..." : "Import"}
              </Button>
            </div>
            <ExamplePreviews repo={ex.repo} />
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
