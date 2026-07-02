// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

import {
  type DiffusionDatasetExample,
  type DiffusionDatasetImportResult,
  importDiffusionDatasetExample,
  listDiffusionDatasetExamples,
} from "../api";

// One-click example-dataset importers. Each card shows the license verbatim so users see
// the terms before importing. On success the parent refreshes its dataset list and selects
// the imported folder (and can seed the trigger prompt from suggested_trigger).
export function ExampleDatasetCards({
  onImported,
}: {
  onImported: (
    result: DiffusionDatasetImportResult,
    example: DiffusionDatasetExample,
  ) => void;
}) {
  const [examples, setExamples] = useState<DiffusionDatasetExample[] | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDiffusionDatasetExamples()
      .then((list) => {
        if (!cancelled) setExamples(list);
      })
      .catch(() => {
        if (!cancelled) setExamples([]); // older backend: just hide the cards
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const doImport = useCallback(
    async (ex: DiffusionDatasetExample) => {
      setBusyId(ex.id);
      try {
        const res = await importDiffusionDatasetExample(ex.id);
        toast.success(
          res.imported > 0
            ? `Imported ${res.image_count} images into "${res.name}"`
            : `"${res.name}" already imported (${res.image_count} images)`,
        );
        onImported(res, ex);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Import failed");
      } finally {
        setBusyId(null);
      }
    },
    [onImported],
  );

  if (!examples || examples.length === 0) return null;

  return (
    <div className="grid gap-2">
      <span className="text-[11px] text-muted-foreground">
        Or start from an example dataset:
      </span>
      <div className="grid gap-2 sm:grid-cols-2">
        {examples.map((ex) => (
          <div
            key={ex.id}
            className="flex flex-col gap-1.5 rounded-lg border border-border p-2.5"
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-xs font-medium">{ex.label}</span>
              <Badge variant="secondary" className="shrink-0 text-[10px]">
                {ex.license}
              </Badge>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              {ex.description}
            </p>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="h-7 w-fit text-xs"
              onClick={() => void doImport(ex)}
              disabled={busyId !== null}
            >
              {busyId === ex.id ? "Importing..." : "Import"}
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}
