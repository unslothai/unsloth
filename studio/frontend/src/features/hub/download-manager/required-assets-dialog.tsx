// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useId, useState } from "react";
import { Checkbox as CheckboxPrimitive } from "radix-ui";
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
import {
  additionalAssetDownloads,
  assetLabel,
  downloadBytes,
  formatDownloadBytes,
  type PlannedDownloadEntry,
} from "./required-assets";

export function RequiredAssetsDownloadDialog({
  entries,
  onConfirm,
  onCancel,
  modelLabel,
  checking = false,
}: {
  checking?: boolean;
  entries: readonly PlannedDownloadEntry[] | null;
  onConfirm: (includeAssets: boolean) => void;
  onCancel: () => void;
  modelLabel?: string;
}) {
  // Keep one portal/content mounted through the metadata check.
  return (
    <AlertDialog
      open={checking || entries !== null}
      onOpenChange={(open) => {
        if (!open) onCancel();
      }}
    >
      {(checking || entries !== null) && (
        <DownloadChoice
          entries={entries ?? []}
          checking={checking}
          onConfirm={onConfirm}
          modelLabel={modelLabel}
        />
      )}
    </AlertDialog>
  );
}
function DownloadChoice({
  checking,
  entries,
  onConfirm,
  modelLabel,
}: {
  checking: boolean;
  entries: readonly PlannedDownloadEntry[];
  onConfirm: (includeAssets: boolean) => void;
  modelLabel?: string;
}) {
  const [includeAssets, setIncludeAssets] = useState(true);
  const choiceId = useId();
  const assets = additionalAssetDownloads(entries);
  const checkpoints = entries.filter((e) => e.checkpoint !== false);
  const modelSize = checkpoints.some((e) => e.bytes <= 0)
    ? "Size unknown"
    : formatDownloadBytes(downloadBytes(checkpoints));
  const fullSize = entries.some((e) => e.bytes <= 0)
    ? "Size unknown"
    : formatDownloadBytes(downloadBytes(entries));
  const displayName = modelLabel
    ?.replace(/^[^/]+\//, "")
    .replace(/-GGUF(?=\s|$)/i, "");
  const assetSize = assets.some((e) => e.bytes <= 0)
    ? "Size unknown"
    : formatDownloadBytes(downloadBytes(assets));
  const total = includeAssets ? fullSize : modelSize;
  return (
    <AlertDialogContent className="sm:max-w-[calc(490px*var(--ui-space-scale,1))]">
      <AlertDialogHeader className="flex flex-row flex-wrap items-center justify-between gap-x-4 gap-y-2 text-left sm:group-data-[size=default]/alert-dialog-content:place-items-center">
        <AlertDialogTitle className="shrink-0">Download model</AlertDialogTitle>
        <AlertDialogDescription className="min-w-0 text-xs break-words sm:ml-auto sm:text-right">
          {displayName ?? "Choose which files to download."}
        </AlertDialogDescription>
      </AlertDialogHeader>
      <div
        className="relative min-h-[calc(246px*var(--ui-space-scale,1))] text-sm"
        aria-busy={checking}
      >
        {checking && (
          <div
            role="status"
            className="absolute inset-0 flex items-center justify-center gap-3 text-muted-foreground"
          >
            <span className="size-4 animate-spin rounded-full border-2 border-current border-r-transparent" />
            Checking required files…
          </div>
        )}
        <div
          className="space-y-4"
          style={{ visibility: checking ? "hidden" : "visible" }}
        >
          {(checkpoints.length > 0 || checking) && (
            <div className="flex justify-between gap-4">
              <span>Model files</span>
              <span className="tabular-nums">
                {formatDownloadBytes(downloadBytes(checkpoints))}
              </span>
            </div>
          )}
          <div className="flex items-center gap-3">
            <CheckboxPrimitive.Root
              id={choiceId}
              checked={includeAssets}
              onCheckedChange={(checked) => setIncludeAssets(checked === true)}
              className="relative flex size-4 shrink-0 cursor-pointer items-center justify-center rounded-full border-0 bg-transparent p-0 leading-none outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 16 16"
                width="16"
                height="16"
                className="block size-4 shrink-0 overflow-visible"
              >
                <circle
                  cx="8"
                  cy="8"
                  r="7.5"
                  className={
                    includeAssets
                      ? "fill-emerald-600 stroke-emerald-600"
                      : "fill-transparent stroke-input"
                  }
                />
                {includeAssets && <circle cx="8" cy="8" r="3" fill="white" />}
              </svg>
            </CheckboxPrimitive.Root>
            <label htmlFor={choiceId} className="cursor-pointer">
              Include required files
            </label>
            <span className="ml-auto shrink-0 tabular-nums">{assetSize}</span>
          </div>
          <ul className="mt-3 h-12 space-y-2 overflow-y-auto pl-7 text-xs text-muted-foreground">
            {assets.map((entry, i) => (
              <li
                key={`${entry.repoId}:${i}`}
                className="flex justify-between gap-4"
              >
                <span title={entry.repoId}>{assetLabel(entry)}</span>
                <span className="shrink-0 tabular-nums">
                  {formatDownloadBytes(entry.bytes)}
                </span>
              </li>
            ))}
          </ul>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Required to run · Shared across compatible variants
          </p>
          <div className="flex justify-between border-t pt-4 font-medium">
            <span>Total download</span>
            <span className="tabular-nums">{total}</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Downloading will not load this model or switch your current model.
          </p>
        </div>
      </div>
      <AlertDialogFooter>
        <AlertDialogCancel>Cancel</AlertDialogCancel>
        <AlertDialogAction
          disabled={checking}
          className="w-full sm:w-32"
          onClick={(e) => {
            e.preventDefault();
            onConfirm(includeAssets);
          }}
        >
          Download
        </AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
}
