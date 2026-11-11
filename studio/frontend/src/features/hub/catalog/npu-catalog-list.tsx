// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  type NpuCatalog,
  type NpuModel,
  type NpuPickerSource,
  NpuSetupNotice,
  npuCapabilities,
  npuDownloadLabel,
  npuMemoryWarning,
  npuResumeLabel,
  npuRowsFor,
  npuSizeLabel,
} from "@/features/npu";
import { toast } from "@/lib/toast";
import { useState } from "react";
import { DeleteConfirmDialog } from "./download-card";

export function NpuCatalogList({
  catalog,
  query,
  onDevice,
  selectedId,
  onSelect,
  onRun,
}: {
  catalog: NpuCatalog | null;
  query: string;
  onDevice: boolean;
  selectedId: string | null;
  onSelect: (model: NpuModel) => void;
  onRun: (model: NpuModel) => void;
}) {
  if (!catalog) return null;

  const body = () => {
    if (!catalog.ready) return <NpuSetupNotice catalog={catalog} />;
    if (catalog.listError) {
      return (
        <p className="px-3 text-ui-13 text-destructive">{catalog.listError}</p>
      );
    }
    if (catalog.models === null)
      return <Spinner className="mx-auto mt-6 size-5" />;
    const rows = npuRowsFor(catalog.models, { onDevice, query });
    if (rows.length === 0) {
      return (
        <p className="px-3 pt-6 text-center text-ui-13 text-muted-foreground">
          {onDevice
            ? "No NPU models on this device yet."
            : "No matching NPU models."}
        </p>
      );
    }
    return rows.map((model) => (
      <NpuCatalogRow
        key={model.id}
        model={model}
        catalog={catalog}
        selected={model.id === selectedId}
        onSelect={() => onSelect(model)}
        onRun={() => onRun(model)}
      />
    ));
  };

  return (
    <div
      data-testid="hub-npu-catalog"
      className="mx-auto flex min-h-0 w-full max-w-[var(--hub-measure)] flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-3"
    >
      {body()}
    </div>
  );
}

function NpuCatalogRow({
  model,
  catalog,
  selected,
  onSelect,
  onRun,
}: {
  model: NpuModel;
  catalog: NpuCatalog;
  selected: boolean;
  onSelect: () => void;
  onRun: () => void;
}) {
  const details = [
    model.checkpoint,
    npuSizeLabel(model.size_gb),
    ...npuCapabilities(model).map((capability) => capability.label),
    model.id in catalog.downloads ? null : npuResumeLabel(model),
  ].filter(Boolean);
  const memoryWarning = npuMemoryWarning(model, catalog.status);

  return (
    <div
      data-testid="hub-npu-row"
      data-model={model.id}
      data-selected={selected || undefined}
      className="catalog-row relative flex items-center gap-3 rounded-md px-3 py-2.5"
    >
      <button
        type="button"
        aria-label={`Show ${model.id}`}
        onClick={onSelect}
        className="absolute inset-0 cursor-pointer rounded-md outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
      />
      <span
        aria-hidden={true}
        className="pointer-events-none relative size-1.5 shrink-0 rounded-full bg-format-npu"
      />
      <div className="pointer-events-none relative min-w-0 flex-1">
        <div className="truncate text-ui-13 font-medium text-foreground">
          {model.id}
        </div>
        <div className="truncate text-ui-12 text-muted-foreground">
          {details.join(" · ")}
        </div>
        {memoryWarning ? (
          <div
            data-testid="hub-npu-memory-warning"
            className="text-ui-12 text-amber-600 dark:text-amber-500"
          >
            {memoryWarning}
          </div>
        ) : null}
      </div>
      <div className="relative flex shrink-0 items-center gap-3">
        <NpuModelActions model={model} catalog={catalog} onRun={onRun} />
      </div>
    </div>
  );
}

/** Download, Run and Delete for one NPU model, shared by its row and its detail pane. */
export function NpuModelActions({
  model,
  catalog,
  onRun,
}: {
  model: NpuModel;
  catalog: NpuCatalog;
  onRun: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const remove = async () => {
    setDeleting(true);
    try {
      await catalog.remove(model);
      toast.success(`Deleted ${model.id}`);
      setConfirming(false);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to delete model",
      );
    } finally {
      setDeleting(false);
    }
  };

  if (model.id in catalog.downloads) {
    return (
      <>
        <span className="shrink-0 text-ui-12 tabular-nums text-muted-foreground">
          {npuDownloadLabel(
            catalog.downloads[model.id],
            model.id in catalog.reconnecting,
            catalog.phases[model.id],
          )}
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => void catalog.cancel(model)}
        >
          Cancel
        </Button>
      </>
    );
  }
  if (!model.downloaded) {
    return (
      <Button
        size="sm"
        variant="outline"
        onClick={() => void catalog.download(model)}
      >
        {model.resume_percent == null ? "Download" : "Resume"}
      </Button>
    );
  }
  return (
    <>
      <span className="flex shrink-0 items-center gap-1.5 text-ui-12 text-muted-foreground">
        <span
          aria-hidden={true}
          className="size-1.5 rounded-full bg-status-success"
        />
        On device
      </span>
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          catalog.warnMemory(model);
          onRun();
        }}
      >
        Run
      </Button>
      <Button size="sm" variant="destructive" onClick={() => setConfirming(true)}>
        Delete
      </Button>
      <DeleteConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        title={`Delete ${model.id}?`}
        description="This removes the model's NPU files from this device."
        deleting={deleting}
        onConfirm={() => void remove()}
      />
    </>
  );
}
