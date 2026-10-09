// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  type NpuModel,
  type NpuPickerSource,
  NpuPoweredBy,
  NpuSetupNotice,
  npuDownloadLabel,
  npuResumeLabel,
  npuRowsFor,
  npuSizeLabel,
  useNpuCatalog,
} from "@/features/npu";
import { toast } from "@/lib/toast";
import { useState } from "react";
import { DeleteConfirmDialog } from "./download-card";

/** The Hub's NPU format: Lemonade's FastFlowLM catalog, the same models the chat picker lists. */
export function NpuCatalogList({
  source,
  query,
  onDevice,
  onRun,
}: {
  source: NpuPickerSource;
  query: string;
  onDevice: boolean;
  onRun: (model: NpuModel) => void;
}) {
  const catalog = useNpuCatalog(source);
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
        progress={catalog.downloads[model.id]}
        downloading={model.id in catalog.downloads}
        reconnecting={model.id in catalog.reconnecting}
        onDownload={() => void catalog.download(model)}
        onRun={() => onRun(model)}
        onRemove={() => catalog.remove(model)}
      />
    ));
  };

  return (
    <div
      data-testid="hub-npu-catalog"
      className="mx-auto flex min-h-0 w-full max-w-[var(--hub-measure)] flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-3"
    >
      {body()}
      {catalog.ready ? (
        <NpuPoweredBy status={catalog.status} className="px-3 pt-3" />
      ) : null}
    </div>
  );
}

function NpuCatalogRow({
  model,
  progress,
  downloading,
  reconnecting,
  onDownload,
  onRun,
  onRemove,
}: {
  model: NpuModel;
  progress: number | null | undefined;
  downloading: boolean;
  reconnecting: boolean;
  onDownload: () => void;
  onRun: () => void;
  onRemove: () => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const details = [
    model.checkpoint,
    npuSizeLabel(model.size_gb),
    model.supports_vision ? "Vision" : null,
    model.supports_reasoning ? "Reasoning" : null,
    model.supports_tools ? "Tools" : null,
    downloading ? null : npuResumeLabel(model),
  ].filter(Boolean);

  const remove = async () => {
    setDeleting(true);
    try {
      await onRemove();
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

  return (
    <div
      data-testid="hub-npu-row"
      data-model={model.id}
      className="flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors hover:bg-accent/50"
    >
      <span
        aria-hidden={true}
        className="size-1.5 shrink-0 rounded-full bg-format-npu"
      />
      <div className="min-w-0 flex-1">
        <div className="truncate text-ui-13 font-medium text-foreground">
          {model.id}
        </div>
        <div className="truncate text-ui-12 text-muted-foreground">
          {details.join(" · ")}
        </div>
      </div>
      {downloading ? (
        <span className="shrink-0 text-ui-12 tabular-nums text-muted-foreground">
          {npuDownloadLabel(progress, reconnecting)}
        </span>
      ) : model.downloaded ? (
        <>
          <span className="flex shrink-0 items-center gap-1.5 text-ui-12 text-muted-foreground">
            <span
              aria-hidden={true}
              className="size-1.5 rounded-full bg-status-success"
            />
            On device
          </span>
          <Button size="sm" variant="outline" onClick={onRun}>
            Run
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setConfirming(true)}>
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
      ) : (
        <Button size="sm" variant="outline" onClick={onDownload}>
          {model.resume_percent == null ? "Download" : "Resume"}
        </Button>
      )}
    </div>
  );
}
