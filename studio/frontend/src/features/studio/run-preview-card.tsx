// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { datasetShortName, modelShortName, ownerOf } from "@/lib/format";
import {
  useTrainingConfigStore,
  useTrainingReadiness,
} from "@/features/training";
import {
  TRAINING_METHOD_LABELS,
  TRAINING_METHOD_NOTES,
} from "@/features/training/lib/training-method-meta";
import { useGpuInfo } from "@/hooks";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import type { ReactElement, ReactNode } from "react";
import { useShallow } from "zustand/react/shallow";

function formatLearningRate(lr: number): string {
  if (!Number.isFinite(lr) || lr === 0) return "0";
  return lr.toExponential().replace(/\.?0+e/, "e");
}

function MetaRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
}): ReactElement {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="shrink-0 text-[11.5px] text-muted-foreground/85">
        {label}
      </span>
      <span
        className={cn(
          "min-w-0 truncate text-[12.5px] text-foreground/90",
          mono && "font-mono text-[12px]",
        )}
      >
        {value}
      </span>
    </div>
  );
}

export function RunPreviewCard({
  startCta,
}: {
  startCta: ReactElement;
}): ReactElement {
  const {
    selectedModel,
    trainingMethod,
    datasetSource,
    dataset,
    uploadedFile,
    datasetSplit,
    maxSteps,
    epochs,
    batchSize,
    gradientAccumulation,
    learningRate,
    contextLength,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      selectedModel: s.selectedModel,
      trainingMethod: s.trainingMethod,
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      uploadedFile: s.uploadedFile,
      datasetSplit: s.datasetSplit,
      maxSteps: s.maxSteps,
      epochs: s.epochs,
      batchSize: s.batchSize,
      gradientAccumulation: s.gradientAccumulation,
      learningRate: s.learningRate,
      contextLength: s.contextLength,
    })),
  );

  const gpu = useGpuInfo();
  const hfToken = useHfTokenStore((s) => s.token);
  const hasToken = !!hfToken && hfToken.trim().length > 0;
  const { isReady, hasModel, hasDataset } = useTrainingReadiness();

  const modelOwner = selectedModel ? ownerOf(selectedModel) : null;
  const modelName = selectedModel ? modelShortName(selectedModel) : null;
  const datasetName =
    datasetSource === "upload"
      ? uploadedFile
        ? datasetShortName(uploadedFile)
        : null
      : dataset
        ? datasetShortName(dataset)
        : null;
  const datasetOwner =
    hasDataset && datasetSource !== "upload" && dataset
      ? ownerOf(dataset)
      : null;
  const methodLabel = TRAINING_METHOD_LABELS[trainingMethod];
  const methodNote = TRAINING_METHOD_NOTES[trainingMethod];
  const lengthLabel =
    maxSteps && maxSteps > 0
      ? `${maxSteps.toLocaleString()} steps`
      : `${epochs} epoch${epochs === 1 ? "" : "s"}`;
  const effectiveBatch = batchSize * gradientAccumulation;
  const showEffectiveBatch = gradientAccumulation > 1;

  return (
    <aside
      className={cn(
        "elevated-card flex flex-col gap-7 bg-foreground/[0.012] p-6",
        "dark:bg-white/[0.018]",
      )}
    >
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-[11px] font-medium tracking-nav text-muted-foreground">
          Run preview
        </h2>
        <span
          className={cn(
            "inline-flex h-5 items-center rounded-full px-2 text-[10px] font-medium tracking-nav",
            isReady
              ? "bg-foreground/[0.06] text-foreground/90 dark:bg-white/[0.08]"
              : "bg-foreground/[0.03] text-muted-foreground/70 dark:bg-white/[0.04]",
          )}
        >
          {isReady ? "Ready" : "Not ready"}
        </span>
      </header>

      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          {modelOwner && (
            <p className="font-mono text-[10.5px] uppercase tracking-[0.04em] text-muted-foreground/70">
              {modelOwner}
            </p>
          )}
          <p
            className={cn(
              "break-words font-heading text-[20px] font-semibold leading-[1.2] tracking-[-0.018em]",
              hasModel ? "text-foreground" : "text-muted-foreground/60",
            )}
            title={selectedModel ?? undefined}
          >
            {modelName ?? "Select a model"}
          </p>
        </div>
        <div className="flex flex-col gap-1">
          {datasetOwner && (
            <p className="font-mono text-[10.5px] uppercase tracking-[0.04em] text-muted-foreground/70">
              {datasetOwner}
            </p>
          )}
          <p
            className={cn(
              "truncate font-mono text-[12px]",
              hasDataset ? "text-foreground/80" : "text-muted-foreground/55",
            )}
            title={
              datasetSource === "upload"
                ? (uploadedFile ?? undefined)
                : (dataset ?? undefined)
            }
          >
            {datasetName ?? "No dataset"}
            {hasDataset && datasetSource !== "upload" && datasetSplit ? (
              <span className="text-muted-foreground/70"> · {datasetSplit}</span>
            ) : null}
          </p>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <MetaRow
          label="Method"
          value={
            <>
              {methodLabel}
              <span className="ml-1.5 text-muted-foreground/70">
                · {methodNote}
              </span>
            </>
          }
        />
        <MetaRow label="Length" value={lengthLabel} />
        <MetaRow
          label="Batch"
          value={
            showEffectiveBatch ? (
              <>
                <span className="font-mono">{batchSize}</span>
                <span className="text-muted-foreground/70"> × </span>
                <span className="font-mono">{gradientAccumulation}</span>
                <span className="text-muted-foreground/70">
                  {" "}
                  = {effectiveBatch}
                </span>
              </>
            ) : (
              <span className="font-mono">{batchSize}</span>
            )
          }
        />
        <MetaRow
          label="Context"
          value={contextLength.toLocaleString()}
          mono
        />
        <MetaRow label="LR" value={formatLearningRate(learningRate)} mono />
      </section>

      <section className="flex flex-col gap-3">
        <MetaRow
          label="Hardware"
          value={
            gpu.available
              ? `${gpu.name} · ${gpu.memoryTotalGb} GB`
              : "No GPU detected"
          }
        />
        <MetaRow
          label="HF token"
          value={hasToken ? "Connected" : "Not set"}
        />
      </section>

      <div className="-mx-6 h-px bg-foreground/[0.07] dark:bg-white/[0.06]" />

      <div className="-mt-2">{startCta}</div>
    </aside>
  );
}
