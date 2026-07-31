// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ownerOf, useHfTokenStore } from "@/features/hub";
import {
  TRAINING_METHOD_META,
  isLocalTrainingModelSelection,
  useTrainingConfigStore,
  useTrainingReadiness,
  useTrainingResourceNotices,
} from "@/features/training";
import { useGpuInfo } from "@/hooks";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useTrainingResourceDisplayNames } from "../hooks/use-training-resource-display-names";

const LEARNING_RATE_ZERO_RE = /\.?0+e/;
const PREVIEW_COUNT_KEYS = {
  step: {
    zero: "studio.preview.stepZero",
    one: "studio.preview.step",
    two: "studio.preview.stepTwo",
    few: "studio.preview.stepFew",
    many: "studio.preview.stepMany",
    other: "studio.preview.steps",
  },
  epoch: {
    zero: "studio.preview.epochZero",
    one: "studio.preview.epoch",
    two: "studio.preview.epochTwo",
    few: "studio.preview.epochFew",
    many: "studio.preview.epochMany",
    other: "studio.preview.epochs",
  },
} as const satisfies Record<
  "step" | "epoch",
  Record<Intl.LDMLPluralRule, TranslationKey>
>;

function formatLearningRate(lr: number): string {
  if (!Number.isFinite(lr) || lr === 0) {
    return "0";
  }
  return lr.toExponential().replace(LEARNING_RATE_ZERO_RE, "e");
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
      <span className="shrink-0 text-ui-11p5 text-muted-foreground/85">
        {label}
      </span>
      <span
        className={cn(
          "min-w-0 truncate text-ui-12p5 text-foreground/90",
          mono && "font-mono text-ui-12",
        )}
      >
        {value}
      </span>
    </div>
  );
}

function ResourceNoticeRow({
  label,
  status,
  description,
}: {
  label: string;
  status: "download" | "partial";
  description: string;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex items-center justify-between gap-3 text-ui-11p5">
      <span className="flex min-w-0 items-center gap-1.5 text-muted-foreground/80">
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={label}
              className="inline-flex size-3.5 shrink-0 items-center justify-center text-muted-foreground/50 leading-none transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-[260px] leading-relaxed">
            {description}
          </TooltipContent>
        </Tooltip>
        {label}
      </span>
      <span className="truncate text-right text-foreground/80">
        {status === "partial"
          ? t("studio.preview.continuesOnStart")
          : t("studio.preview.downloadsOnStart")}
      </span>
    </div>
  );
}

type ResourceNotice = ReturnType<typeof useTrainingResourceNotices>[number];

function resourceNoticeDescriptionKey(notice: ResourceNotice) {
  if (notice.kind === "model") {
    return notice.status === "partial"
      ? "studio.preview.noticeModelPartial"
      : "studio.preview.noticeModelDownload";
  }
  return notice.status === "partial"
    ? "studio.preview.noticeDatasetPartial"
    : "studio.preview.noticeDatasetDownload";
}

function ResourceNoticeList({
  notices,
}: {
  notices: ResourceNotice[];
}): ReactElement | null {
  const t = useT();
  if (notices.length === 0) {
    return null;
  }
  return (
    <section className="flex flex-col gap-2">
      <p className="text-ui-10p5 font-medium uppercase tracking-[0.05em] text-muted-foreground/60">
        {t("studio.preview.files")}
      </p>
      {notices.map((notice) => (
        <ResourceNoticeRow
          key={`${notice.kind}:${notice.status}:${notice.id}`}
          label={t(
            notice.kind === "model"
              ? "studio.preview.model"
              : "studio.preview.dataset",
          )}
          status={notice.status}
          description={t(resourceNoticeDescriptionKey(notice))}
        />
      ))}
    </section>
  );
}

function BatchPreviewValue({
  batchSize,
  gradientAccumulation,
}: {
  batchSize: number;
  gradientAccumulation: number;
}): ReactElement {
  if (gradientAccumulation <= 1) {
    return <span className="font-mono">{batchSize}</span>;
  }
  return (
    <>
      <span className="font-mono">{batchSize}</span>
      <span className="text-muted-foreground/70"> × </span>
      <span className="font-mono">{gradientAccumulation}</span>
      <span className="text-muted-foreground/70">
        {" "}
        = {batchSize * gradientAccumulation}
      </span>
    </>
  );
}

function PreviewResource({
  variant,
  owner,
  name,
  pendingLabel,
  selected,
  title,
  split,
}: {
  variant: "model" | "dataset";
  owner: string | null;
  name: string | null;
  pendingLabel: string;
  selected: boolean;
  title?: string;
  split?: string | null;
}): ReactElement {
  const isModel = variant === "model";
  return (
    <div className="flex flex-col gap-1">
      {owner && (
        <p className="font-mono text-ui-10p5 uppercase tracking-[0.04em] text-muted-foreground/70">
          {owner}
        </p>
      )}
      <p
        className={cn(
          isModel
            ? "break-words font-heading text-xl font-semibold leading-[1.2] tracking-[-0.018em]"
            : "truncate font-mono text-ui-12",
          selected
            ? isModel
              ? "text-foreground"
              : "text-foreground/80"
            : isModel
              ? "text-muted-foreground/60"
              : "text-muted-foreground/55",
        )}
        title={title}
      >
        {name ?? pendingLabel}
        {split && <span className="text-muted-foreground/70"> · {split}</span>}
      </p>
    </div>
  );
}

function modelPreview(
  model: string | null,
  isLocal: boolean,
  displayName: string | null,
): { name: string | null; owner: string | null } {
  if (!model) {
    return { name: null, owner: null };
  }
  return {
    name: displayName,
    owner: isLocal ? null : ownerOf(model),
  };
}

function datasetPreview({
  source,
  dataset,
  uploadedFile,
  displayName,
  s3Config,
  hasDataset,
}: {
  source: "huggingface" | "upload" | "s3";
  dataset: string | null;
  uploadedFile: string | null;
  displayName: string | null;
  s3Config: { bucket?: string; prefix?: string } | null;
  hasDataset: boolean;
}): { name: string | null; owner: string | null; title?: string } {
  if (source === "s3") {
    const name = s3Config?.bucket
      ? `s3://${s3Config.bucket}${s3Config.prefix ? `/${s3Config.prefix}` : ""}`
      : null;
    return { name, owner: null, title: name ?? undefined };
  }
  if (source === "upload") {
    return {
      name: uploadedFile ? displayName : null,
      owner: null,
      title: uploadedFile ?? undefined,
    };
  }
  return {
    name: displayName,
    owner: hasDataset && dataset ? ownerOf(dataset) : null,
    title: dataset ?? undefined,
  };
}

export function RunPreviewCard({
  startCta,
}: {
  startCta: ReactElement;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const numberFormatter = useMemo(
    () => new Intl.NumberFormat(locale),
    [locale],
  );
  const pluralRules = useMemo(() => new Intl.PluralRules(locale), [locale]);
  const {
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    modelFormat,
    trainingMethod,
    datasetSource,
    dataset,
    uploadedFile,
    s3Config,
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
      modelKnownCached: s.modelKnownCached,
      modelLocalPath: s.modelLocalPath,
      modelFormat: s.modelFormat,
      trainingMethod: s.trainingMethod,
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      uploadedFile: s.uploadedFile,
      s3Config: s.s3Config,
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
  const resourceNotices = useTrainingResourceNotices();
  const resourceDisplayNames = useTrainingResourceDisplayNames({
    selectedModel,
    modelKnownCached,
    modelLocalPath,
    modelFormat,
    datasetSource,
    dataset,
    uploadedFile,
  });

  const modelIsLocal = isLocalTrainingModelSelection({
    model: selectedModel,
    knownCached: modelKnownCached,
    localPath: modelLocalPath,
  });
  const modelDetails = modelPreview(
    selectedModel,
    modelIsLocal,
    resourceDisplayNames.modelName,
  );
  const datasetDetails = datasetPreview({
    source: datasetSource,
    dataset,
    uploadedFile,
    displayName: resourceDisplayNames.datasetName,
    s3Config,
    hasDataset,
  });
  const methodMeta =
    TRAINING_METHOD_META[trainingMethod] ?? TRAINING_METHOD_META.qlora;
  const lengthUnit = maxSteps && maxSteps > 0 ? "step" : "epoch";
  const lengthCount = lengthUnit === "step" ? maxSteps : epochs;
  const lengthLabel = t(
    PREVIEW_COUNT_KEYS[lengthUnit][pluralRules.select(lengthCount)],
    { count: numberFormatter.format(lengthCount) },
  );

  return (
    <aside
      className={cn(
        "elevated-card flex flex-col gap-7 bg-foreground/[0.012] p-6 dark:!border-border/70",
        "dark:bg-white/[0.018]",
      )}
    >
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-ui-11 font-medium tracking-nav text-muted-foreground">
          {t("studio.preview.title")}
        </h2>
        <span
          className={cn(
            "inline-flex h-5 items-center rounded-full px-2 text-ui-10 font-medium tracking-nav",
            isReady
              ? "bg-foreground/[0.06] text-foreground/90 dark:bg-white/[0.08]"
              : "bg-foreground/[0.03] text-muted-foreground/70 dark:bg-white/[0.04]",
          )}
        >
          {isReady ? t("studio.preview.ready") : t("studio.preview.notReady")}
        </span>
      </header>

      <section className="flex flex-col gap-4">
        <PreviewResource
          variant="model"
          owner={modelDetails.owner}
          name={modelDetails.name}
          pendingLabel={t("studio.preview.modelPending")}
          selected={hasModel}
          title={selectedModel ?? undefined}
        />
        <PreviewResource
          variant="dataset"
          owner={datasetDetails.owner}
          name={datasetDetails.name}
          pendingLabel={t("studio.preview.datasetPending")}
          selected={hasDataset}
          title={datasetDetails.title}
          split={
            hasDataset && datasetSource === "huggingface" ? datasetSplit : null
          }
        />
      </section>

      <section className="flex flex-col gap-3">
        <MetaRow
          label={t("studio.preview.method")}
          value={
            <>
              {t(methodMeta.labelKey)}
              <span className="ml-1.5 text-muted-foreground/70">
                · {t(methodMeta.noteKey)}
              </span>
            </>
          }
        />
        <MetaRow label={t("studio.preview.length")} value={lengthLabel} />
        <MetaRow
          label={t("studio.preview.batch")}
          value={
            <BatchPreviewValue
              batchSize={batchSize}
              gradientAccumulation={gradientAccumulation}
            />
          }
        />
        <MetaRow
          label={t("studio.preview.context")}
          value={numberFormatter.format(contextLength)}
          mono={true}
        />
        <MetaRow
          label={t("studio.preview.lr")}
          value={formatLearningRate(learningRate)}
          mono={true}
        />
      </section>

      <section className="flex flex-col gap-3">
        <MetaRow
          label={t("studio.preview.hardware")}
          value={
            gpu.available
              ? `${gpu.name} · ${gpu.memoryTotalGb} GB`
              : t("studio.preview.noGpu")
          }
        />
        <MetaRow
          label={t("studio.preview.hfToken")}
          value={
            hasToken ? t("studio.preview.saved") : t("studio.preview.notSet")
          }
        />
      </section>

      <ResourceNoticeList notices={resourceNotices} />

      <div className="-mx-6 h-px bg-foreground/[0.07] dark:bg-white/[0.06]" />

      <div className="-mt-2">{startCta}</div>
    </aside>
  );
}
