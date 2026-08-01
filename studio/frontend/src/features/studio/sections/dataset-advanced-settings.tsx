// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { type TranslationKey, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import type { DatasetFormat } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import {
  type DatasetStreamingBlocker,
  normalizeSliceInput,
} from "./dataset-panel-helpers";

const DATASET_STREAMING_BLOCKER_KEYS: Record<
  DatasetStreamingBlocker,
  TranslationKey
> = {
  source: "studio.dataset.streaming.blockers.source",
  maxSteps: "studio.dataset.streaming.blockers.maxSteps",
  trainOnCompletions: "studio.dataset.streaming.blockers.trainOnCompletions",
  evalSplit: "studio.dataset.streaming.blockers.evalSplit",
  visionModel: "studio.dataset.streaming.blockers.visionModel",
  audioModel: "studio.dataset.streaming.blockers.audioModel",
  embeddingModel: "studio.dataset.streaming.blockers.embeddingModel",
  imageDataset: "studio.dataset.streaming.blockers.imageDataset",
  audioDataset: "studio.dataset.streaming.blockers.audioDataset",
  appleSilicon: "studio.dataset.streaming.blockers.appleSilicon",
};

export function DatasetAdvancedSettings({
  datasetFormat,
  datasetSliceEnd,
  datasetSliceStart,
  datasetStreaming,
  isStreamingSupported,
  setDatasetFormat,
  setDatasetSliceEnd,
  setDatasetSliceStart,
  setDatasetStreaming,
  streamingBlockers,
}: {
  datasetFormat: DatasetFormat;
  datasetSliceEnd: string | null;
  datasetSliceStart: string | null;
  datasetStreaming: boolean;
  isStreamingSupported: boolean;
  setDatasetFormat: (format: DatasetFormat) => void;
  setDatasetSliceEnd: (value: string | null) => void;
  setDatasetSliceStart: (value: string | null) => void;
  setDatasetStreaming: (value: boolean) => void;
  streamingBlockers: readonly DatasetStreamingBlocker[];
}) {
  const t = useT();
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger
        className={cn(
          "flex w-full cursor-pointer items-center gap-1.5 rounded-sm text-xs text-muted-foreground",
          PICKER_FOCUS_VISIBLE_CLASS,
        )}
      >
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.dataset.advanced")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              {t("studio.dataset.targetFormat")}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    aria-label={t("studio.dataset.targetFormatTooltip")}
                    className={cn(
                      "rounded-sm text-foreground/70 hover:text-foreground",
                      PICKER_FOCUS_VISIBLE_CLASS,
                    )}
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  {t("studio.dataset.targetFormatTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <Select
              value={datasetFormat}
              onValueChange={(value) =>
                setDatasetFormat(value as DatasetFormat)
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{t("studio.dataset.auto")}</SelectItem>
                <SelectItem value="alpaca">Alpaca</SelectItem>
                <SelectItem value="chatml">ChatML</SelectItem>
                <SelectItem value="sharegpt">ShareGPT</SelectItem>
                <SelectItem value="raw">
                  {t("studio.dataset.rawText")}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="datasetStreaming"
              checked={datasetStreaming}
              disabled={!isStreamingSupported}
              onCheckedChange={(value) => setDatasetStreaming(Boolean(value))}
            />
            <label
              htmlFor="datasetStreaming"
              className={`text-xs text-muted-foreground ${
                isStreamingSupported
                  ? "cursor-pointer"
                  : "cursor-not-allowed opacity-60"
              }`}
            >
              {t("studio.dataset.streaming.label")}
            </label>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label={t("studio.dataset.streamingInfoAriaLabel")}
                  className={cn(
                    "rounded-sm text-foreground/70 hover:text-foreground",
                    PICKER_FOCUS_VISIBLE_CLASS,
                  )}
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {isStreamingSupported ? (
                  <span>{t("studio.dataset.streaming.description")}</span>
                ) : (
                  <div className="max-w-xs">
                    <p className="font-medium">
                      {t("studio.dataset.streaming.unavailable")}
                    </p>
                    <ul className="mt-1 list-disc space-y-0.5 pl-4">
                      {streamingBlockers.map((reason) => (
                        <li key={reason}>
                          {t(DATASET_STREAMING_BLOCKER_KEYS[reason])}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                {t("studio.dataset.trainSplitStart")}
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label={t("studio.dataset.trainSplitStartTooltip")}
                      className={cn(
                        "rounded-sm text-foreground/70 hover:text-foreground",
                        PICKER_FOCUS_VISIBLE_CLASS,
                      )}
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {t("studio.dataset.trainSplitStartTooltip")}
                  </TooltipContent>
                </Tooltip>
              </span>
              <Input
                type="number"
                inputMode="numeric"
                min={0}
                step={1}
                placeholder="0"
                value={datasetSliceStart ?? ""}
                onChange={(event) =>
                  setDatasetSliceStart(normalizeSliceInput(event.target.value))
                }
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                {t("studio.dataset.trainSplitEnd")}
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label={t("studio.dataset.trainSplitEndTooltip")}
                      className={cn(
                        "rounded-sm text-foreground/70 hover:text-foreground",
                        PICKER_FOCUS_VISIBLE_CLASS,
                      )}
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {t("studio.dataset.trainSplitEndTooltip")}
                  </TooltipContent>
                </Tooltip>
              </span>
              <Input
                type="number"
                inputMode="numeric"
                min={0}
                step={1}
                placeholder={t("studio.dataset.endPlaceholder")}
                value={datasetSliceEnd ?? ""}
                onChange={(event) =>
                  setDatasetSliceEnd(normalizeSliceInput(event.target.value))
                }
              />
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
