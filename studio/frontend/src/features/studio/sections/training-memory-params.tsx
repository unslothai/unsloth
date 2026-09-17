// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TabsContent } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  isRawTextDatasetFormat,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import type { GradientCheckpointing } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "./params-section-controls";

const VISION_IMAGE_SIZE_PRESETS = [256, 384, 512, 768, 1024, 1536, 2048];

function PackingOption({
  isMac,
  checked,
  onChange,
}: {
  isMac: boolean;
  checked: boolean;
  onChange: (value: boolean) => void;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex items-center gap-2">
      <Checkbox
        id="packing"
        checked={checked}
        disabled={isMac}
        onCheckedChange={(value) => onChange(!!value)}
      />
      <label
        htmlFor="packing"
        className={`text-xs text-muted-foreground ${
          isMac ? "cursor-not-allowed opacity-60" : "cursor-pointer"
        }`}
      >
        {t("studio.params.enablePacking")}
      </label>
      {isMac && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent>
            Packing is not supported on Apple Silicon (MLX).
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

function TrainOnCompletionsOption({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean;
  disabled: boolean;
  onChange: (value: boolean) => void;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex items-center gap-2">
      <Checkbox
        id="trainOnCompletions"
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => onChange(!!value)}
      />
      <label
        htmlFor="trainOnCompletions"
        aria-disabled={disabled || undefined}
        title={
          disabled
            ? t("studio.dataset.streaming.completionsUnavailable")
            : undefined
        }
        className={`text-xs text-muted-foreground ${
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer"
        }`}
      >
        {t("studio.params.assistantCompletionsOnly")}
      </label>
    </div>
  );
}

export function TrainingMemoryParams(): ReactElement {
  const t = useT();
  const isMac = usePlatformStore((state) => state.deviceType === "mac");
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      trainingMethod: state.trainingMethod,
      datasetFormat: state.datasetFormat,
      datasetStreaming: state.datasetStreaming,
      packing: state.packing,
      trainOnCompletions: state.trainOnCompletions,
      gradientCheckpointing: state.gradientCheckpointing,
      isVisionModel: state.isVisionModel,
      isEmbeddingModel: state.isEmbeddingModel,
      isDatasetImage: state.isDatasetImage,
      visionImageSize: state.visionImageSize,
      setVisionImageSize: state.setVisionImageSize,
      setPacking: state.setPacking,
      setTrainOnCompletions: state.setTrainOnCompletions,
      setGradientCheckpointing: state.setGradientCheckpointing,
    })),
  );
  const showVisionLora = store.isVisionModel && store.isDatasetImage === true;
  const selectedModelLower = (store.selectedModel ?? "").toLowerCase();
  const showVisionImageSize =
    showVisionLora &&
    !(
      selectedModelLower.includes("deepseek") &&
      selectedModelLower.includes("ocr")
    );
  const showPacking = !(showVisionLora || store.isEmbeddingModel);
  const showTrainOnCompletions = !(
    store.isEmbeddingModel ||
    store.trainingMethod === "cpt" ||
    isRawTextDatasetFormat(store.datasetFormat)
  );

  return (
    <TabsContent value="memory" className="mt-3 flex flex-col gap-3">
      {showVisionImageSize && (
        <ParamsRow
          label="Image Size"
          tooltip={
            <>
              Resize images by maximum side length. Default uses the model image
              size. Larger images use up more context. Does not upscale or
              change aspect ratio.{" "}
              <a
                href="https://unsloth.ai/docs/basics/vision-fine-tuning"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline"
              >
                Read more
              </a>
            </>
          }
        >
          <Select
            value={
              store.visionImageSize == null
                ? "default"
                : String(store.visionImageSize)
            }
            onValueChange={(value) =>
              store.setVisionImageSize(
                value === "default" ? null : Number(value),
              )
            }
          >
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Default</SelectItem>
              {store.visionImageSize != null &&
                !VISION_IMAGE_SIZE_PRESETS.includes(store.visionImageSize) && (
                  <SelectItem value={String(store.visionImageSize)}>
                    {store.visionImageSize}
                  </SelectItem>
                )}
              {VISION_IMAGE_SIZE_PRESETS.map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ParamsRow>
      )}
      <ParamsRow
        label={t("studio.params.gradCheckpoint")}
        tooltip={
          <>
            {t("studio.params.gradCheckpointTooltip")}{" "}
            <a
              href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline"
            >
              {t("studio.params.readMore")}
            </a>
          </>
        }
      >
        <Select
          value={store.gradientCheckpointing}
          onValueChange={(value) =>
            store.setGradientCheckpointing(value as GradientCheckpointing)
          }
        >
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">{t("studio.params.none")}</SelectItem>
            <SelectItem value="true">{t("studio.params.standard")}</SelectItem>
            {isMac ? (
              <SelectItem value="mlx">MLX</SelectItem>
            ) : (
              <SelectItem value="unsloth">Unsloth</SelectItem>
            )}
          </SelectContent>
        </Select>
      </ParamsRow>
      {showPacking && (
        <PackingOption
          isMac={isMac}
          checked={store.packing}
          onChange={store.setPacking}
        />
      )}
      {showTrainOnCompletions && (
        <TrainOnCompletionsOption
          checked={store.trainOnCompletions}
          disabled={store.datasetStreaming}
          onChange={store.setTrainOnCompletions}
        />
      )}
    </TabsContent>
  );
}
