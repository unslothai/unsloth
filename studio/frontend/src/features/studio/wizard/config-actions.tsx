// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  parseYamlConfig,
  serializeConfigToYaml,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import {
  downloadFile,
  isDownloadCancelled,
  pickNativeTrainingConfig,
} from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  Archive04Icon,
  CleanIcon,
  CloudUploadIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ChangeEvent, useRef } from "react";
import {
  TrainingConfigFileError,
  readBrowserTrainingConfig,
  trainingConfigFilename,
} from "./training-config-file";

export function ConfigActions() {
  const t = useT();
  const selectedModel = useTrainingConfigStore((s) => s.selectedModel);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const applyYamlConfig = (content: string, filename: string) => {
    try {
      const config = parseYamlConfig(content);
      useTrainingConfigStore.getState().applyConfigPatch(config);
      toast.success(t("studio.training.configLoaded"), {
        description: filename,
      });
    } catch (error) {
      toast.error(t("studio.training.failedToLoadConfig"), {
        description:
          error instanceof Error
            ? error.message
            : t("studio.training.invalidYamlFile"),
      });
    }
  };

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    readBrowserTrainingConfig(file)
      .then((content) => applyYamlConfig(content, file.name))
      .catch((error: unknown) => {
        toast.error(t("studio.training.failedToReadFile"), {
          description:
            error instanceof TrainingConfigFileError
              ? t(error.translationKey)
              : error instanceof Error
                ? error.message
                : undefined,
        });
      });
  };

  const handleLoadConfig = async () => {
    if (!isTauri) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const selected = await pickNativeTrainingConfig();
      if (selected) {
        applyYamlConfig(selected.content, selected.name);
      }
    } catch (error) {
      toast.error(t("studio.training.failedToReadFile"), {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const reportSaveError = (error: unknown) => {
    if (!isDownloadCancelled(error)) {
      toast.error(t("studio.training.failedToSaveConfig"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const handleSaveConfig = () => {
    try {
      const state = useTrainingConfigStore.getState();
      const includeVisionFields =
        state.isVisionModel && state.isDatasetImage !== false;
      const selectedModelLower = (state.selectedModel ?? "").toLowerCase();
      const isDeepseekOcr =
        selectedModelLower.includes("deepseek") &&
        selectedModelLower.includes("ocr");
      const yaml = serializeConfigToYaml(
        state,
        includeVisionFields,
        includeVisionFields && !isDeepseekOcr,
      );
      const filename = trainingConfigFilename({
        model: state.selectedModel,
        method: state.trainingMethod,
        dataset: state.dataset ?? state.uploadedFile,
      });
      downloadFile(yaml, filename, "text/yaml;charset=utf-8").catch(
        reportSaveError,
      );
    } catch (error) {
      reportSaveError(error);
    }
  };

  const handleResetConfig = () => {
    useTrainingConfigStore.getState().resetToModelDefaults();
    toast.success(t("studio.training.parametersReset"));
  };

  return (
    <div className="flex flex-wrap gap-2">
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={() => void handleLoadConfig()}
          >
            <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
            {t("studio.wizard.loadYaml")}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {t("studio.training.uploadConfigTooltip")}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            data-tour="studio-save"
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={handleSaveConfig}
          >
            <HugeiconsIcon icon={Archive04Icon} className="size-3.5" />
            {t("studio.wizard.saveYaml")}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {t("studio.training.saveConfigTooltip")}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className="h-9 cursor-pointer rounded-lg"
            onClick={handleResetConfig}
            disabled={!selectedModel}
          >
            <HugeiconsIcon icon={CleanIcon} className="size-3.5" />
            {t("studio.wizard.resetDefaults")}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {t("studio.training.resetConfigTooltip")}
        </TooltipContent>
      </Tooltip>
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml"
        className="hidden"
        onChange={handleFileUpload}
      />
    </div>
  );
}
