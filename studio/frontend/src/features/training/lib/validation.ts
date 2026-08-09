// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateHubResourceId } from "@/components/resource-picker/hub-resource-id";
import type { TranslationKey } from "@/i18n";
import type { TrainingConfigState } from "../types/config";
import { requiresExplicitCachedDatasetSplit } from "./dataset-split-policy";
import {
  validateManualDatasetSplit,
  validateManualDatasetSubset,
} from "./manual-dataset-options";
import { isLocalTrainingModelSelection } from "./model-selection";
import { isUntrainableModelFormat } from "./model-support";
import {
  isTrainingLoraVariantSupportedOnDevice,
  isTrainingMethodSupportedOnDevice,
  isTrainingModelTypeSupportedOnDevice,
} from "./training-methods";

export type StartValidationResult =
  | { ok: true; errorKey: null }
  | { ok: false; errorKey: TranslationKey };

function isTrainingConfigSupportedOnDevice(
  config: TrainingConfigState,
  deviceType?: string,
): boolean {
  if (!isTrainingMethodSupportedOnDevice(config.trainingMethod, deviceType)) {
    return false;
  }
  if (!isTrainingModelTypeSupportedOnDevice(config.modelType, deviceType)) {
    return false;
  }
  if (
    !isTrainingLoraVariantSupportedOnDevice(
      config.loraVariant,
      config.trainingMethod,
      deviceType,
    )
  ) {
    return false;
  }
  if (deviceType !== "mac") {
    return true;
  }
  return !(config.isEmbeddingModel || config.isDatasetAudio === true);
}

export function hasIncompatibleTrainingModalities(
  config: TrainingConfigState,
): boolean {
  return (
    (!config.isVisionModel && config.isDatasetImage === true) ||
    (!config.isAudioModel && config.isDatasetAudio === true)
  );
}

export function validateS3Source(
  config: TrainingConfigState,
): StartValidationResult {
  if (
    config.modelType === "vision" ||
    config.modelType === "audio" ||
    config.isVisionModel ||
    config.isAudioModel
  ) {
    return {
      ok: false,
      errorKey: "studio.training.validation.s3MultimodalUnsupported",
    };
  }
  const s3 = config.s3Config;
  if (!s3 || !s3.bucket.trim()) {
    return {
      ok: false,
      errorKey: "studio.training.validation.s3BucketRequired",
    };
  }
  const hasKeys = Boolean(s3.accessKeyId && s3.secretAccessKey);
  if (!s3.useIamRole && !hasKeys) {
    return {
      ok: false,
      errorKey: "studio.training.validation.s3CredentialsRequired",
    };
  }
  return { ok: true, errorKey: null };
}

function validateDatasetSelection(
  config: TrainingConfigState,
): StartValidationResult {
  if (config.datasetSource === "huggingface") {
    if (!config.dataset) {
      return {
        ok: false,
        errorKey: "studio.training.validation.hfDatasetRequired",
      };
    }
    if (!validateHubResourceId(config.dataset).ok) {
      return {
        ok: false,
        errorKey: "studio.datasetPicker.reasonInvalidHubId",
      };
    }
    if (config.manualDatasetOptionsValid === false) {
      return {
        ok: false,
        errorKey: "studio.dataset.selectors.manualInvalid",
      };
    }
    const requiresSplit = requiresExplicitCachedDatasetSplit(config);
    if (requiresSplit) {
      return {
        ok: false,
        errorKey: "studio.training.validation.hfDatasetSplitRequired",
      };
    }
    if (
      validateManualDatasetSubset(config.datasetSubset ?? "") !== null ||
      validateManualDatasetSplit(
        config.datasetSplit ?? "",
        false,
        !config.datasetStreaming,
      ) !== null ||
      validateManualDatasetSplit(
        config.datasetEvalSplit ?? "",
        false,
        !config.datasetStreaming,
      ) !== null
    ) {
      return {
        ok: false,
        errorKey: "studio.dataset.selectors.manualInvalid",
      };
    }
  } else if (config.datasetSource === "upload") {
    if (!config.uploadedFile) {
      return {
        ok: false,
        errorKey: "studio.training.validation.localDatasetRequired",
      };
    }
  } else if (config.datasetSource === "s3") {
    return validateS3Source(config);
  } else {
    return {
      ok: false,
      errorKey: "studio.training.validation.unsupportedDatasetSource",
    };
  }

  return { ok: true, errorKey: null };
}

export function validateTrainingConfig(
  config: TrainingConfigState,
  deviceType?: string,
): StartValidationResult {
  if (!config.selectedModel) {
    return {
      ok: false,
      errorKey: "studio.training.validation.modelRequired",
    };
  }
  if (
    !isLocalTrainingModelSelection({
      model: config.selectedModel,
      knownCached: config.modelKnownCached,
      localPath: config.modelLocalPath,
    }) &&
    !validateHubResourceId(config.selectedModel).ok
  ) {
    return {
      ok: false,
      errorKey: "studio.modelPicker.reasonInvalidHubId",
    };
  }
  if (isUntrainableModelFormat(config.modelFormat)) {
    return {
      ok: false,
      errorKey:
        config.modelFormat === "gguf"
          ? "studio.modelPicker.reasonGguf"
          : "studio.modelPicker.reasonAdapter",
    };
  }

  if (!isTrainingConfigSupportedOnDevice(config, deviceType)) {
    return {
      ok: false,
      errorKey: "studio.params.notSupportedAppleSilicon",
    };
  }

  if (!Number.isFinite(config.learningRate) || config.learningRate <= 0) {
    return {
      ok: false,
      errorKey: "studio.training.validation.learningRatePositive",
    };
  }
  if (
    config.trainingMethod === "cpt" &&
    config.embeddingLearningRate != null &&
    (!Number.isFinite(config.embeddingLearningRate) ||
      config.embeddingLearningRate <= 0 ||
      config.embeddingLearningRate >= 1)
  ) {
    return {
      ok: false,
      errorKey: "studio.training.validation.embeddingLearningRateRange",
    };
  }

  return validateDatasetSelection(config);
}
