


import type { TrainingConfigState } from "../types/config";

export const MODEL_DEFAULT_STATE_KEYS = [
  "epochs",
  "contextLength",
  "learningRate",
  "optimizerType",
  "lrSchedulerType",
  "loraRank",
  "loraAlpha",
  "loraDropout",
  "loraVariant",
  "batchSize",
  "gradientAccumulation",
  "weightDecay",
  "warmupSteps",
  "maxSteps",
  "saveSteps",
  "evalSteps",
  "packing",
  "trainOnCompletions",
  "gradientCheckpointing",
  "randomSeed",
  "visionImageSize",
  "enableWandb",
  "wandbProject",
  "enableTensorboard",
  "tensorboardDir",
  "logFrequency",
  "finetuneVisionLayers",
  "trustRemoteCode",
  "finetuneLanguageLayers",
  "finetuneAttentionModules",
  "finetuneMLPModules",
  "targetModules",
] as const satisfies readonly (keyof TrainingConfigState)[];

export type ModelDefaultsPatch = Partial<
  Pick<TrainingConfigState, (typeof MODEL_DEFAULT_STATE_KEYS)[number]>
>;

export function trainingConfigPatchTouchesModelDefaults(
  patch: Partial<TrainingConfigState>,
): boolean {
  return MODEL_DEFAULT_STATE_KEYS.some((key) => Object.hasOwn(patch, key));
}
