import type { TrainingConfigState } from "../types/config";
import type { TrainingStartRequest } from "../types/api";

const BACKEND_LORA_TYPE = "LoRA/QLoRA";
const BACKEND_FULL_TYPE = "Full Finetuning";

export function toBackendTrainingType(trainingMethod: string): string {
  return trainingMethod === "full" ? BACKEND_FULL_TYPE : BACKEND_LORA_TYPE;
}

export function buildTrainingStartPayload(
  config: TrainingConfigState,
): TrainingStartRequest {
  const adapterMethod = config.trainingMethod !== "full";
  const isQlorMethod = config.trainingMethod === "qlora";
  const hfDataset = config.datasetSource === "huggingface" ? config.dataset : null;
  const customFormatMapping = buildCustomFormatMapping(config);

  return {
    model_name: config.selectedModel ?? "",
    training_type: toBackendTrainingType(config.trainingMethod),
    hf_token: config.hfToken.trim() || null,
    load_in_4bit: adapterMethod ? isQlorMethod : false,
    max_seq_length: config.contextLength,
    hf_dataset: hfDataset,
    hf_dataset_config: hfDataset ? config.datasetSubset : null,
    hf_dataset_split: hfDataset ? config.datasetSplit : null,
    local_datasets: [],
    format_type: config.datasetFormat,
    custom_format_mapping: customFormatMapping,
    num_epochs: config.epochs,
    learning_rate: String(config.learningRate),
    batch_size: config.batchSize,
    gradient_accumulation_steps: config.gradientAccumulation,
    warmup_steps: config.warmupSteps,
    warmup_ratio: null,
    max_steps: config.maxSteps,
    save_steps: config.saveSteps,
    eval_steps: config.evalSteps,
    weight_decay: config.weightDecay,
    random_seed: config.randomSeed,
    packing: config.packing,
    optim: "adamw_8bit",
    lr_scheduler_type: "linear",
    use_lora: adapterMethod,
    lora_r: config.loraRank,
    lora_alpha: config.loraAlpha,
    lora_dropout: config.loraDropout,
    target_modules: adapterMethod
      ? config.targetModules.filter((m) => m !== "all-linear")
      : [],
    gradient_checkpointing: config.gradientCheckpointing,
    use_rslora: config.loraVariant === "rslora",
    use_loftq: config.loraVariant === "loftq",
    train_on_completions: config.trainOnCompletions,
    finetune_vision_layers: config.finetuneVisionLayers,
    finetune_language_layers: config.finetuneLanguageLayers,
    finetune_attention_modules: config.finetuneAttentionModules,
    finetune_mlp_modules: config.finetuneMLPModules,
    is_dataset_multimodal: !!config.isDatasetMultimodal,
    enable_wandb: config.enableWandb,
    wandb_token: config.enableWandb ? config.wandbToken.trim() || null : null,
    wandb_project: config.enableWandb
      ? config.wandbProject.trim() || null
      : null,
    enable_tensorboard: config.enableTensorboard,
    tensorboard_dir: config.enableTensorboard
      ? config.tensorboardDir.trim() || null
      : null,
  };
}

function buildCustomFormatMapping(
  config: TrainingConfigState,
): Record<string, string> | undefined {
  const { input, output } = config.datasetManualMapping;
  if (!input || !output) return undefined;

  if (config.isVisionModel && config.isDatasetMultimodal) {
    return { [input]: "image", [output]: "text" };
  }

  return { [input]: "user", [output]: "assistant" };
}
