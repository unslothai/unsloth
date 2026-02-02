import type {
  ModelType,
  StepConfig,
} from "@/types/training";
import type { PipelineType } from "@huggingface/hub";

export const STEPS: StepConfig[] = [
  {
    number: 1,
    title: "Model Type",
    subtitle: "Select type",
    description: "Choose the type of model you want to fine-tune",
  },
  {
    number: 2,
    title: "Model",
    subtitle: "Select model",
    description: "Choose a base model and training method",
  },
  {
    number: 3,
    title: "Dataset",
    subtitle: "Add dataset",
    description: "Select or upload a training dataset",
  },
  {
    number: 4,
    title: "Parameters",
    subtitle: "Configure",
    description: "Fine-tune your training hyperparameters",
  },
  {
    number: 5,
    title: "Summary",
    subtitle: "Review",
    description: "Review your configuration before starting",
  },
];

export const MODEL_TYPES: ReadonlyArray<{
  value: ModelType;
  label: string;
  description: string;
}> = [
  {
    value: "vision",
    label: "Vision",
    description: "Image understanding models",
  },
  {
    value: "tts",
    label: "TTS",
    description: "Text-to-speech models",
  },
  {
    value: "embeddings",
    label: "Embeddings",
    description: "Text embedding models",
  },
  {
    value: "text",
    label: "Text",
    description: "Language models",
  },
];

export const CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768];

export const TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
];

export const DEFAULT_HYPERPARAMS = {
  epochs: 3,
  contextLength: 2048,
  learningRate: 2e-4,
  loraRank: 16,
  loraAlpha: 32,
  loraDropout: 0.05,
  loraVariant: "lora" as const,
  batchSize: 4,
  gradientAccumulation: 8,
  weightDecay: 0.01,
  warmupSteps: 5,
  maxSteps: 0,
  saveSteps: 0,
  packing: false,
  trainOnCompletions: false,
  gradientCheckpointing: "unsloth" as const,
  randomSeed: 3407,
  enableWandb: false,
  wandbToken: "",
  wandbProject: "llm-finetuning",
  enableTensorboard: false,
  tensorboardDir: "runs",
  logFrequency: 10,
  finetuneVisionLayers: true,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMLPModules: true,
  targetModules: TARGET_MODULES,
};

export const MODEL_TYPE_TO_HF_TASK: Record<ModelType, PipelineType> = {
  text: "text-generation",
  vision: "image-text-to-text",
  tts: "text-to-speech",
  embeddings: "feature-extraction",
};
