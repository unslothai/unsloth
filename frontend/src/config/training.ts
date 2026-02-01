import type {
  DatasetOption,
  ModelOption,
  ModelType,
  StepConfig,
} from "@/types/training";

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

export const MODELS: ModelOption[] = [
  // Vision models
  {
    id: "llava-1.6-7b",
    name: "LLaVA 1.6 7B",
    type: "vision",
    params: "7B",
    vram: "~6GB",
    context: "4K",
    recommended: true,
  },
  {
    id: "llava-1.6-13b",
    name: "LLaVA 1.6 13B",
    type: "vision",
    params: "13B",
    vram: "~10GB",
    context: "4K",
    recommended: true,
  },
  {
    id: "qwen-vl-7b",
    name: "Qwen-VL 7B",
    type: "vision",
    params: "7B",
    vram: "~6GB",
    context: "8K",
  },
  // TTS models
  { id: "bark", name: "Bark", type: "tts", params: "1B", recommended: true },
  { id: "xtts-v2", name: "XTTS v2", type: "tts", params: "500M" },
  // Embedding models
  {
    id: "bge-large",
    name: "BGE Large",
    type: "embeddings",
    params: "335M",
    recommended: true,
  },
  { id: "e5-large", name: "E5 Large", type: "embeddings", params: "335M" },
  { id: "gte-large", name: "GTE Large", type: "embeddings", params: "335M" },
  // Text models
  {
    id: "llama-3.1-8b",
    name: "Llama 3.1 8B",
    type: "text",
    params: "8B",
    vram: "~6GB",
    context: "128K",
    recommended: true,
  },
  {
    id: "llama-3.1-70b",
    name: "Llama 3.1 70B",
    type: "text",
    params: "70B",
    vram: "~40GB",
    context: "128K",
  },
  {
    id: "mistral-7b",
    name: "Mistral 7B",
    type: "text",
    params: "7B",
    vram: "~5GB",
    context: "32K",
  },
  {
    id: "qwen2-7b",
    name: "Qwen2 7B",
    type: "text",
    params: "7B",
    vram: "~5GB",
    context: "32K",
  },
  {
    id: "phi-3-mini",
    name: "Phi-3 Mini",
    type: "text",
    params: "3.8B",
    vram: "~3GB",
    context: "128K",
  },
  {
    id: "gemma-2-9b",
    name: "Gemma 2 9B",
    type: "text",
    params: "9B",
    vram: "~7GB",
    context: "8K",
  },
  {
    id: "gemma-3-27b",
    name: "Gemma 3 27B",
    type: "text",
    params: "27B",
    vram: "~18GB",
    context: "128K",
    recommended: true,
  },
];

export const DATASETS: DatasetOption[] = [
  {
    id: "alpaca",
    name: "Alpaca",
    description: "Instruction following dataset",
    size: "52K",
    recommended: true,
  },
  {
    id: "dolly-15k",
    name: "Dolly 15K",
    description: "Databricks instruction dataset",
    size: "15K",
    recommended: true,
  },
  {
    id: "openorca",
    name: "OpenOrca",
    description: "Large-scale instruct dataset",
    size: "4M",
  },
  {
    id: "wizard-lm",
    name: "WizardLM",
    description: "Complex instruction dataset",
    size: "196K",
  },
  {
    id: "lima",
    name: "LIMA",
    description: "Curated high-quality dataset",
    size: "1K",
  },
  {
    id: "sharegpt",
    name: "ShareGPT",
    description: "Conversation dataset",
    size: "90K",
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
