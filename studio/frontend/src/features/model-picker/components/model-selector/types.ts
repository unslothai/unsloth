


import type { ReactNode } from "react";
import type { PerModelConfig } from "../../model-config/per-model-config";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
  isGguf?: boolean;
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported" | "local";
  exportType?: "lora" | "merged" | "gguf";
}

export interface ExternalModelOption extends ModelOption {
  providerId: string;
  providerName: string;
  /** Registry key (e.g. openai, gemini) for provider branding. */
  providerType: string;
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora" | "exported" | "local" | "external";
  isLora: boolean;
  ggufVariant?: string;
  /** Exact GGUF filename for the picked quant (filenames do not always follow the repo name, e.g. FLUX.1-schnell -> flux1-schnell-*.gguf). */
  ggufFilename?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  /** Native GGUF context, threaded so a staged pick can seed the slider. */
  contextLength?: number | null;
  /** Direct local .gguf file picked without a variant (custom folder / LM
   *  Studio). Marks it as a GGUF source for the deferred-load staging flow. */
  isGguf?: boolean;
  /** Staged metadata confirmed the separate DiffusionGemma runner. */
  isDiffusion?: boolean;
  config?: PerModelConfig;
  forceReload?: boolean;
  /** model_path to send when the pick loads from elsewhere, e.g. a pinned snapshot dir. */
  loadId?: string | null;
  /** Native path token so an active-model reload can reopen a file-picked GGUF. */
  nativePathToken?: string;
  nativePathExpiresAtMs?: number | null;
}

export interface ModelPickTarget {
  id: string;
  displayName: string;
  ggufVariant?: string | null;
  isGguf: boolean;
  /**
   * Whether an OpenAI-compatible request can actually load this model. Not the same as isGguf:
   * local_model_resolver skips Ollama's scanner. Defaults to isGguf when unknown.
   */
  apiLoadable?: boolean;
  /**
   * Identity the saved settings are keyed by, when that is not what loads: a repo cached
   * outside the active HF cache loads by snapshot path while its settings key on the repo
   * id. Probes that must open the model keep using `id`.
   */
  configId?: string;
  meta: ModelSelectorChangeMeta;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
}
