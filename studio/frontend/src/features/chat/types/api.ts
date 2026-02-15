export interface BackendModelDetails {
  id: string;
  name?: string | null;
  is_vision?: boolean;
  is_lora?: boolean;
}

export interface ListModelsResponse {
  models: BackendModelDetails[];
  default_models: string[];
}

export interface BackendLoraInfo {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
}

export interface ListLorasResponse {
  loras: BackendLoraInfo[];
  outputs_dir: string;
}

export interface LoadModelRequest {
  model_path: string;
  hf_token: string | null;
  max_seq_length: number;
  load_in_4bit: boolean;
  is_lora: boolean;
}

export interface LoadModelResponse {
  status: string;
  model: string;
  display_name: string;
  is_vision: boolean;
  is_lora: boolean;
}

export interface UnloadModelRequest {
  model_path: string;
}

export interface InferenceStatusResponse {
  active_model: string | null;
  is_vision: boolean;
  loading: string[];
  loaded: string[];
}

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OpenAIChatCompletionsRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  top_k: number;
  repetition_penalty: number;
  image_base64?: string;
  use_adapter?: boolean | string | null;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string;
}

export interface OpenAIChatChunkChoice {
  delta?: OpenAIChatDelta;
  finish_reason?: string | null;
}

export interface OpenAIChatChunk {
  choices?: OpenAIChatChunkChoice[];
}
