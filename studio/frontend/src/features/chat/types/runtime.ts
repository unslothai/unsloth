export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  maxTokens: number;
  systemPrompt: string;
  checkpoint: string;
}

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  minP: 0.01,
  repetitionPenalty: 1.1,
  maxTokens: 512,
  systemPrompt: "",
  checkpoint: "",
};

export interface ChatModelSummary {
  id: string;
  name: string;
  description?: string;
  isVision: boolean;
  isLora: boolean;
}

export interface ChatLoraSummary {
  id: string;
  name: string;
  baseModel: string;
  updatedAt?: number;
}
