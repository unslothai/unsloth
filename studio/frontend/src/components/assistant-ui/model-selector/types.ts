import type { ReactNode } from "react";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported";
  exportType?: "lora" | "merged";
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora" | "exported";
  isLora: boolean;
  ggufVariant?: string;
}

