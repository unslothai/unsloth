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
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora";
  isLora: boolean;
  ggufVariant?: string;
}

