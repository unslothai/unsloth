import type { TourStep } from "@/features/tour";

export const exportTourSteps: TourStep[] = [
  {
    id: "checkpoint",
    target: "export-checkpoint",
    title: "Pick checkpoint",
    body: <>Choose which checkpoint (or final model) to export.</>,
  },
  {
    id: "method",
    target: "export-method",
    title: "Export method",
    body: <>Select GGUF vs safetensors, then quant if needed.</>,
  },
  {
    id: "cta",
    target: "export-cta",
    title: "Export",
    body: <>When ready, export to local or HF Hub.</>,
  },
];

