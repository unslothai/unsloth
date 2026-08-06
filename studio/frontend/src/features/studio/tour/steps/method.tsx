


import { ReadMore, type TourStep } from "@/features/tour";

export const studioMethodStep: TourStep = {
  id: "method",
  target: "studio-method",
  title: "Method: QLoRA vs LoRA vs Full",
  body: (
    <>
      LoRA: trains small adapter weights (fast, common default). QLoRA: LoRA on
      4-bit base weights (much lower VRAM). Full: updates all weights (highest
      cost, usually needs more data to be worth it).{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
