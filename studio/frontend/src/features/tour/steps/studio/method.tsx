import type { TourStep } from "@/features/tour";
import { ReadMore } from "../read-more";

export const studioMethodStep: TourStep = {
  id: "method",
  target: "studio-method",
  title: "Method: QLoRA vs LoRA vs Full",
  body: (
    <>
      QLoRA: lowest VRAM (4-bit). LoRA: fast + solid (16-bit adapters). Full:
      slowest, highest cost, updates all weights. <ReadMore />
    </>
  ),
};

