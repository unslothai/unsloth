import type { TourStep } from "@/features/tour";
import { ReadMore } from "../read-more";

export const studioBaseModelStep: TourStep = {
  id: "base-model",
  target: "studio-base-model",
  title: "Base model from Hugging Face",
  body: (
    <>
      Search Hub here. Paste <span className="font-mono">org/model</span> too.
      Pick something close to your domain to save compute. <ReadMore />
    </>
  ),
};

