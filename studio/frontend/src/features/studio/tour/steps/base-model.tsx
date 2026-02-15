import { ReadMore, type TourStep } from "@/features/tour";

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
