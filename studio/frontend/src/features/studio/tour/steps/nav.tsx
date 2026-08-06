


import { ReadMore, type TourStep } from "@/features/tour";

export const studioNavStep: TourStep = {
  id: "nav",
  target: "navbar",
  title: "Quick orientation",
  body: (
    <>
      Unsloth: pick base model, dataset, hyperparams, then start training. After
      you start, you’ll see a Training view with live loss/metrics. Chat is for
      testing base vs LoRA adapters. Export packages checkpoints for deployment.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-for-beginners" />
    </>
  ),
};
