import { ReadMore, type TourStep } from "@/features/tour";

export const studioDatasetStep: TourStep = {
  id: "dataset",
  target: "studio-dataset",
  title: "Dataset",
  body: (
    <>
      Search Hub or paste <span className="font-mono">user/dataset</span>. Preview
      a few rows: formatting matters more than size. If outputs look off in
      Chat later, 80% chance it’s dataset formatting/template.{" "}
      <ReadMore href="https://docs.unsloth.ai/basics/fine-tuning-llms-guide" />
    </>
  ),
};
