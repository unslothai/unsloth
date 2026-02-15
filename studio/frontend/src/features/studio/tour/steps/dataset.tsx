import { ReadMore, type TourStep } from "@/features/tour";

export const studioDatasetStep: TourStep = {
  id: "dataset",
  target: "studio-dataset",
  title: "Dataset",
  body: (
    <>
      Search Hub or paste <span className="font-mono">user/dataset</span>.
      Preview a few rows before you burn hours of compute. <ReadMore />
    </>
  ),
};
