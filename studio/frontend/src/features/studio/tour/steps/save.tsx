import type { TourStep } from "@/features/tour";

export const studioSaveStep: TourStep = {
  id: "save",
  target: "studio-save",
  title: "Save config",
  body: (
    <>
      Save good runs. Repeatability beats vibe. You can iterate from a known
      baseline.
    </>
  ),
};

