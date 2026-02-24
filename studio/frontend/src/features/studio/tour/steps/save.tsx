import type { TourStep } from "@/features/tour";

export const studioSaveStep: TourStep = {
  id: "save",
  target: "studio-save",
  title: "Save config",
  body: (
    <>
      Save your training config as a YAML file. Re-running the same baseline
      makes it obvious if a change helped (or if you just got lucky).
    </>
  ),
};
