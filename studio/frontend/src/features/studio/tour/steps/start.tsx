import type { TourStep } from "@/features/tour";

export const studioStartStep: TourStep = {
  id: "start",
  target: "studio-start",
  title: "Start training",
  body: (
    <>
      One click. If it fails, the error text is the first place to look (token,
      path, config).
    </>
  ),
};

