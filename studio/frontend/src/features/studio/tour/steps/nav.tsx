import type { TourStep } from "@/features/tour";

export const studioNavStep: TourStep = {
  id: "nav",
  target: "navbar",
  title: "Quick orientation",
  body: (
    <>
      Studio is where you fine-tune. Export ships results. Chat is for poking at
      models. This tour is Studio-only (for now).
    </>
  ),
};

