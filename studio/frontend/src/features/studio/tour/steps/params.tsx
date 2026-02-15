import { ReadMore, type TourStep } from "@/features/tour";

export const studioParamsStep: TourStep = {
  id: "params",
  target: "studio-params",
  title: "Dial hyperparams",
  body: (
    <>
      Epochs + context length + LR. Keep it boring: small changes, one at a
      time. <ReadMore />
    </>
  ),
};
