import type { TourStep } from "@/features/tour";
import { ReadMore } from "../read-more";

export const studioLocalModelStep: TourStep = {
  id: "local-model",
  target: "studio-local-model",
  title: "Local model path",
  body: (
    <>
      Point to a local folder (<span className="font-mono">./models/...</span>)
      or a custom HF repo. Use this when you already downloaded weights.{" "}
      <ReadMore />
    </>
  ),
};

