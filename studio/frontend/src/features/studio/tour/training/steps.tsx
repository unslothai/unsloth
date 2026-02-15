import type { TourStep } from "@/features/tour";

export const studioTrainingTourSteps: TourStep[] = [
  {
    id: "nav",
    target: "navbar",
    title: "Training view",
    body: <>Live run status + metrics. You can stop anytime.</>,
  },
  {
    id: "progress",
    target: "studio-training-progress",
    title: "Progress + ETA",
    body: <>Phase, steps, loss, speed, ETA. This card updates live.</>,
  },
  {
    id: "stop",
    target: "studio-training-stop",
    title: "Stop / save",
    body: <>Stop training, optionally save adapters/checkpoints.</>,
  },
];

