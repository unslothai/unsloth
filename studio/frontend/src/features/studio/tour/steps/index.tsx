


import type { TourStep } from "@/features/tour";
import { studioBaseModelStep } from "./base-model";
import { studioDatasetStep } from "./dataset";
import { studioMethodStep } from "./method";
import { studioNavStep } from "./nav";
import { studioParamsStep } from "./params";
import { studioSaveStep } from "./save";
import { studioStartStep } from "./start";

export const studioTourSteps: TourStep[] = [
  studioNavStep,
  studioBaseModelStep,
  studioMethodStep,
  studioDatasetStep,
  studioParamsStep,
  studioStartStep,
  studioSaveStep,
];
