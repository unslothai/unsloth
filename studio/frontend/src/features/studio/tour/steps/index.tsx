// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";
import { studioBaseModelStep } from "./base-model";
import { studioDatasetStep } from "./dataset";
import { studioLocalModelStep } from "./local-model";
import { studioMethodStep } from "./method";
import { studioNavStep } from "./nav";
import { studioParamsStep } from "./params";
import { studioSaveStep } from "./save";
import { studioStartStep } from "./start";

type Translator = (key: TranslationKey) => string;

export function getStudioTourSteps(t: Translator): TourStep[] {
  return [
    studioNavStep(t),
    studioLocalModelStep(t),
    studioBaseModelStep(t),
    studioMethodStep(t),
    studioDatasetStep(t),
    studioParamsStep(t),
    studioStartStep(t),
    studioSaveStep(t),
  ];
}
