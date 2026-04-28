// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioMethodStep(t: Translator): TourStep {
  return {
    id: "method",
    target: "studio-method",
    title: t("studio.tour.method.title"),
    body: (
      <>
        {t("studio.tour.method.body")}{" "}
        <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
      </>
    ),
  };
}
