// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioLocalModelStep(t: Translator): TourStep {
  return {
    id: "local-model",
    target: "studio-local-model",
    title: t("studio.tour.localModel.title"),
    body: (
      <>
        {t("studio.tour.localModel.bodyPrefix")}{" "}
        <span className="font-mono">./models/...</span> {t("studio.tour.localModel.bodySuffix")}{" "}
        <ReadMore href="https://unsloth.ai/docs/basics/fine-tuning-llms-guide" />
      </>
    ),
  };
}
