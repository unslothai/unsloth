// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioBaseModelStep(t: Translator): TourStep {
  return {
    id: "base-model",
    target: "studio-base-model",
    title: t("studio.tour.baseModel.title"),
    body: (
      <>
        {t("studio.tour.baseModel.bodyPrefix")} <span className="font-mono">org/model</span>{" "}
        {t("studio.tour.baseModel.bodySuffix")}{" "}
        <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use" />
      </>
    ),
  };
}
