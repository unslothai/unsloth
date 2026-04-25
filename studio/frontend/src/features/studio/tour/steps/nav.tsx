// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioNavStep(t: Translator): TourStep {
  return {
    id: "nav",
    target: "navbar",
    title: t("studio.tour.nav.title"),
    body: (
      <>
        {t("studio.tour.nav.body")}{" "}
        <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-for-beginners" />
      </>
    ),
  };
}
