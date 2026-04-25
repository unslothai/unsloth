// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioStartStep(t: Translator): TourStep {
  return {
    id: "start",
    target: "studio-start",
    title: t("studio.tour.start.title"),
    body: <>{t("studio.tour.start.body")}</>,
  };
}
