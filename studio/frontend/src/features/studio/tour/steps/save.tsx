// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function studioSaveStep(t: Translator): TourStep {
  return {
    id: "save",
    target: "studio-save",
    title: t("studio.tour.save.title"),
    body: <>{t("studio.tour.save.body")}</>,
  };
}
