// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n";

type Translator = (key: TranslationKey) => string;

export function getStudioTrainingTourSteps(t: Translator): TourStep[] {
  return [
    {
      id: "nav",
      target: "navbar",
      title: t("studio.tour.training.nav.title"),
      body: <>{t("studio.tour.training.nav.body")}</>,
    },
    {
      id: "progress",
      target: "studio-training-progress",
      title: t("studio.tour.training.progress.title"),
      body: <>{t("studio.tour.training.progress.body")}</>,
    },
    {
      id: "train-loss",
      target: "studio-training-loss",
      title: t("studio.tour.training.loss.title"),
      body: <>{t("studio.tour.training.loss.body")}</>,
    },
    {
      id: "eval-loss",
      target: "studio-eval-loss",
      title: t("studio.tour.training.eval.title"),
      body: <>{t("studio.tour.training.eval.body")}</>,
    },
    {
      id: "stop",
      target: "studio-training-stop",
      title: t("studio.tour.training.stop.title"),
      body: <>{t("studio.tour.training.stop.body")}</>,
    },
  ];
}
