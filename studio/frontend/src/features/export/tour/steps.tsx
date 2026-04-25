// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { TranslationKey } from "@/features/i18n/messages";

export function getExportTourSteps(
  t: (key: TranslationKey) => string,
): TourStep[] {
  return [
  {
    id: "training-run",
    target: "export-training-run",
    title: t("export.tour.trainingRun.title"),
    body: (
      <>
        {t("export.tour.trainingRun.body")}
      </>
    ),
  },
  {
    id: "checkpoint",
    target: "export-checkpoint",
    title: t("export.tour.checkpoint.title"),
    body: (
      <>
        {t("export.tour.checkpoint.body")}
      </>
    ),
  },
  {
    id: "method",
    target: "export-method",
    title: t("export.tour.method.title"),
    body: (
      <>
        {t("export.tour.method.body")}
      </>
    ),
  },
  {
    id: "cta",
    target: "export-cta",
    title: t("export.tour.cta.title"),
    body: (
      <>
        {t("export.tour.cta.body")}
      </>
    ),
  },
  ];
}
