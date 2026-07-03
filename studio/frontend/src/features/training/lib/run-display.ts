// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingRunSummary } from "../types/history";

type TrainingRunTitleFields = Pick<
  TrainingRunSummary,
  "display_name" | "project_name" | "model_name"
>;

function nonEmpty(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

export function getTrainingRunDisplayTitle(run: TrainingRunTitleFields): string {
  return nonEmpty(run.display_name) ?? nonEmpty(run.project_name) ?? run.model_name;
}

export function getTrainingRunModelSubtitle(
  run: TrainingRunTitleFields,
): string | null {
  return getTrainingRunDisplayTitle(run) === run.model_name ? null : run.model_name;
}
