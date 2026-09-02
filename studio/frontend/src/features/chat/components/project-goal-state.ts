// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectRecord } from "../types";

export type VisibleProjectGoalStatus = "active" | "paused" | "completed";

export function projectGoalViewState(project: ProjectRecord): {
  goal: string;
  status: VisibleProjectGoalStatus | null;
  label: string;
  hint: string;
} {
  const goal = project.goal?.trim() ?? "";
  const status = goal ? (project.goalStatus ?? "active") : null;
  return {
    goal,
    status,
    label: goal || "No project goal set",
    hint: goal
      ? "Manage it here or from chat with `/goal help`."
      : "Set a durable objective here or from chat with `/goal set <text>`.",
  };
}
