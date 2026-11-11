// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkTaskInfo } from "../api/benchmark-api";

// Recipe tasks need a held-out dataset and scoring spec that only the Flows
// Benchmark node supplies; the Evals page would always get a 400 for them.
export function evalsPickerTasks(
  tasks: BenchmarkTaskInfo[],
): BenchmarkTaskInfo[] {
  return tasks.filter((t) => t.runner !== "recipe");
}
