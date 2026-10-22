// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkLogEntry } from "../api/benchmark-api";

export function getBenchmarkLogLineClass(entry: BenchmarkLogEntry): string {
  if (entry.stream === "stderr") {
    return "text-rose-300/90";
  }
  if (entry.stream === "status") {
    return "text-sky-300/90";
  }
  return "text-green-400/90";
}
