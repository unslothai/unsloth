// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The benchmark API for the runtime store tests: every run answers success, a status read is idle.
export const calls: string[] = [];

export async function getBenchmarkStatus() {
  return { is_benchmark_active: false, last_op_seq: 0 };
}

export async function runBenchmark() {
  calls.push("run");
  return { success: true };
}

export async function cancelBenchmark() {
  calls.push("cancel");
  return { success: true };
}

export function isRecoverableTransportError() {
  return false;
}
