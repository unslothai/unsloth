// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const calls: string[] = [];

export async function getBenchmarkStatus() {
  return { is_benchmark_active: false, last_op_seq: 0 };
}

export const runParams: Record<string, unknown>[] = [];

export async function runBenchmark(params: Record<string, unknown>) {
  calls.push("run");
  runParams.push(params);
  return { success: true };
}

export async function cancelBenchmark() {
  calls.push("cancel");
  return { success: true };
}

export function isRecoverableTransportError() {
  return false;
}
