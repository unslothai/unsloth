// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const calls: string[] = [];

let statusGate: Promise<void> | null = null;

/** Hold the next getBenchmarkStatus() calls until `gate` resolves; null releases. */
export function setStatusGate(gate: Promise<void> | null): void {
  statusGate = gate;
}

let statusScript: Record<string, unknown>[] = [];
export const statusTokens: (string | undefined)[] = [];

/** Answers for the next getBenchmarkStatus() calls, in order; the last one repeats. */
export function setStatusScript(script: Record<string, unknown>[]): void {
  statusScript = script;
}

export async function getBenchmarkStatus(runToken?: string) {
  if (statusGate) await statusGate;
  statusTokens.push(runToken);
  if (statusScript.length > 1) return statusScript.shift();
  return statusScript[0] ?? { is_benchmark_active: false, last_op_seq: 0 };
}

let runDrops = false;

/** Make runBenchmark() fail as a dropped connection. */
export function setRunDrops(value: boolean): void {
  runDrops = value;
}

export const runParams: Record<string, unknown>[] = [];

export async function runBenchmark(params: Record<string, unknown>) {
  calls.push("run");
  runParams.push(params);
  if (runDrops)
    throw Object.assign(new Error("dropped"), { unslothTransportFailure: true });
  return { success: true };
}

export async function cancelBenchmark() {
  calls.push("cancel");
  return { success: true };
}

export function isRecoverableTransportError(err: unknown) {
  return (
    (err as { unslothTransportFailure?: unknown } | null)
      ?.unslothTransportFailure === true
  );
}
