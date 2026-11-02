// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageTiming } from "@assistant-ui/core";

/** What the earlier passes of a continued reply took. Continue carries it so the merged reply's timing
 *  describes the whole answer, as its text and "Worked for" time already do. */
export type PriorPassTiming = {
  totalStreamTime: number;
  totalChunks: number;
  toolCallCount: number;
  tokenCount?: number;
  firstTokenTime?: number;
  durationMs?: number;
  startedAt?: number;
  serverTimings?: Record<string, number>;
};

function count(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function numbers(value: unknown): Record<string, number> | undefined {
  const fields = record(value);
  if (!fields) return undefined;
  const kept = Object.entries(fields).filter(
    (entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
  return kept.length > 0 ? Object.fromEntries(kept) : undefined;
}

/** A reply's own figures, read from its metadata; undefined when it recorded no timing. A reply that was
 *  stopped or broke off carries them, with every earlier pass, as `resumeTiming`: a Stop keeps only the
 *  last streamed metadata, which has no `timing`, and an error's has no server figures. */
export function priorPassTiming(metadata: unknown): PriorPassTiming | undefined {
  const message = record(metadata);
  const custom = record(message?.custom);
  const carried = readPriorPassTiming(custom?.resumeTiming);
  if (carried) return carried;
  const timing = record(custom?.timing) ?? record(message?.timing);
  const details = record(custom?.responseDetails);
  const serverTimings = record(custom?.serverTimings);
  return readPriorPassTiming({
    totalStreamTime: timing?.totalStreamTime,
    totalChunks: timing?.totalChunks,
    toolCallCount: timing?.toolCallCount,
    tokenCount: timing?.tokenCount,
    firstTokenTime: timing?.firstTokenTime,
    durationMs: details?.durationMs,
    startedAt: details?.startedAt,
    // A diffusion model's figures are steps and blocks, not a decode that continues.
    serverTimings: serverTimings?.diffusion === true ? undefined : serverTimings,
  });
}

/** Validates a carried PriorPassTiming; undefined for anything without a total time. */
export function readPriorPassTiming(value: unknown): PriorPassTiming | undefined {
  const fields = record(value);
  const totalStreamTime = count(fields?.totalStreamTime);
  if (!fields || totalStreamTime === undefined) return undefined;
  const optional = {
    tokenCount: count(fields.tokenCount),
    firstTokenTime: count(fields.firstTokenTime),
    durationMs: count(fields.durationMs),
    startedAt: count(fields.startedAt),
    serverTimings: numbers(fields.serverTimings),
  };
  return {
    totalStreamTime,
    totalChunks: count(fields.totalChunks) ?? 0,
    toolCallCount: count(fields.toolCallCount) ?? 0,
    ...Object.fromEntries(Object.entries(optional).filter(([, v]) => v !== undefined)),
  };
}

/** Counts and times add up over the passes; the first token is the reply's first. */
export function withPriorTiming(
  timing: MessageTiming,
  prior: PriorPassTiming | undefined,
  tokensPerSecond?: number,
): MessageTiming {
  if (!prior) return timing;
  const totalStreamTime = prior.totalStreamTime + (timing.totalStreamTime ?? 0);
  const tokenCount =
    prior.tokenCount !== undefined || timing.tokenCount !== undefined
      ? (prior.tokenCount ?? 0) + (timing.tokenCount ?? 0)
      : undefined;
  return {
    ...timing,
    firstTokenTime:
      prior.firstTokenTime ??
      (timing.firstTokenTime !== undefined ? prior.totalStreamTime + timing.firstTokenTime : undefined),
    totalStreamTime,
    tokenCount,
    tokensPerSecond:
      tokensPerSecond ??
      (tokenCount !== undefined && totalStreamTime > 0 ? tokenCount / (totalStreamTime / 1000) : undefined),
    totalChunks: prior.totalChunks + timing.totalChunks,
    toolCallCount: prior.toolCallCount + timing.toolCallCount,
  };
}

const ADDED = ["prompt_n", "prompt_ms", "cache_n", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted"];

/** llama-server's figures summed over the passes. A rate is the time-weighted mean of the passes'
 *  rates, which keeps llama.cpp's own definition (decode steps over decode time). */
export function withPriorServerTimings<T extends object>(
  current: T | undefined,
  prior: Record<string, number> | undefined,
): T | undefined {
  const now = current as Record<string, unknown> | undefined;
  if (!now || !prior || now.diffusion === true) return current;
  const value = (fields: Record<string, unknown>, key: string) => count(fields[key]) ?? 0;
  const merged: Record<string, unknown> = { ...now };
  for (const key of ADDED) {
    if (key in now || key in prior) merged[key] = value(now, key) + value(prior, key);
  }
  for (const [rate, ms, n, perToken] of [
    ["prompt_per_second", "prompt_ms", "prompt_n", "prompt_per_token_ms"],
    ["predicted_per_second", "predicted_ms", "predicted_n", "predicted_per_token_ms"],
  ] as const) {
    const totalMs = value(merged, ms);
    if (totalMs <= 0) continue;
    merged[rate] = (value(now, rate) * value(now, ms) + value(prior, rate) * value(prior, ms)) / totalMs;
    const tokens = value(merged, n);
    if (tokens > 0) merged[perToken] = totalMs / tokens;
  }
  return merged as T;
}

/** The earlier passes plus this one, in the same shape: what a Continue after this pass carries. */
export function addPass(prior: PriorPassTiming | undefined, pass: PriorPassTiming): PriorPassTiming {
  if (!prior) return pass;
  const optional = {
    tokenCount:
      prior.tokenCount !== undefined || pass.tokenCount !== undefined
        ? (prior.tokenCount ?? 0) + (pass.tokenCount ?? 0)
        : undefined,
    firstTokenTime:
      prior.firstTokenTime ??
      (pass.firstTokenTime !== undefined ? prior.totalStreamTime + pass.firstTokenTime : undefined),
    durationMs:
      prior.durationMs !== undefined || pass.durationMs !== undefined
        ? (prior.durationMs ?? prior.totalStreamTime) + (pass.durationMs ?? pass.totalStreamTime)
        : undefined,
    startedAt: prior.startedAt ?? pass.startedAt,
    serverTimings: withPriorServerTimings(pass.serverTimings, prior.serverTimings) ?? prior.serverTimings,
  };
  return {
    totalStreamTime: prior.totalStreamTime + pass.totalStreamTime,
    totalChunks: prior.totalChunks + pass.totalChunks,
    toolCallCount: prior.toolCallCount + pass.toolCallCount,
    ...Object.fromEntries(Object.entries(optional).filter(([, v]) => v !== undefined)),
  };
}
