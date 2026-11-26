// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Fn = (...args: never[]) => unknown;

export const deps: {
  calls: string[];
  getRecipeJobStatus: Fn;
  getRecipeJobDataset: Fn;
  getRecipeJobAnalysis: Fn;
  streamRecipeJobEvents: Fn;
} = {
  calls: [],
  getRecipeJobStatus: () => undefined,
  getRecipeJobDataset: () => undefined,
  getRecipeJobAnalysis: () => undefined,
  streamRecipeJobEvents: () => undefined,
};

export const getRecipeJobStatus = (...args: never[]) =>
  deps.getRecipeJobStatus(...args);
export const getRecipeJobDataset = (...args: never[]) =>
  deps.getRecipeJobDataset(...args);
export const getRecipeJobAnalysis = (...args: never[]) =>
  deps.getRecipeJobAnalysis(...args);
export const streamRecipeJobEvents = (...args: never[]) =>
  deps.streamRecipeJobEvents(...args);

export function toastSuccess(message: string): void {
  deps.calls.push(`toastSuccess:${message}`);
}

export function toastError(message: string, description?: string): void {
  deps.calls.push(`toastError:${message}|${description ?? ""}`);
}

export function bumpInventoryVersion(): void {
  deps.calls.push("bumpInventoryVersion");
}
