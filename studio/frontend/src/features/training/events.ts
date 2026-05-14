// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingRunSummary } from "./types/history";

type UpdateListener = (run: TrainingRunSummary) => void;
type DeleteListener = (runId: string) => void;
type ChangedListener = () => void;

const updateListeners = new Set<UpdateListener>();
const deleteListeners = new Set<DeleteListener>();
const changedListeners = new Set<ChangedListener>();

export function onTrainingRunUpdated(fn: UpdateListener): () => void {
  updateListeners.add(fn);
  return () => {
    updateListeners.delete(fn);
  };
}

export function onTrainingRunDeleted(fn: DeleteListener): () => void {
  deleteListeners.add(fn);
  return () => {
    deleteListeners.delete(fn);
  };
}

export function onTrainingRunsChanged(fn: ChangedListener): () => void {
  changedListeners.add(fn);
  return () => {
    changedListeners.delete(fn);
  };
}

export function emitTrainingRunUpdated(run: TrainingRunSummary): void {
  for (const fn of updateListeners) {
    fn(run);
  }
}

export function emitTrainingRunDeleted(runId: string): void {
  for (const fn of deleteListeners) {
    fn(runId);
  }
}

export function emitTrainingRunsChanged(): void {
  for (const fn of changedListeners) {
    fn();
  }
}
