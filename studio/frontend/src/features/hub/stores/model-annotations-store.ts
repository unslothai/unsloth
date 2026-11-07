// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth";
import { create } from "zustand";
import {
  readModelAnnotations,
  writeModelAnnotation,
} from "../api/model-annotations";
import type {
  ModelAnnotation,
  ModelAnnotations,
} from "../lib/model-annotations";

export interface ModelAnnotationTarget {
  id: string;
  title: string;
  annotation?: ModelAnnotation;
}

interface ModelAnnotationsState {
  annotations: ModelAnnotations;
  loaded: boolean;
  error: string | null;
  editing: ModelAnnotationTarget | null;
}

export const useModelAnnotationsStore = create<ModelAnnotationsState>(() => ({
  annotations: {},
  loaded: false,
  error: null,
  editing: null,
}));

let hydration: Promise<void> | null = null;

/** Load once per login, deduplicating parallel mounts and ignoring a previous account's response. */
export function loadModelAnnotations(): Promise<void> {
  if (useModelAnnotationsStore.getState().loaded) return Promise.resolve();
  if (hydration) return hydration;
  const epoch = getAuthSessionEpoch();
  useModelAnnotationsStore.setState({ error: null });
  hydration = readModelAnnotations()
    .then(
      (annotations) => {
        if (getAuthSessionEpoch() === epoch) {
          useModelAnnotationsStore.setState({
            annotations,
            loaded: true,
            error: null,
          });
        }
      },
      (error: unknown) => {
        if (getAuthSessionEpoch() === epoch) {
          useModelAnnotationsStore.setState({
            error:
              error instanceof Error
                ? error.message
                : "Could not load model details",
          });
        }
      },
    )
    .finally(() => {
      if (getAuthSessionEpoch() === epoch) hydration = null;
    });
  return hydration;
}

/** Open one editor outside virtualized rows so scrolling cannot discard a draft. */
export function editModelAnnotation(target: ModelAnnotationTarget): void {
  useModelAnnotationsStore.setState({ editing: target });
}

/** Close the editor without changing saved data. */
export function closeModelAnnotationEditor(): void {
  useModelAnnotationsStore.setState({ editing: null });
}

/** Update the cache only after a confirmed save, retaining other rows and current-session isolation. */
export async function saveModelAnnotation(
  modelId: string,
  annotation: ModelAnnotation,
): Promise<void> {
  const epoch = getAuthSessionEpoch();
  const saved = await writeModelAnnotation(modelId, annotation);
  if (getAuthSessionEpoch() !== epoch) return;
  useModelAnnotationsStore.setState((state) => {
    const annotations = { ...state.annotations };
    if (saved) annotations[modelId] = saved;
    else delete annotations[modelId];
    return {
      annotations,
      editing: state.editing?.id === modelId ? null : state.editing,
    };
  });
}

/** Personal notes are server-backed; never carry them or an open draft into another login. */
function resetModelAnnotations(): void {
  hydration = null;
  useModelAnnotationsStore.setState({
    annotations: {},
    loaded: false,
    error: null,
    editing: null,
  });
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetModelAnnotations);
  window.addEventListener(AUTH_SESSION_STORED_EVENT, resetModelAnnotations);
}
