// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type {
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "../components/model-selector/types";
import {
  ggufVariantsMatch,
  isOllamaLinkPath,
  isStandaloneGgufPath,
  modelDisplayName,
  residentModelIdMatches,
} from "./model-identity";
import { adoptLegacyConfigKey } from "./per-model-config";

export interface ModelConfigHandoffRequest {
  requestId: string;
  id: string;
  displayName?: string;
  meta: ModelSelectorChangeMeta;
}

interface ModelConfigHandoffState {
  request: ModelConfigHandoffRequest | null;
  submit: (request: ModelConfigHandoffRequest) => void;
  clear: (requestId: string) => void;
}

interface RandomUuidSource {
  randomUUID?: () => string;
}

function fallbackRequestId(): string {
  const random = Math.random().toString(36).slice(2, 10).padEnd(8, "0");
  return `${Date.now()}-${random}`;
}

export function createModelConfigHandoffRequestId(
  cryptoSource: RandomUuidSource | null = globalThis.crypto ?? null,
): string {
  if (typeof cryptoSource?.randomUUID !== "function") {
    return fallbackRequestId();
  }
  try {
    return cryptoSource.randomUUID();
  } catch {
    return fallbackRequestId();
  }
}

export function modelConfigTarget(
  id: string,
  meta: ModelSelectorChangeMeta,
  displayName?: string,
): ModelPickTarget {
  const name = displayName?.trim() || modelDisplayName(id);
  const isGguf = meta.isGguf ?? Boolean(meta.ggufVariant);
  const separateLoadId =
    meta.source === "hub" && Boolean(meta.loadId?.trim()) && meta.loadId !== id
      ? meta.loadId
      : null;
  const loadId = separateLoadId ?? id;
  return {
    id: loadId,
    displayName: meta.ggufVariant ? `${name} · ${meta.ggufVariant}` : name,
    ggufVariant: meta.ggufVariant ?? null,
    isGguf,
    apiLoadable: isGguf && !isOllamaLinkPath(id) && !isOllamaLinkPath(loadId),
    ...(separateLoadId ? { configId: id } : {}),
    meta,
  };
}

export function modelConfigTargetMatchesSelection(
  target: ModelPickTarget,
  selectedId: string | null | undefined,
): boolean {
  return residentModelIdMatches(
    selectedId,
    target.configId,
    target.id,
    target.meta.loadId,
  );
}

export function modelConfigTargetIsResident({
  target,
  selectedId,
  activeGgufVariant,
  loaded = true,
}: {
  target: ModelPickTarget;
  selectedId: string | null | undefined;
  activeGgufVariant: string | null | undefined;
  loaded?: boolean;
}): boolean {
  if (!loaded) {
    return false;
  }
  if (!modelConfigTargetMatchesSelection(target, selectedId)) {
    return false;
  }
  if (
    target.isGguf &&
    target.ggufVariant == null &&
    (isStandaloneGgufPath(target.id) || isOllamaLinkPath(target.id))
  ) {
    return true;
  }
  return ggufVariantsMatch(activeGgufVariant, target.ggufVariant);
}

export function modelConfigHandoffForDestination(
  request: ModelConfigHandoffRequest | null,
  destination: {
    active: boolean;
    newChatId?: string | null;
    threadId?: string | null;
    compareId?: string | null;
    projectId?: string | null;
  },
): ModelConfigHandoffRequest | null {
  if (!request) {
    return null;
  }
  if (
    !destination.active ||
    destination.threadId ||
    destination.compareId ||
    destination.projectId ||
    request.requestId !== destination.newChatId
  ) {
    return null;
  }
  return request;
}

export const useModelConfigHandoffStore = create<ModelConfigHandoffState>(
  (set) => ({
    request: null,
    submit: (request) => set({ request }),
    clear: (requestId) =>
      set((state) =>
        state.request?.requestId === requestId ? { request: null } : state,
      ),
  }),
);

export function requestModelConfigHandoff(
  request: ModelConfigHandoffRequest,
): void {
  if (
    request.meta.source === "hub" &&
    request.meta.loadId?.trim() &&
    request.meta.loadId !== request.id
  ) {
    adoptLegacyConfigKey(
      request.id,
      request.meta.loadId,
      request.meta.ggufVariant,
    );
  }
  useModelConfigHandoffStore.getState().submit(request);
}

export function clearModelConfigHandoff(requestId: string): void {
  useModelConfigHandoffStore.getState().clear(requestId);
}
