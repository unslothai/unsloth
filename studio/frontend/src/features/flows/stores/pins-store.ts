// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { MediaValue } from "../types";

/**
 * Outputs pinned from a node's history, per flow, kept in this browser only. They go with each run as `pinned`,
 * so a pinned node is not run and its pinned output feeds the next blocks. Kept out of the flow document, so
 * sharing or exporting a flow never carries them.
 */
type Pins = Record<string, Record<string, unknown>>;

const KEY = (flowId: string) => `unsloth.flows.pins.${flowId}`;

function read(flowId: string): Pins {
  try {
    const raw = globalThis.localStorage?.getItem(KEY(flowId));
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? (parsed as Pins) : {};
  } catch {
    return {};
  }
}

function write(flowId: string, pins: Pins): void {
  try {
    if (Object.keys(pins).length)
      globalThis.localStorage?.setItem(KEY(flowId), JSON.stringify(pins));
    else globalThis.localStorage?.removeItem(KEY(flowId));
  } catch {
    // Storage full or blocked: the pin lasts for this page only.
  }
}

interface PinsState {
  flowId: string | null;
  pins: Pins;
  load: (flowId: string) => void;
  setPin: (nodeId: string, outputs: Record<string, unknown> | null) => void;
}

export const usePinsStore = create<PinsState>((set, get) => ({
  flowId: null,
  pins: {},
  load: (flowId) => set({ flowId, pins: read(flowId) }),
  setPin: (nodeId, outputs) => {
    const { flowId, pins } = get();
    if (!flowId) return;
    const next = { ...pins };
    if (outputs) next[nodeId] = outputs;
    else delete next[nodeId];
    write(flowId, next);
    set({ pins: next });
  },
}));

/** Pins for the nodes still in the flow; a pin on a deleted node is dropped. */
export function pinsForRun(nodeIds: Iterable<string>): Pins {
  const { pins } = usePinsStore.getState();
  const live = new Set(nodeIds);
  return Object.fromEntries(
    Object.entries(pins).filter(([nodeId]) => live.has(nodeId)),
  );
}

export function sameOutputs(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function isMedia(value: unknown): value is MediaValue {
  return (
    !!value &&
    typeof value === "object" &&
    typeof (value as { ref?: unknown }).ref === "string" &&
    ["image", "audio", "video"].includes(
      String((value as { kind?: unknown }).kind),
    )
  );
}

/** What a history cell shows: the first media value in the outputs, else a short text. */
export function historyPreview(
  outputs: Record<string, unknown>,
): { media: MediaValue } | { text: string } {
  for (const value of Object.values(outputs)) {
    if (isMedia(value)) return { media: value };
    if (Array.isArray(value) && isMedia(value[0])) return { media: value[0] };
  }
  const first = Object.values(outputs)[0];
  const text =
    typeof first === "string" ? first : JSON.stringify(first ?? null);
  return { text: text.length > 80 ? `${text.slice(0, 79)}…` : text };
}
