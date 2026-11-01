// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { MediaValue } from "../types";

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
  } catch {}
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

export type HistoryKind =
  | "image"
  | "audio"
  | "video"
  | "text"
  | "number"
  | "list"
  | "json"
  | "dataset";

/** A history entry as a small Library-style card: what it is, a name, and one line of detail. */
export function historyCard(outputs: Record<string, unknown>): {
  kind: HistoryKind;
  title: string;
  detail: string;
  media?: MediaValue;
} {
  const preview = historyPreview(outputs);
  if ("media" in preview) {
    const m = preview.media;
    const size =
      m.kind === "image" && m.width && m.height
        ? `${m.width}×${m.height}`
        : m.duration_s
          ? `${Math.round(m.duration_s)}s`
          : "";
    const name = m.kind[0].toUpperCase() + m.kind.slice(1);
    return {
      kind: m.kind,
      title: m.prompt?.trim() || name,
      detail: size || name,
      media: m,
    };
  }
  const value = Object.values(outputs)[0];
  if (typeof value === "string")
    return {
      kind: "text",
      title: value.trim() || "Empty text",
      detail: `${value.length} chars`,
    };
  if (typeof value === "number" || typeof value === "boolean")
    return {
      kind: "number",
      title: String(value),
      detail: typeof value === "boolean" ? "Toggle" : "Number",
    };
  if (Array.isArray(value))
    return {
      kind: "list",
      title: `${value.length} item${value.length === 1 ? "" : "s"}`,
      detail: "List",
    };
  if (
    value &&
    typeof value === "object" &&
    (value as { kind?: unknown }).kind === "dataset"
  ) {
    const rows = (value as { rows?: number | null }).rows;
    const ref = String((value as { ref?: unknown }).ref ?? "Dataset");
    return {
      kind: "dataset",
      title: ref.split("/").pop() || ref,
      detail: rows ? `${rows} rows` : "Dataset",
    };
  }
  const keys =
    value && typeof value === "object" ? Object.keys(value).length : 0;
  return {
    kind: "json",
    title: "JSON",
    detail: `${keys} key${keys === 1 ? "" : "s"}`,
  };
}

/** "just now", "5m ago", "3h ago", then the date. `at` is epoch ms, as the history API sends it. */
export function historyAgo(at: number, now = Date.now()): string {
  const seconds = Math.max(0, (now - at) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(at).toLocaleDateString();
}
