// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure rules behind the gallery filters, the "needs" line and the download check before a run. */

import type { PlanModel } from "./api";

export type GalleryCategory =
  | "games"
  | "images"
  | "video"
  | "audio"
  | "documents"
  | "chat"
  | "data"
  | "training";

export const GALLERY_CATEGORIES: { id: GalleryCategory; label: string }[] = [
  { id: "games", label: "Games" },
  { id: "images", label: "Images" },
  { id: "video", label: "Video" },
  { id: "audio", label: "Audio" },
  { id: "documents", label: "Documents" },
  { id: "chat", label: "Chat and agents" },
  { id: "data", label: "Data" },
  { id: "training", label: "Training" },
];

export interface CategorySource {
  kind: string;
  tags?: string[] | null;
  needs?: string[] | null;
}

const TAGS: Record<Exclude<GalleryCategory, "games">, string[]> = {
  images: [
    "image",
    "vision",
    "comic",
    "inpaint",
    "outpaint",
    "controlnet",
    "lora",
  ],
  video: ["video", "image to video"],
  audio: ["audio", "tts", "music", "podcast", "transcribe", "voice clone"],
  documents: ["documents", "knowledge base", "rag", "writing", "learning"],
  chat: ["chat", "agent", "memory", "web search"],
  data: [
    "data",
    "json",
    "table",
    "chart",
    "extract",
    "classify",
    "automation",
    "webhook",
    "schedule",
  ],
  training: ["training", "notebook", "fine-tune"],
};

const NEEDS: Partial<Record<GalleryCategory, string[]>> = {
  images: ["image", "vision"],
  video: ["video"],
  audio: ["tts", "clone", "music", "audio", "stt"],
  training: ["training"],
};

/** Every category an example belongs to; an example with no match shows under All only. */
export function categoriesOf(flow: CategorySource): GalleryCategory[] {
  const tags = new Set((flow.tags ?? []).map((t) => t.toLowerCase()));
  const needs = new Set(flow.needs ?? []);
  const out: GalleryCategory[] = [];
  for (const { id } of GALLERY_CATEGORIES) {
    const hit =
      id === "games"
        ? flow.kind === "story" || flow.kind === "world"
        : TAGS[id].some((t) => tags.has(t)) ||
          (NEEDS[id] ?? []).some((n) => needs.has(n));
    if (hit) out.push(id);
  }
  return out;
}

/** Category chips worth showing: only those with at least one example, in a fixed order. */
export function visibleCategories(
  flows: CategorySource[],
): { id: GalleryCategory; label: string; count: number }[] {
  const counts = new Map<GalleryCategory, number>();
  for (const flow of flows) {
    for (const c of categoriesOf(flow)) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  return GALLERY_CATEGORIES.filter((c) => counts.has(c.id)).map((c) => ({
    ...c,
    count: counts.get(c.id) ?? 0,
  }));
}

const NEED_LABEL: Record<string, string> = {
  chat: "chat model",
  vision: "vision model",
  image: "image model",
  video: "video model",
  tts: "voice model",
  clone: "voice model",
  music: "music model",
  audio: "music model",
  stt: "speech to text",
  embedding: "embeddings",
  training: "training (GPU)",
};

/** The card line: "No models needed", or "Needs: chat model, image model". */
export function needsLine(needs: string[] | null | undefined): string | null {
  if (!needs) return null;
  if (needs.length === 0) return "No models needed";
  const labels: string[] = [];
  for (const need of needs) {
    const label = NEED_LABEL[need] ?? need;
    if (!labels.includes(label)) labels.push(label);
  }
  return `Needs: ${labels.join(", ")}`;
}

export function modelsToDownload(
  models: PlanModel[] | null | undefined,
): PlanModel[] {
  return (models ?? []).filter(
    (m) => Boolean(m.repo) && !m.loaded && m.downloaded === false,
  );
}

/**
 * `stale`: the flow or a model pick changed while the check was out; `clear`: nothing to download (safe to
 * remember); `unchecked`: the plan had errors, so its empty model list proves nothing.
 */
export function planCheck(
  plan: { models?: PlanModel[] | null; errors?: unknown[] | null },
  askedFor: { flowId: string; models: string },
  current: { flowId: string | null | undefined; models: string },
): "stale" | "download" | "clear" | "unchecked" {
  if (current.flowId !== askedFor.flowId || current.models !== askedFor.models) return "stale";
  if (modelsToDownload(plan.models).length > 0) return "download";
  return plan.errors && plan.errors.length > 0 ? "unchecked" : "clear";
}

export function downloadBytes(models: PlanModel[]): number | null {
  let total = 0;
  for (const m of models) {
    if (typeof m.size_bytes !== "number") return null;
    total += m.size_bytes;
  }
  return total;
}

/** Which models the enabled blocks pick, as one string (the download check's cache key). */
export function modelSignature(
  nodes: Record<
    string,
    { params?: Record<string, unknown> | null; disabled?: boolean } | undefined
  >,
): string {
  const picks: string[] = [];
  for (const node of Object.values(nodes)) {
    // A disabled block plans nothing: enabling it must ask the plan again.
    if (node?.disabled) continue;
    for (const value of Object.values(node?.params ?? {})) {
      if (value && typeof value === "object" && "repo" in value) {
        const v = value as { repo?: unknown; variant?: unknown };
        if (typeof v.repo === "string" && v.repo)
          picks.push(
            `${v.repo.toLowerCase()}:${String(v.variant ?? "").toLowerCase()}`,
          );
      }
    }
  }
  return picks.sort().join("|");
}

export interface StepFailure {
  id: string;
  title: string;
  error: string;
}

/** The failed blocks of a run in graph order, named the way the canvas names them, for "X failed: why". */
export function failedSteps(
  nodeOrder: string[],
  nodes: Record<string, { type: string; label?: string | null } | undefined>,
  runtime: Record<
    string,
    { status: string; error?: string | null } | undefined
  >,
  titleOf: (type: string) => string | undefined,
): StepFailure[] {
  const out: StepFailure[] = [];
  for (const id of nodeOrder) {
    const rt = runtime[id];
    const node = nodes[id];
    if (!node || rt?.status !== "failed") continue;
    out.push({
      id,
      title: node.label || titleOf(node.type) || node.type,
      error: (rt.error ?? "").trim() || "It stopped with an error",
    });
  }
  return out;
}
