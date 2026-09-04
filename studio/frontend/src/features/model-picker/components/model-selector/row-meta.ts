// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers for model-row presentation: owner/name split, lora trigger label, format pills,
// param chip, tabular size. No React/DOM deps so they stay easy to test.

import type { LoraModelOption } from "./types";

export type FormatTone = "gguf" | "mlx" | "checkpoint" | "adapter";

// Format keyword to DotTag tone. Looked up by full token and by first word, so "Full finetune" resolves via "full".
export const FORMAT_TONE: Record<string, FormatTone> = {
  gguf: "gguf",
  mlx: "mlx",
  local: "checkpoint",
  safetensors: "checkpoint",
  checkpoint: "checkpoint",
  lora: "adapter",
  merged: "adapter",
  adapter: "adapter",
  exported: "adapter",
  full: "adapter",
};

/** Split "owner/name" on the last slash. No slash means name only. */
export function splitRepoLabel(label: string): {
  owner: string | null;
  name: string;
} {
  const slash = label.lastIndexOf("/");
  if (slash <= 0 || slash === label.length - 1) {
    return { owner: null, name: label };
  }
  return { owner: label.slice(0, slash), name: label.slice(slash + 1) };
}

/** Our official Hub owners. Rows drop the owner prefix, since the Unsloth section already carries
 *  it; the full repo id stays in the row tooltip. */
export function isUnslothOwner(owner: string | null | undefined): boolean {
  const normalized = owner?.toLowerCase();
  return normalized === "unsloth" || normalized === "unslothai";
}

/** Selector trigger label. Training and exported runs are named "<run>/<model>", so the run is the
 *  label; local inventory rows keep theirs, since an Ollama repo name carries real slashes. */
export function loraOptionLabel(
  option: Pick<LoraModelOption, "name" | "source">,
): string {
  if (option.source === "local" || !option.name.includes("/")) {
    return option.name;
  }
  return option.name.split("/")[0].trim();
}

export type MetaToken =
  | { kind: "format"; label: string; tone: FormatTone }
  | { kind: "size"; label: string }
  | { kind: "param"; label: string }
  | { kind: "text"; label: string };

const META_SIZE_RE = /(?:KB|MB|GB|TB)\b/i;
const META_APPROX_RE = /^~/;
const META_PARAM_RE = /^\d+(?:\.\d+)?B$/i;
const META_WHITESPACE_RE = /\s+/;

/** Classify a meta token: size (has KB/MB/GB/TB or leading "~"), param (bare "<n>B"), format
 *  keyword, or plain text. */
export function classifyMetaToken(raw: string): MetaToken | null {
  const t = raw.trim();
  if (!t) return null;
  if (META_SIZE_RE.test(t) || META_APPROX_RE.test(t)) {
    return { kind: "size", label: t };
  }
  if (META_PARAM_RE.test(t)) {
    return { kind: "param", label: t.toUpperCase() };
  }
  const lower = t.toLowerCase();
  const tone =
    FORMAT_TONE[lower] ?? FORMAT_TONE[lower.split(META_WHITESPACE_RE)[0]];
  if (tone) {
    return { kind: "format", label: t, tone };
  }
  return { kind: "text", label: t };
}

/** Parse the dot-separated meta string into structured tokens. */
export function parseMetaTokens(meta?: string | null): {
  formats: { label: string; tone: FormatTone }[];
  param?: string;
  size?: string;
  texts: string[];
} {
  const formats: { label: string; tone: FormatTone }[] = [];
  const texts: string[] = [];
  let param: string | undefined;
  let size: string | undefined;
  if (!meta) return { formats, texts };
  for (const part of meta.split("·")) {
    const token = classifyMetaToken(part);
    if (!token) continue;
    if (token.kind === "format") {
      formats.push({ label: token.label, tone: token.tone });
    } else if (token.kind === "size") {
      size ??= token.label;
    } else if (token.kind === "param") {
      param ??= token.label;
    } else {
      texts.push(token.label);
    }
  }
  return { formats, param, size, texts };
}
