// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Music page's shapes: what the status says a music model can do (`audio_music`), and the
// drafts the page keeps per mode. Free of app imports so the node test runner can load it.

import type { AudioSourceSelection } from "../audio-run-request";

export type MusicMode = "song" | "sfx" | "edit";

export const MUSIC_MODES: readonly MusicMode[] = ["song", "sfx", "edit"];

/** ACE-Step: repaint, extend, cover, continue. Stable Audio: inpaint, restyle. */
export type MusicEditAction =
  | "repaint"
  | "extend"
  | "cover"
  | "continue"
  | "inpaint"
  | "restyle";

export const MUSIC_EDIT_ACTIONS: readonly MusicEditAction[] = [
  "repaint",
  "extend",
  "cover",
  "continue",
  "inpaint",
  "restyle",
];

export interface MusicRange {
  start_s: number;
  end_s: number;
}

export interface MusicDurationRule {
  min: number;
  max: number;
  default: number;
  /** MiniMax and YuE2 treat length as a target, not an exact cut. */
  approximate: boolean;
}

export interface MusicVariationsRule {
  max: number;
  /** "batch": one call makes them all (Stable Audio); "sequential": one call each. */
  how: "batch" | "sequential";
  /** How many one call can make with the server as loaded; more needs a reload. */
  loaded: number;
}

export interface MusicModeRule {
  id: MusicMode;
  lyrics?: "required" | "optional" | "unused";
  description?: "required" | "optional";
  instrumental?: "toggle" | "always" | "never";
  /** Section tag casing the model was trained on: `[verse]` or `[Verse]`. */
  section_case?: "lower" | "title" | null;
  duration?: MusicDurationRule | null;
  variations?: MusicVariationsRule | null;
  actions?: MusicEditAction[];
  max_ranges?: number;
  max_source_s?: number;
}

export interface MusicCapabilities {
  modes: MusicModeRule[];
}

/** Song mode's own values. Lyrics and the description stay in the page's existing drafts. */
export interface MusicSongDraft {
  instrumental: boolean;
  durationS: number | null;
  variations: number;
}

export interface MusicSfxDraft {
  prompt: string;
  durationS: number | null;
  variations: number;
}

export interface MusicEditDraft {
  source: AudioSourceSelection | null;
  action: MusicEditAction | null;
  ranges: MusicRange[];
  /** Cover strength or "How much to change"; null means the model default. */
  strength: number | null;
  extendS: number;
  /** What the changed part should sound like. */
  prompt: string;
}

export interface MusicDrafts {
  song: MusicSongDraft;
  sfx: MusicSfxDraft;
  edit: MusicEditDraft;
}

/** Parses the status `audio_music` value; null when absent or malformed (the legacy rail). */
export function parseMusicCapabilities(
  value: unknown,
): MusicCapabilities | null {
  if (!value || typeof value !== "object") return null;
  const modes = (value as { modes?: unknown }).modes;
  if (!Array.isArray(modes)) return null;
  const parsed: MusicModeRule[] = [];
  for (const raw of modes) {
    if (!raw || typeof raw !== "object") continue;
    const mode = raw as Record<string, unknown>;
    if (!MUSIC_MODES.includes(mode.id as MusicMode)) continue;
    if (parsed.some((rule) => rule.id === mode.id)) continue;
    parsed.push(parseModeRule(mode));
  }
  return parsed.length > 0 ? { modes: parsed } : null;
}

function oneOf<T extends string>(
  value: unknown,
  allowed: readonly T[],
): T | undefined {
  return allowed.includes(value as T) ? (value as T) : undefined;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseDuration(value: unknown): MusicDurationRule | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const min = finite(raw.min);
  const max = finite(raw.max);
  if (min === null || max === null || max < min) return null;
  const fallback = finite(raw.default);
  return {
    min,
    max,
    default:
      fallback === null
        ? Math.min(max, Math.max(min, 30))
        : Math.min(max, Math.max(min, fallback)),
    approximate: raw.approximate === true,
  };
}

function parseVariations(value: unknown): MusicVariationsRule | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const max = finite(raw.max);
  if (max === null || max < 2) return null;
  const loaded = finite(raw.loaded);
  return {
    max: Math.floor(max),
    how: raw.how === "batch" ? "batch" : "sequential",
    loaded: loaded === null ? 1 : Math.max(1, Math.floor(loaded)),
  };
}

function parseModeRule(mode: Record<string, unknown>): MusicModeRule {
  const id = mode.id as MusicMode;
  const actions = Array.isArray(mode.actions)
    ? mode.actions.filter((action): action is MusicEditAction =>
        MUSIC_EDIT_ACTIONS.includes(action as MusicEditAction),
      )
    : undefined;
  const maxRanges = finite(mode.max_ranges);
  const maxSource = finite(mode.max_source_s);
  return {
    id,
    lyrics: oneOf(mode.lyrics, ["required", "optional", "unused"] as const),
    description: oneOf(mode.description, ["required", "optional"] as const),
    instrumental: oneOf(mode.instrumental, [
      "toggle",
      "always",
      "never",
    ] as const),
    section_case: oneOf(mode.section_case, ["lower", "title"] as const) ?? null,
    duration: parseDuration(mode.duration),
    variations: parseVariations(mode.variations),
    ...(actions ? { actions } : {}),
    ...(maxRanges !== null
      ? { max_ranges: Math.max(0, Math.floor(maxRanges)) }
      : {}),
    ...(maxSource !== null ? { max_source_s: maxSource } : {}),
  };
}
