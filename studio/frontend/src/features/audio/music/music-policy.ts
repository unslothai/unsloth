// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app imports: the node test runner loads this directly.

import type {
  AudioOptionScalar,
  AudioTextRunRequest,
  AudioSourceSelection,
} from "../audio-run-request";
import { sourceRefOf } from "../audio-run-request";
import { MUSIC_EDIT_DEFAULT_STRENGTH } from "./music-edit-rules";
import type {
  MusicCapabilities,
  MusicEditDraft,
  MusicMode,
  MusicModeRule,
  MusicSfxDraft,
  MusicSongDraft,
} from "./music-types";

export const MUSIC_MODE_LABELS: Record<MusicMode, string> = {
  song: "Song",
  sfx: "Sound effect",
  edit: "Edit",
};

export const MUSIC_MODE_HINTS: Record<MusicMode, string> = {
  song: "A song or an instrumental from a description and lyrics.",
  sfx: "A short sound from a description.",
  edit: "Change part of a clip, extend it, or give it a new style.",
};

export function musicModeLabel(rule: MusicModeRule): string {
  return rule.id === "song" && rule.instrumental === "always"
    ? "Instrumental"
    : MUSIC_MODE_LABELS[rule.id];
}

export function musicModeHint(rule: MusicModeRule): string {
  return rule.id === "song" && rule.instrumental === "always"
    ? "Music without vocals, from a description."
    : MUSIC_MODE_HINTS[rule.id];
}

export const MUSIC_MAX_VARIATIONS = 4;

export function effectiveMusicMode(
  capabilities: MusicCapabilities,
  picked: MusicMode,
): MusicModeRule {
  return (
    capabilities.modes.find((rule) => rule.id === picked) ??
    capabilities.modes[0]
  );
}

const SECTION_NAMES = [
  "Intro",
  "Verse",
  "Pre-chorus",
  "Chorus",
  "Bridge",
  "Outro",
] as const;

export interface SectionTag {
  name: string;
  tag: string;
}

export function sectionTags(
  sectionCase: MusicModeRule["section_case"],
): SectionTag[] {
  return SECTION_NAMES.map((name) => ({
    name,
    tag:
      sectionCase === "title"
        ? `[${name.replace(/(^|-)([a-z])/g, (_m, dash: string, ch: string) => dash + ch.toUpperCase())}]`
        : `[${name.toLowerCase()}]`,
  }));
}

export function insertSectionTag(
  text: string,
  cursor: number,
  tag: string,
): { text: string; cursor: number } {
  const at = Math.max(0, Math.min(cursor, text.length));
  const before = text.slice(0, at);
  const after = text.slice(at);
  const lead =
    before.length === 0
      ? ""
      : before.endsWith("\n\n")
        ? ""
        : before.endsWith("\n")
          ? "\n"
          : "\n\n";
  const tail = after.startsWith("\n") ? "" : "\n";
  const inserted = `${lead}${tag}${tail}`;
  return {
    text: before + inserted + after,
    cursor: before.length + lead.length + tag.length + 1,
  };
}

export interface InstrumentalChoice {
  shown: boolean;
  enabled: boolean;
  value: boolean;
  reason: string | null;
}

export function instrumentalChoice(
  rule: MusicModeRule,
  picked: boolean,
): InstrumentalChoice {
  switch (rule.instrumental) {
    case "toggle":
      return { shown: true, enabled: true, value: picked, reason: null };
    case "always":
      return {
        shown: false,
        enabled: false,
        value: true,
        reason: "This model makes instrumentals only.",
      };
    case "never":
      return {
        shown: true,
        enabled: false,
        value: false,
        reason: "This model always sings the lyrics.",
      };
    default:
      return { shown: false, enabled: false, value: false, reason: null };
  }
}

export function lyricsShown(
  rule: MusicModeRule,
  instrumental: boolean,
): boolean {
  if (rule.id !== "song") return false;
  if (rule.lyrics === "unused" || rule.lyrics === undefined) return false;
  return !instrumental;
}

export function durationLabel(rule: MusicModeRule): string {
  return rule.duration?.approximate
    ? "Length (seconds, approximate)"
    : "Length (seconds)";
}

export function musicDurationFor(
  rule: MusicModeRule,
  picked: number | null,
): number | null {
  const duration = rule.duration;
  if (!duration) return null;
  if (picked === null || !Number.isFinite(picked)) return duration.default;
  return Math.min(duration.max, Math.max(duration.min, picked));
}

export function variationsMax(rule: MusicModeRule): number | null {
  const variations = rule.variations;
  if (!variations || variations.max < 2) return null;
  return Math.min(variations.max, MUSIC_MAX_VARIATIONS);
}

export function variationsFor(rule: MusicModeRule, picked: number): number {
  const max = variationsMax(rule);
  if (max === null) return 1;
  return Math.max(1, Math.min(max, Math.floor(picked) || 1));
}

export function reloadNotice(
  rule: MusicModeRule,
  variations: number,
  modelName: string | null,
): string | null {
  const rules = rule.variations;
  if (!rules || rules.how !== "batch" || variations <= rules.loaded)
    return null;
  const name = modelName?.trim() || "the model";
  return `Reloads ${name} to make ${variations} variations at once, about 15 s the first time.`;
}

export const MUSIC_EXAMPLES = {
  description: [
    {
      label: "Acoustic pop",
      text: "Upbeat acoustic pop, bright female vocals, guitar and light drums, 100 BPM",
    },
    {
      label: "Lo-fi hip hop",
      text: "Lo-fi hip hop beat, warm vinyl crackle, mellow piano, relaxed",
    },
    {
      label: "Orchestral",
      text: "Epic orchestral trailer music, big drums, rising strings",
    },
  ],
  instrumental: [
    {
      label: "Lo-fi",
      text: "Warm lo-fi hip hop beat, vinyl crackle, mellow piano, relaxed",
    },
    {
      label: "House",
      text: "Uplifting house music with bright synths, 124 BPM",
    },
    {
      label: "Orchestral",
      text: "Epic orchestral trailer music, big drums, rising strings",
    },
  ],
  lyrics: [
    {
      label: "River song",
      text: "Sunlight on the river, golden in the morning\nWe walk along the water, singing as we go\n\nOh, oh, the river runs\nOh, oh, into the sun",
    },
  ],
  sfx: [
    { label: "Rain", text: "Heavy rain on a tin roof with distant thunder" },
    { label: "Footsteps", text: "Footsteps on gravel, slow walk" },
    { label: "Creaky door", text: "A door creaking open in an empty hall" },
  ],
  edit: [
    "Same melody with a saxophone lead",
    "Softer, with only piano and strings",
  ],
} as const;

export function exampleLyrics(
  sectionCase: MusicModeRule["section_case"],
  text: string,
): string {
  const [verse, chorus] = text.split("\n\n");
  const tags = sectionTags(sectionCase);
  const tag = (name: string) => tags.find((item) => item.name === name)?.tag;
  return chorus
    ? `${tag("Verse")}\n${verse}\n\n${tag("Chorus")}\n${chorus}`
    : `${tag("Verse")}\n${verse}`;
}

export interface MusicInputs {
  rule: MusicModeRule;
  description: string;
  lyrics: string;
  song: MusicSongDraft;
  sfx: MusicSfxDraft;
  edit: MusicEditDraft;
  editProblem?: string | null;
}

export function musicBlocker(inputs: MusicInputs): string | null {
  const { rule } = inputs;
  if (rule.id === "song") {
    const instrumental = instrumentalChoice(
      rule,
      inputs.song.instrumental,
    ).value;
    const description = inputs.description.trim();
    const lyrics = inputs.lyrics.trim();
    if (rule.description === "required" && !description)
      return "Describe the music. This model needs a description.";
    if (rule.lyrics === "required" && !instrumental && !lyrics)
      return "Write the lyrics. This model sings them.";
    if (!description && (!lyrics || instrumental))
      return instrumental
        ? "Describe the music you want."
        : "Describe the music or write lyrics.";
    return null;
  }
  if (rule.id === "sfx") {
    return inputs.sfx.prompt.trim() ? null : "Describe the sound you want.";
  }
  return inputs.editProblem ?? null;
}

export function buildMusicRunRequest(
  inputs: MusicInputs & {
    options?: Record<string, AudioOptionScalar>;
    seed?: number | null;
  },
): AudioTextRunRequest {
  const { rule } = inputs;
  const base = {
    workflow: "music" as const,
    options: inputs.options,
    seed: inputs.seed ?? null,
  };
  if (rule.id === "song") {
    const instrumental = instrumentalChoice(
      rule,
      inputs.song.instrumental,
    ).value;
    const sendLyrics = lyricsShown(rule, instrumental);
    return {
      ...base,
      text: inputs.description.trim(),
      music: {
        mode: "song",
        lyrics: sendLyrics ? inputs.lyrics : null,
        instrumental: rule.instrumental === "toggle" ? instrumental : false,
        duration_s: musicDurationFor(rule, inputs.song.durationS),
        variations: variationsFor(rule, inputs.song.variations),
      },
    };
  }
  if (rule.id === "sfx") {
    return {
      ...base,
      text: inputs.sfx.prompt.trim(),
      music: {
        mode: "sfx",
        duration_s: musicDurationFor(rule, inputs.sfx.durationS),
        variations: variationsFor(rule, inputs.sfx.variations),
      },
    };
  }
  const edit = inputs.edit;
  const action = edit.action ?? rule.actions?.[0] ?? "repaint";
  const usesRanges = action === "repaint" || action === "inpaint";
  return {
    ...base,
    text: edit.prompt.trim(),
    music: {
      mode: "edit",
      source: editSourceRef(edit.source),
      duration_s: action === "continue" ? musicDurationFor(rule, null) : null,
      edit: {
        action,
        ranges: usesRanges ? edit.ranges : [],
        // Send the value the slider shows; the runtime defaults (1.0) differ.
        strength:
          action === "cover" || action === "restyle"
            ? (edit.strength ?? MUSIC_EDIT_DEFAULT_STRENGTH[action] ?? null)
            : null,
        extend_s: action === "extend" ? edit.extendS : null,
      },
    },
  };
}

function editSourceRef(source: AudioSourceSelection | null) {
  if (!source || source.kind === "voice") return null;
  return sourceRefOf(source);
}
