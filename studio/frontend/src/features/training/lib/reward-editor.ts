// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import yaml from "js-yaml";
import type { RewardRecord } from "../api/rewards-api";
import type { RuleType } from "./reward-summary";

// validate_rule in backend rewards.py stays the only validator; numbers are sent as typed when they parse.
export type DraftRule = Record<string, unknown> & { type: RuleType };

export interface RewardDraft {
  name: string;
  description: string;
  rule: DraftRule;
}

export const NORMALIZERS = [
  "strip",
  "lower",
  "remove_commas",
  "collapse_spaces",
] as const;

const BUNDLED_PRESET: Partial<Record<RuleType, string>> = {
  exact_match: "exact-answer",
  numeric: "numeric-close",
  regex: "strict-xml-format",
};

/** A starting rule for a type: the bundled reward of that type when there is one. */
export function defaultRewardRule(
  type: RuleType,
  library: RewardRecord[],
): DraftRule {
  const preset = library.find(
    (r) => r.name === BUNDLED_PRESET[type] && r.source === "bundled" && r.valid,
  );
  if (preset?.rule) {
    return structuredClone(preset.rule) as DraftRule;
  }
  switch (type) {
    case "exact_match":
      return {
        type,
        compare_to: "answer",
        extract: { between: ["<answer>", "</answer>"] },
        normalize: ["strip"],
        score: { match: 1, miss: 0 },
        missing: 0,
      };
    case "numeric":
      return {
        type,
        compare_to: "answer",
        bands: [{ within: 0, score: 1 }],
        else: 0,
        missing: 0,
      };
    case "regex":
      return {
        type,
        pattern: "",
        mode: "fullmatch",
        score: { match: 1, miss: 0 },
      };
    case "json_schema":
      return {
        type,
        schema: { type: "object", required: [] },
        score: { match: 1, miss: 0 },
      };
    case "length":
      return { type, max_chars: 1000, score: { under: 0, over: -1 } };
  }
}

export function copyRewardName(name: string, library: RewardRecord[]): string {
  const base = `${name.slice(0, 52)}-copy`;
  let next = base;
  for (let n = 2; library.some((r) => r.name === next); n++) {
    next = `${base}-${n}`;
  }
  return next;
}

/** The backend's normalize_reward_name, or null when the name is not valid. */
export function normalizeRewardName(name: string): string | null {
  const normalized = name.normalize("NFKC").trim().toLowerCase();
  return /^[a-z0-9][a-z0-9-]{0,63}$/.test(normalized) ? normalized : null;
}

function numeric(value: unknown): unknown {
  if (typeof value !== "string" || value.trim() === "") {
    return value;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : value;
}

/** REWARD.md text for a draft; the backend validates it on preview and on save. */
export function rewardDraftMarkdown(draft: RewardDraft): string {
  const rule: Record<string, unknown> = { ...draft.rule };
  for (const key of ["missing", "else", "max_chars"]) {
    if (key in rule) {
      rule[key] = numeric(rule[key]);
    }
  }
  if (rule.score && typeof rule.score === "object") {
    rule.score = Object.fromEntries(
      Object.entries(rule.score).map(([k, v]) => [k, numeric(v)]),
    );
  }
  if (Array.isArray(rule.bands)) {
    rule.bands = rule.bands.map((band: Record<string, unknown>) => ({
      within: numeric(band.within),
      score: numeric(band.score),
    }));
  }
  const options = { noRefs: true, lineWidth: -1 };
  const head = {
    name: normalizeRewardName(draft.name) ?? draft.name,
    kind: "rule",
    description: draft.description,
  };
  return `---\n${yaml.dump(head, options)}---\n${yaml.dump(rule, options)}`;
}

export function parseKeyList(text: string): string[] {
  return text
    .split(",")
    .map((key) => key.trim())
    .filter(Boolean);
}

/** The typed text while it still parses to `keys` (so "answer," keeps its comma), else `keys`. */
export function keyListText(text: string, keys: string[]): string {
  const typed = parseKeyList(text);
  return typed.length === keys.length &&
    typed.every((key, i) => key === keys[i])
    ? text
    : keys.join(", ");
}
