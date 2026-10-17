// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import yaml from "js-yaml";
import type { RewardRecord, RewardPreviewItem } from "../api/rewards-api";
import type { GrpoRewardSelection } from "@/types/training";
import type { RuleType } from "./reward-summary";
import { resolveRlMapping } from "./rl-roles";

export type DraftNumber = number | string;
export type Extraction = { between: [string, string] } | { regex: string };
export interface DraftRule {
  [key: string]: unknown;
  type: RuleType;
  extract?: Extraction | null;
  reference_extract?: Extraction | null;
  compare_to?: string;
  normalize?: string[];
  score?: Record<string, DraftNumber>;
  missing?: DraftNumber;
  pattern?: string;
  mode?: "fullmatch" | "search";
  bands?: { within: DraftNumber; score: DraftNumber }[];
  else?: DraftNumber;
  schema?: { type: string; required: string[] };
  max_chars?: DraftNumber;
}
export interface RewardDraft {
  name: string;
  description: string;
  rule: DraftRule;
}
export interface RewardPreviewGroup {
  rewards: RewardPreviewItem[];
  reference: string | null;
}
export const RULE_TYPES: RuleType[] = [
  "exact_match",
  "numeric",
  "regex",
  "json_schema",
  "length",
];
export const NORMALIZERS = [
  "strip",
  "lower",
  "remove_commas",
  "collapse_spaces",
] as const;
export const SAMPLE_REWARD_COMPLETION =
  "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>";

export function canEditReward(record: RewardRecord): boolean {
  return (
    record.valid &&
    !record.shadowed &&
    record.kind === "rule" &&
    RULE_TYPES.includes(record.rule?.type as RuleType)
  );
}

export function defaultRewardRule(
  type: RuleType,
  library: RewardRecord[],
): DraftRule {
  const presetName = {
    exact_match: "exact-answer",
    numeric: "numeric-close",
    regex: "strict-xml-format",
  }[type as "exact_match" | "numeric" | "regex"];
  const preset = library.find(
    (r) => r.name === presetName && r.source === "bundled" && r.valid,
  );
  if (preset?.rule) return structuredClone(preset.rule) as DraftRule;
  switch (type) {
    case "exact_match":
      return {
        type,
        compare_to: "answer",
        extract: { between: ["<answer>", "</answer>"] },
        reference_extract: { regex: "####\\s*(.+?)\\s*$" },
        normalize: ["strip", "remove_commas", "lower"],
        score: { match: 2, miss: 0 },
        missing: 0,
      };
    case "numeric":
      return {
        type,
        compare_to: "answer",
        extract: null,
        reference_extract: null,
        bands: [{ within: 0, score: 1 }],
        else: 0,
        missing: 0,
      };
    case "regex":
      return {
        type,
        pattern: "",
        mode: "fullmatch",
        score: { match: 0, miss: 0 },
      };
    case "json_schema":
      return {
        type,
        extract: null,
        schema: { type: "object", required: [] },
        score: { match: 0, miss: 0 },
      };
    case "length":
      return { type, max_chars: "", score: { over: 0, under: 0 } };
  }
}

export function copyRewardName(name: string, library: RewardRecord[]): string {
  const base = name.slice(0, 52) + "-copy";
  let next = base;
  let n = 2;
  while (library.some((r) => r.name === next)) next = base + "-" + n++;
  return next;
}

export class RewardDraftError extends Error {
  readonly code:
    | "invalidName"
    | "invalidNumber"
    | "invalidExtraction"
    | "missingCondition"
    | "tooLarge";
  constructor(code: RewardDraftError["code"]) {
    super(code);
    this.code = code;
  }
}

function number(value: unknown): number {
  if (
    (typeof value !== "number" && typeof value !== "string") ||
    String(value).trim() === "" ||
    !Number.isFinite(Number(value))
  ) {
    throw new RewardDraftError("invalidNumber");
  }
  return Number(value);
}

function validateExtraction(value: Extraction | null | undefined): void {
  if (!value) return;
  if ("between" in value ? value.between.some((part) => !part) : !value.regex) {
    throw new RewardDraftError("invalidExtraction");
  }
}

/** Serialize data as YAML; the backend remains the authority for rule validation. */
export function rewardDraftMarkdown(draft: RewardDraft): string {
  const name = draft.name.normalize("NFKC").trim().toLowerCase();
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(name))
    throw new RewardDraftError("invalidName");
  const rule = structuredClone(draft.rule);
  validateExtraction(rule.extract);
  validateExtraction(rule.reference_extract);
  if (rule.score)
    rule.score = Object.fromEntries(
      Object.entries(rule.score).map(([key, value]) => [key, number(value)]),
    );
  if (rule.type === "exact_match" || rule.type === "numeric") {
    if (!rule.compare_to?.trim())
      throw new RewardDraftError("missingCondition");
    rule.missing = number(rule.missing);
  }
  if (rule.type === "numeric") {
    if (!rule.bands?.length) throw new RewardDraftError("missingCondition");
    rule.bands = rule.bands.map((band) => ({
      within: number(band.within),
      score: number(band.score),
    }));
    rule.else = number(rule.else);
  }
  if (rule.type === "regex" && !rule.pattern)
    throw new RewardDraftError("missingCondition");
  if (rule.type === "length") {
    rule.max_chars = number(rule.max_chars);
    if (!Number.isInteger(rule.max_chars))
      throw new RewardDraftError("invalidNumber");
  }
  const options = { noRefs: true, lineWidth: -1 };
  const markdown =
    "---\n" +
    yaml.dump({ name, kind: "rule", description: draft.description }, options) +
    "---\n" +
    yaml.dump(rule, options);
  if (new TextEncoder().encode(markdown).length > 64 * 1024)
    throw new RewardDraftError("tooLarge");
  return markdown;
}

export function rewardReferenceColumn(
  rule: Record<string, unknown> | null | undefined,
): string | null {
  return (rule?.type === "exact_match" || rule?.type === "numeric") &&
    typeof rule.compare_to === "string"
    ? rule.compare_to
    : null;
}

/** Auto-fill text references using the training role mapping. Non-text cells need a manual reference. */
export function rewardReference(
  column: string,
  row: Record<string, unknown> | null,
  mapping: Record<string, string>,
): { value: string; source: string | null } {
  const resolved = resolveRlMapping("grpo", Object.keys(row ?? {}), mapping);
  const source =
    column === "answer"
      ? Object.entries(resolved).find(([, role]) => role === "answer")?.[0]
      : column;
  if (!row || !source || typeof row[source] !== "string") {
    return { value: "", source: null };
  }
  return { value: row[source], source };
}

/** A preview request accepts one reference, so different columns need separate requests. */
export function groupRewardPreviews(
  selections: GrpoRewardSelection[],
  library: RewardRecord[],
  references: Record<string, string | null>,
): RewardPreviewGroup[] {
  const groups = new Map<string | null, RewardPreviewGroup>();
  for (const selection of selections) {
    const record = library.find(
      (r) => r.name === selection.name && !r.shadowed,
    );
    const column = rewardReferenceColumn(record?.rule);
    const group = groups.get(column) ?? {
      rewards: [],
      reference: column ? (references[column] ?? null) : null,
    };
    group.rewards.push(selection);
    groups.set(column, group);
  }
  return [...groups.values()];
}
