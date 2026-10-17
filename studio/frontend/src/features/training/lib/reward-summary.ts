// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rule shapes come from validate_rule in studio/backend/core/training/rewards.py.
export type RuleType =
  | "regex"
  | "exact_match"
  | "numeric"
  | "json_schema"
  | "length";

export interface RewardSummary {
  type: RuleType | null;
  key: string;
  params: Record<string, string>;
  score: string;
  otherwise: string | null;
}

type Rule = Record<string, unknown>;

export function formatScore(n: number): string {
  if (n === 0) {
    return "0";
  }
  const abs = Number(Math.abs(n).toFixed(2));
  const text = Number.isInteger(abs) ? abs.toFixed(1) : String(abs);
  return `${n > 0 ? "+" : "−"}${text}`;
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function scoreOf(rule: Rule, key: string): number {
  const score = rule.score;
  return score && typeof score === "object"
    ? num((score as Record<string, unknown>)[key])
    : 0;
}

function tagOf(rule: Rule): string | null {
  const extract = rule.extract as { between?: unknown } | null | undefined;
  const between = extract?.between;
  return Array.isArray(between) && typeof between[0] === "string"
    ? between[0]
    : null;
}

/** Plain-language summary of a rule reward, as an i18n key plus params. */
export function summarizeRule(rule: Rule | null | undefined): RewardSummary {
  const type = (rule?.type as RuleType | undefined) ?? null;
  const tag = rule ? tagOf(rule) : null;
  const column = typeof rule?.compare_to === "string" ? rule.compare_to : "";
  const target = tag ? "Tag" : "Reply";
  if (!rule || !type) {
    return {
      type: null,
      key: "unknown",
      params: {},
      score: "",
      otherwise: null,
    };
  }
  const withMiss = (
    key: string,
    params: Record<string, string>,
    hit: string,
    miss: string,
  ) => {
    const missValue = scoreOf(rule, miss);
    return {
      type,
      key,
      params,
      score: formatScore(scoreOf(rule, hit)),
      otherwise: missValue === 0 ? null : formatScore(missValue),
    };
  };
  switch (type) {
    case "regex":
      return withMiss(
        rule.mode === "search" ? "regexContains" : "regexWhole",
        {},
        "match",
        "miss",
      );
    case "exact_match":
      return withMiss(
        `exact${target}`,
        { tag: tag ?? "", column },
        "match",
        "miss",
      );
    case "json_schema":
      return withMiss(`json${target}`, { tag: tag ?? "" }, "match", "miss");
    case "length":
      return withMiss(
        "length",
        { max: String(num(rule.max_chars)) },
        "over",
        "under",
      );
    case "numeric": {
      const bands = Array.isArray(rule.bands) ? rule.bands : [];
      const best = Math.max(
        ...bands.map((b) => num((b as Rule).score)),
        Number.NEGATIVE_INFINITY,
      );
      const otherwise = num(rule.else);
      return {
        type,
        key: `numeric${target}`,
        params: { tag: tag ?? "", column },
        score: Number.isFinite(best) ? formatScore(best) : "",
        otherwise: otherwise === 0 ? null : formatScore(otherwise),
      };
    }
    default:
      return {
        type: null,
        key: "unknown",
        params: {},
        score: "",
        otherwise: null,
      };
  }
}

/** XML-style tags a rule reads, so the prompt can be checked for asking for them. */
export function ruleTags(rule: Rule | null | undefined): string[] {
  if (!rule) {
    return [];
  }
  const found = new Set<string>();
  const tag = tagOf(rule);
  if (tag && /^<[A-Za-z_][\w-]*>$/.test(tag)) {
    found.add(tag);
  }
  const sources = [
    rule.pattern,
    (rule.extract as Rule | null | undefined)?.regex,
  ];
  for (const source of sources) {
    if (typeof source === "string") {
      for (const m of source
        .replace(/\\/g, "")
        .matchAll(/<([A-Za-z_][\w-]*)>/g)) {
        found.add(`<${m[1]}>`);
      }
    }
  }
  return [...found];
}

export interface RewardMarkdownHead {
  name: string | null;
  kind: string | null;
  description: string | null;
}

/** Frontmatter fields of a pasted REWARD.md, for a preview before importing. */
export function readRewardHead(markdown: string): RewardMarkdownHead | null {
  const match = markdown.match(/^\s*---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) {
    return null;
  }
  const field = (name: string) => {
    const line = match[1].match(new RegExp(`^${name}:\\s*(.+)$`, "m"));
    return line ? line[1].trim().replace(/^['"]|['"]$/g, "") : null;
  };
  return {
    name: field("name"),
    kind: field("kind"),
    description: field("description"),
  };
}
