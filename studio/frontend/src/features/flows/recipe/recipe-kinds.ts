// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What each recipe block is called and how it looks on a Flows node card: Recipe Studio's icons and names,
// tones from the card's tone set. Plain data, loaded with the canvas.

import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
  DocumentAttachmentIcon,
  DocumentCodeIcon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  GithubIcon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Plug01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { CATEGORY_TONES } from "../editor/node-tones";
import type { RecipeColumn } from "./recipe-columns";

export const RECIPE_TONES = {
  sampler: CATEGORY_TONES.audio,
  person: CATEGORY_TONES.logic,
  llm: CATEGORY_TONES.output,
  expression: CATEGORY_TONES.text,
  validator: CATEGORY_TONES.video,
  seed: CATEGORY_TONES.data,
  provider: CATEGORY_TONES.logic,
  model: CATEGORY_TONES.transform,
  tools: CATEGORY_TONES.table,
} as const;

/** A block Recipe Studio can add: its kind and type in `blocks/definitions.ts`, and how Flows shows it. */
export interface RecipeBlock {
  kind: "sampler" | "llm" | "expression" | "validator" | "seed";
  type: string;
  label: string;
  icon: IconSvgElement;
  tone: string;
}

const block = (
  kind: RecipeBlock["kind"],
  type: string,
  label: string,
  icon: IconSvgElement,
  tone: string,
): RecipeBlock => ({ kind, type, label, icon, tone });

/** Every column Recipe Studio can add, in its sheet's order. */
export const COLUMN_BLOCKS: RecipeBlock[] = [
  block("llm", "text", "AI text", PencilEdit02Icon, RECIPE_TONES.llm),
  block("llm", "structured", "AI structured data", CodeIcon, RECIPE_TONES.llm),
  block("llm", "code", "AI code", CodeSimpleIcon, RECIPE_TONES.llm),
  block("llm", "judge", "AI scorer", BalanceScaleIcon, RECIPE_TONES.llm),
  block("sampler", "category", "Category", Tag01Icon, RECIPE_TONES.sampler),
  block("sampler", "subcategory", "Subcategory", TagsIcon, RECIPE_TONES.sampler),
  block("sampler", "uniform", "Random number", EqualSignIcon, RECIPE_TONES.sampler),
  block("sampler", "gaussian", "Bell-curve number", Parabola02Icon, RECIPE_TONES.sampler),
  block("sampler", "bernoulli", "Yes/no value", EqualSignIcon, RECIPE_TONES.sampler),
  block("sampler", "datetime", "Date and time", Clock01Icon, RECIPE_TONES.sampler),
  block("sampler", "timedelta", "Time offset", Clock01Icon, RECIPE_TONES.sampler),
  block("sampler", "uuid", "Unique ID", FingerPrintIcon, RECIPE_TONES.sampler),
  block("sampler", "person", "Synthetic person", UserAccountIcon, RECIPE_TONES.person),
  block("expression", "expression", "Formula", FunctionIcon, RECIPE_TONES.expression),
  block("validator", "validator_python", "Python check", Shield02Icon, RECIPE_TONES.validator),
  block("validator", "validator_sql", "SQL check", Shield02Icon, RECIPE_TONES.validator),
  block("validator", "validator_oxc", "JS/TS check", Shield02Icon, RECIPE_TONES.validator),
  block("validator", "validator_json", "JSON check", Shield02Icon, RECIPE_TONES.validator),
  block("validator", "validator_markdown", "Markdown check", Shield02Icon, RECIPE_TONES.validator),
];

export const SEED_BLOCKS: RecipeBlock[] = [
  block("seed", "seed_hf", "Hugging Face dataset", Plant01Icon, RECIPE_TONES.seed),
  block("seed", "seed_local", "CSV or JSON file", DocumentCodeIcon, RECIPE_TONES.seed),
  block("seed", "seed_unstructured", "Document file", DocumentAttachmentIcon, RECIPE_TONES.seed),
  block("seed", "seed_github", "GitHub repositories", GithubIcon, RECIPE_TONES.seed),
];

/** Recipe-level blocks; Recipe Studio files them under its AI sheet. */
export const SETUP_BLOCKS: RecipeBlock[] = [
  block("llm", "model_provider", "Provider connection", Shield02Icon, RECIPE_TONES.provider),
  block("llm", "model_config", "Model preset", Plant01Icon, RECIPE_TONES.model),
  block("llm", "tool_config", "Tool access", Plug01Icon, RECIPE_TONES.tools),
];

export const SEED_TYPE_BLOCK: Record<string, string> = {
  hf: "seed_hf",
  local: "seed_local",
  unstructured: "seed_unstructured",
  // biome-ignore lint/style/useNamingConvention: api schema
  github_repo: "seed_github",
};

/** Icon and tone of a seed, provider, model preset or tool profile card. */
export function setupLook(
  kind: string,
  seedType?: string,
): { icon: IconSvgElement; tone: string } {
  if (kind === "seed") {
    const seed = SEED_BLOCKS.find((b) => b.type === SEED_TYPE_BLOCK[seedType ?? ""]);
    return { icon: (seed ?? SEED_BLOCKS[0]).icon, tone: RECIPE_TONES.seed };
  }
  const setup = SETUP_BLOCKS.find((b) => b.type === kind) ?? SETUP_BLOCKS[0];
  return { icon: setup.icon, tone: setup.tone };
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

/** Which validator a validation column is, from the callable marker or code language Recipe Studio writes. */
export function validatorBlock(column: RecipeColumn): string {
  const params = record(column.validator_params);
  const fn = text(params.validation_function);
  if (fn.startsWith("unsloth_json_validator")) return "validator_json";
  if (fn.startsWith("unsloth_markdown_validator")) return "validator_markdown";
  if (fn.startsWith("unsloth_oxc_validator")) return "validator_oxc";
  const lang = text(params.code_lang) || text(column.code_lang);
  return lang.startsWith("sql") ? "validator_sql" : "validator_python";
}

/** The block a column was made from. */
export function columnBlock(column: RecipeColumn): RecipeBlock {
  const type = text(column.column_type);
  let key = "";
  if (type === "sampler") {
    const sampler = text(column.sampler_type);
    key = sampler === "person_from_faker" ? "person" : sampler;
  } else if (type.startsWith("llm-")) key = type.slice(4);
  else if (type === "expression") key = "expression";
  else if (type === "validation") key = validatorBlock(column);
  return (
    COLUMN_BLOCKS.find((b) => b.type === key) ??
    block("sampler", key, key || "Column", DiceFaces03Icon, RECIPE_TONES.sampler)
  );
}

export interface ColumnFacts {
  /** short values shown as chips: category values, a number range */
  chips: string[];
  fields: { label: string; value: string }[];
}

function range(low: unknown, high: unknown, unit = ""): string[] {
  const show = (v: unknown) => (v === undefined || v === null ? "" : String(v).trim());
  if (!show(low) && !show(high)) return [];
  return [`${show(low) || "?"} – ${show(high) || "?"}${unit ? ` ${unit}` : ""}`];
}

/** The two or three things worth seeing on a column's card. */
export function columnFacts(column: RecipeColumn): ColumnFacts {
  const type = text(column.column_type);
  const params = record(column.params);
  const chips: string[] = [];
  const fields: { label: string; value: string }[] = [];
  if (type === "sampler") {
    const sampler = text(column.sampler_type);
    if (sampler === "category" && Array.isArray(params.values))
      chips.push(...params.values.map(String));
    else if (sampler === "subcategory") {
      if (text(params.category))
        fields.push({ label: "From", value: text(params.category) });
      chips.push(
        ...new Set(Object.values(record(params.values)).flat().map(String)),
      );
    } else if (sampler === "uniform") chips.push(...range(params.low, params.high));
    else if (sampler === "gaussian")
      chips.push(`mean ${params.mean ?? "?"}`, `sd ${params.stddev ?? "?"}`);
    else if (sampler === "bernoulli") chips.push(`p ${params.p ?? "?"}`);
    else if (sampler === "datetime") chips.push(...range(params.start, params.end));
    else if (sampler === "timedelta") {
      if (text(params.reference_column_name))
        fields.push({ label: "From", value: text(params.reference_column_name) });
      chips.push(...range(params.dt_min, params.dt_max, text(params.unit)));
    } else if (sampler === "person" || sampler === "person_from_faker") {
      chips.push(text(params.locale) || "any locale");
      if (text(params.city)) chips.push(text(params.city));
    }
  } else if (type.startsWith("llm-")) {
    if (text(column.model_alias))
      fields.push({ label: "Model", value: text(column.model_alias) });
    if (type === "llm-code" && text(column.code_lang))
      fields.push({ label: "Language", value: text(column.code_lang) });
    if (text(column.tool_alias))
      fields.push({ label: "Tools", value: text(column.tool_alias) });
    if (type === "llm-judge" && Array.isArray(column.scores))
      for (const score of column.scores)
        if (text(record(score).name)) chips.push(text(record(score).name));
  } else if (type === "expression") {
    if (text(column.expr))
      fields.push({ label: "Formula", value: text(column.expr) });
  } else if (type === "validation") {
    const targets = Array.isArray(column.target_columns)
      ? column.target_columns.map(String)
      : [];
    if (targets.length)
      fields.push({ label: "Checks", value: targets.join(", ") });
  }
  return { chips: chips.slice(0, 6), fields: fields.slice(0, 3) };
}
