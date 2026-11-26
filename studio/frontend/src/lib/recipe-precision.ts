// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What "Restore these settings" puts back for the build: the precision a saved image or clip ran with, and
// whether its model is the resident one (a different model is offered as a reload, never loaded silently).

import { type ResolvedControl, resolvedSelectValue } from "./resolved-precision.ts";

export const TRANSFORMER_QUANT_CHOICES = ["auto", "none", "int8", "fp8", "nvfp4", "mxfp8"] as const;
export type TransformerQuantChoice = (typeof TRANSFORMER_QUANT_CHOICES)[number];

export const TEXT_ENCODER_QUANT_CHOICES = ["auto", "none", "fp8", "fp8_dynamic", "int8", "nvfp4"] as const;
export type TextEncoderQuantChoice = (typeof TEXT_ENCODER_QUANT_CHOICES)[number];

function choice<T extends string>(options: readonly T[], value: string): T | null {
  const v = value.trim().toLowerCase();
  // The engaged value spells "no quant" as "off"; the option is "none".
  return options.find((o) => o === v || (o === "none" && v === "off")) ?? null;
}

/** The Precision option a loaded build's resolved record seeds, or null to leave the select alone. */
export function seededTransformerQuant(resolved: ResolvedControl | null | undefined): TransformerQuantChoice | null {
  return resolvedSelectValue(resolved, (v) => choice(TRANSFORMER_QUANT_CHOICES, v));
}

/** The Text encoder precision option a loaded build's record seeds; "off" is Dense, NOT Default. */
export function seededTextEncoderQuant(resolved: ResolvedControl | null | undefined): TextEncoderQuantChoice | null {
  return resolvedSelectValue(resolved, (v) => choice(TEXT_ENCODER_QUANT_CHOICES, v));
}

interface RecipeBuild {
  model?: string | null;
  model_kind?: string | null;
  transformer_quant?: string | null;
  text_encoder_quant?: string | null;
}

/** The Precision a recipe ran with: its engaged scheme, or Off for a dense pipeline that ran none. Null (leave the
 * select alone) for a GGUF or connection run and for records older than the field. */
export function recipeTransformerQuant(recipe: RecipeBuild): TransformerQuantChoice | null {
  const engaged = recipe.transformer_quant;
  if (typeof engaged === "string" && engaged.trim()) {
    const picked = choice(TRANSFORMER_QUANT_CHOICES, engaged);
    return picked === "auto" ? null : picked;
  }
  return engaged === null && recipe.model_kind === "pipeline" ? "none" : null;
}

/** The Text encoder precision a recipe ran with; null leaves the select alone (no engaged value on record). */
export function recipeTextEncoderQuant(recipe: RecipeBuild): TextEncoderQuantChoice | null {
  const engaged = recipe.text_encoder_quant;
  if (typeof engaged !== "string" || !engaged.trim()) return null;
  const picked = choice(TEXT_ENCODER_QUANT_CHOICES, engaged);
  return picked === "auto" ? null : picked;
}

function sameModel(a: string | null | undefined, b: string | null | undefined): boolean {
  const x = (a ?? "").trim().toLowerCase();
  return x !== "" && x === (b ?? "").trim().toLowerCase();
}

/** The recipe's model when it is not the resident one (or nothing is loaded), else null. Connection runs and
 * records without a model never ask for a reload. */
export function recipeModelToReload(
  recipe: RecipeBuild,
  status: { loaded?: boolean; repo_id?: string | null; display_repo_id?: string | null } | null | undefined,
): string | null {
  const model = recipe.model?.trim();
  if (!model || recipe.model_kind === "connection") return null;
  if (status?.loaded && (sameModel(model, status.repo_id) || sameModel(model, status.display_repo_id))) return null;
  return model;
}

/** Whether a model id is a Hub repo the picker can load by id alone (not a local path). */
export function isHubRepoId(model: string): boolean {
  return /^[\w.-]+\/[\w.-]+$/.test(model.trim());
}
