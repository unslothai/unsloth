// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What "Restore these settings" puts back for the build: the precision a saved image or clip ran with, and
// whether its model is the resident one (a different model is named for the user to reload, never loaded).

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

/** The Precision a recipe ran with, or null to leave the select alone. Only a recorded scheme counts: the gallery
 * API sends a missing field as null too, so a null cannot tell a dense run from a record older than the field. */
export function recipeTransformerQuant(recipe: RecipeBuild): TransformerQuantChoice | null {
  const engaged = recipe.transformer_quant;
  if (typeof engaged === "string" && engaged.trim()) {
    const picked = choice(TRANSFORMER_QUANT_CHOICES, engaged);
    return picked === "auto" ? null : picked;
  }
  return null;
}

/** The Text encoder precision a recipe ran with, or null to leave the select alone (null on record is ambiguous, as
 * for Precision). */
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

interface ResidentBuild {
  loaded?: boolean;
  repo_id?: string | null;
  display_repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  gguf_variant?: string | null;
}

function basename(path: string): string {
  const norm = path.replace(/\\/g, "/");
  return norm.slice(norm.lastIndexOf("/") + 1).trim().toLowerCase();
}

/** Whether the resident build runs the recipe's checkpoint: same repo, and for a GGUF recipe the same file (or,
 * where the status names only the quant, the same quant). `variantOf` maps a GGUF filename to its quant label. */
function residentRunsRecipe(
  model: string,
  recipe: RecipeBuild & { gguf_filename?: string | null },
  status: ResidentBuild,
  variantOf?: (filename: string) => string,
): boolean {
  if (!status.loaded || !(sameModel(model, status.repo_id) || sameModel(model, status.display_repo_id))) return false;
  const kind = (recipe.model_kind ?? "").trim().toLowerCase();
  const residentKind = (status.model_kind ?? "").trim().toLowerCase();
  if (kind && residentKind && kind !== residentKind) return false;
  const file = recipe.gguf_filename?.trim();
  if (kind !== "gguf" || !file) return true;
  if (status.gguf_filename?.trim()) return basename(status.gguf_filename) === basename(file);
  if (status.gguf_variant?.trim() && variantOf) {
    return variantOf(basename(file)).toLowerCase() === status.gguf_variant.trim().toLowerCase();
  }
  return true;
}

/** The recipe's model when the resident build does not run its checkpoint (another repo, another GGUF file of the
 * same repo, or nothing loaded), else null. Connection runs and records without a model never ask for a reload. */
export function recipeModelToReload(
  recipe: RecipeBuild & { gguf_filename?: string | null },
  status: ResidentBuild | null | undefined,
  variantOf?: (filename: string) => string,
): string | null {
  const model = recipe.model?.trim();
  if (!model || recipe.model_kind === "connection") return null;
  if (status && residentRunsRecipe(model, recipe, status, variantOf)) return null;
  return model;
}
