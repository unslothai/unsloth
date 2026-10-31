// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe JSON in and out of Flows; the backend builds and checks the payloads (`/import/recipe`, `/export/recipe`).

/** The recipe in pasted text, or why it is not one. */
export function parseRecipeJson(
  text: string,
): { payload: Record<string, unknown> } | { error: string } {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return { error: "That is not valid JSON." };
  }
  if (!value || typeof value !== "object" || Array.isArray(value))
    return { error: "A recipe is a JSON object." };
  const root = value as Record<string, unknown>;
  if (root.format === "unsloth-flow")
    return { error: "That is a flow; use Import instead." };
  const recipe =
    root.recipe && typeof root.recipe === "object"
      ? (root.recipe as Record<string, unknown>)
      : root;
  if (!Array.isArray(recipe.columns))
    return { error: "A recipe needs a columns list." };
  return { payload: root };
}

/** `text-to-sql.recipe.json` for a group labelled "Text to SQL". */
export function recipeFileName(label: string | undefined): string {
  const slug = (label ?? "")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
  return `${slug || "recipe"}.recipe.json`;
}
