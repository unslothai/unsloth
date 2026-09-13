// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function eligibleImageFields(schema: unknown): string[] {
  if (!schema || typeof schema !== "object") {
    return [];
  }
  const value = schema as Record<string, unknown>;
  if (
    value.type !== "object" ||
    !value.properties ||
    typeof value.properties !== "object"
  ) {
    return [];
  }
  const unsupported = [
    "$ref",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependencies",
    "dependentSchemas",
    "dependentRequired",
    "patternProperties",
    "unevaluatedProperties",
  ];
  if (unsupported.some((key) => key in value)) {
    return [];
  }
  return Object.entries(value.properties).flatMap(([name, field]) => {
    if (!field || typeof field !== "object") {
      return [];
    }
    const property = field as Record<string, unknown>;
    return property.type === "string" &&
      ![...unsupported, "const", "enum"].some((key) => key in property)
      ? [name]
      : [];
  });
}
