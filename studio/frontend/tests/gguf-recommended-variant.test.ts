// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import ts from "typescript";

import {
  recommendedGgufVariant,
  shouldShowGgufRecommendation,
} from "../src/features/hub/lib/gguf-recommendation.ts";

const GIBIBYTE = 1024 ** 3;
const FIT_LIMIT_GB = 8;
const PICKERS_SOURCE_PATH = fileURLToPath(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
);

type TestVariant = {
  quant: string;
  size_bytes: number;
  downloaded?: boolean;
  partial?: boolean;
};

function variant(
  quant: string,
  sizeGb: number,
  overrides: Partial<TestVariant> = {},
): TestVariant {
  return {
    quant,
    size_bytes: sizeGb * GIBIBYTE,
    ...overrides,
  };
}

const fitsUnder = (limitGb: number) => (sizeBytes: number) =>
  sizeBytes <= limitGb * GIBIBYTE;

function effectiveRecommendationInitializer(): ts.Expression | null {
  const source = ts.createSourceFile(
    PICKERS_SOURCE_PATH,
    readFileSync(PICKERS_SOURCE_PATH, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let initializer: ts.Expression | null = null;
  function visit(node: ts.Node): void {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === "effectiveRecommended"
    ) {
      initializer = node.initializer ?? null;
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  return initializer;
}

function containsRecommendationIntegration(node: ts.Node): boolean {
  if (
    ts.isCallExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === "recommendedGgufVariant" &&
    node.arguments.length === 3
  ) {
    const predicate = node.arguments[2];
    if (
      !ts.isArrowFunction(predicate) ||
      predicate.parameters.length !== 1 ||
      !ts.isIdentifier(predicate.parameters[0].name) ||
      !ts.isBinaryExpression(predicate.body) ||
      predicate.body.operatorToken.kind !==
        ts.SyntaxKind.ExclamationEqualsEqualsToken ||
      !ts.isCallExpression(predicate.body.left) ||
      !ts.isIdentifier(predicate.body.left.expression) ||
      predicate.body.left.expression.text !== "getGgufFit" ||
      predicate.body.left.arguments.length !== 1 ||
      !ts.isIdentifier(predicate.body.left.arguments[0]) ||
      !ts.isStringLiteral(predicate.body.right) ||
      predicate.body.right.text !== "oom"
    ) {
      return false;
    }

    const parameterName = predicate.parameters[0].name.text;
    return predicate.body.left.arguments[0].text === parameterName;
  }
  return node.getChildren().some(containsRecommendationIntegration);
}

test("prefers the default variant when it fits", () => {
  const largerFit = variant("Q6_K", 6);
  const defaultFit = variant("Q4_K_M", 4);

  assert.equal(
    recommendedGgufVariant(
      [largerFit, defaultFit],
      defaultFit.quant,
      fitsUnder(FIT_LIMIT_GB),
    ),
    defaultFit,
  );
});

test("uses the largest non-OOM variant when the default is OOM", () => {
  const smallerFit = variant("Q4_K_M", 4);
  const largestFit = variant("Q6_K", 6);
  const defaultOom = variant("Q8_0", 20);

  assert.equal(
    recommendedGgufVariant(
      [smallerFit, defaultOom, largestFit],
      defaultOom.quant,
      fitsUnder(FIT_LIMIT_GB),
    ),
    largestFit,
  );
});

test("uses the largest fitting variant when no default is provided", () => {
  const smallerFit = variant("Q4_K_M", 4);
  const largestFit = variant("Q6_K", 6);

  assert.equal(
    recommendedGgufVariant(
      [largestFit, smallerFit],
      null,
      fitsUnder(FIT_LIMIT_GB),
    ),
    largestFit,
  );
});

test("uses the largest fitting variant when the named default is absent", () => {
  const smallerFit = variant("Q4_K_M", 4);
  const largestFit = variant("Q6_K", 6);

  assert.equal(
    recommendedGgufVariant(
      [largestFit, smallerFit],
      "MISSING",
      fitsUnder(FIT_LIMIT_GB),
    ),
    largestFit,
  );
});

test("keeps a downloaded fitting variant as the recommendation instead of promoting an OOM download", () => {
  const downloadedFit = variant("Q4_K_M", 4, { downloaded: true });
  const unavailableOom = variant("Q8_0", 20);

  assert.equal(
    recommendedGgufVariant(
      [unavailableOom, downloadedFit],
      downloadedFit.quant,
      fitsUnder(FIT_LIMIT_GB),
    ),
    downloadedFit,
  );
});

test("keeps a partial fitting variant as the recommendation instead of promoting an OOM download", () => {
  const partialFit = variant("Q4_K_M", 4, { partial: true });
  const unavailableOom = variant("Q8_0", 20);

  assert.equal(
    recommendedGgufVariant(
      [unavailableOom, partialFit],
      partialFit.quant,
      fitsUnder(FIT_LIMIT_GB),
    ),
    partialFit,
  );
});

test("falls back to the smallest variant when every candidate is OOM", () => {
  const largestOom = variant("Q8_0", 20);
  const middleOom = variant("Q5_K_M", 19);
  const smallestOom = variant("Q6_K", 18);

  assert.equal(
    recommendedGgufVariant(
      [largestOom, smallestOom, middleOom],
      null,
      fitsUnder(1),
    ),
    smallestOom,
  );
});

test("returns null when there are no variants", () => {
  assert.equal(recommendedGgufVariant([], null, fitsUnder(FIT_LIMIT_GB)), null);
});

test("the chat picker delegates its executable recommendation to the shared rule", () => {
  const initializer = effectiveRecommendationInitializer();

  assert.ok(initializer, "effectiveRecommended initializer was not found");
  assert.equal(containsRecommendationIntegration(initializer), true);
});

test("shows the badge only for the matching downloadable recommendation", () => {
  const recommended = variant("Q4_K_M", 4);

  assert.equal(shouldShowGgufRecommendation(recommended, recommended), true);
  assert.equal(
    shouldShowGgufRecommendation(
      { ...recommended, downloaded: true },
      recommended,
    ),
    false,
  );
  assert.equal(
    shouldShowGgufRecommendation(
      { ...recommended, partial: true },
      recommended,
    ),
    false,
  );
  assert.equal(
    shouldShowGgufRecommendation(variant("Q6_K", 6), recommended),
    false,
  );
  assert.equal(shouldShowGgufRecommendation(recommended, null), false);
  assert.equal(shouldShowGgufRecommendation(null, null), false);
});
