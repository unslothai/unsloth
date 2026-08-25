// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of kit.ts on purpose: 24 test files import that, and only the handful
// asserting JSX wiring should pay for loading the TypeScript compiler.

import ts from "typescript";

/** The opening tag of `node`, for both `<x>` and `<x />`. */
export const openingTag = (node: ts.Node): ts.JsxOpeningLikeElement | null => {
  if (ts.isJsxSelfClosingElement(node)) return node;
  if (ts.isJsxElement(node)) return node.openingElement;
  return null;
};
