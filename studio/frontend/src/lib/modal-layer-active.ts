// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

/**
 * True while a modal Radix layer (dialog, alert dialog) holds the page.
 *
 * Radix marks modalness on the body — `pointer-events: none` for the rest of
 * the document — which is exactly what should shield off-dialog triggers. A
 * trigger that drives its open state in JavaScript after preventDefault
 * bypasses that shield (#9244), so it needs to consult it explicitly. Reading
 * the style covers every modal dialog without each call site enumerating them.
 */
export function modalLayerActive(doc: Document = document): boolean {
  const body = doc.body;
  if (!body) return false;
  return doc.defaultView?.getComputedStyle(body).pointerEvents === "none";
}
