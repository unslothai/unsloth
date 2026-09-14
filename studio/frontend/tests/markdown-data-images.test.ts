// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";

const TAG = "search-image";

test("withDataImageSupport allows data: image sources through sanitize", () => {
  const pipeline = withDataImageSupport({ [TAG]: ["token"] });
  assert.equal(pipeline.length, 3);
  const [, harden] = pipeline[2] as [(...args: never[]) => unknown, Record<string, unknown>];
  assert.equal(harden.allowDataImages, true, "harden stage must be the one that gates data images");
  const [, schema] = pipeline[1] as [unknown, { tagNames: string[]; attributes: Record<string, string[]>; protocols: Record<string, string[]> }];
  assert.ok(schema.protocols.src.includes("data"), "sanitize must let data: through so harden can decide");
  for (const scheme of ["http", "https"]) assert.ok(schema.protocols.src.includes(scheme));
  assert.ok(schema.tagNames.includes(TAG), "caller allowed tags survive the pipeline swap");
  assert.deepEqual(schema.attributes[TAG], ["token"]);
});

test("withDataImageSupport carries Streamdown's own schema widening, not just defaultSchema", () => {
  // The pipeline is derived from `defaultRehypePlugins.sanitize[1]`, which is Streamdown's OWN
  // schema (defaultSchema PLUS its own widenings), not bare hast-util-sanitize defaultSchema.
  // Re-deriving from defaultSchema would silently drop these and read as a no-op change.
  const [, schema] = withDataImageSupport({ [TAG]: ["token"] })[1] as [
    unknown,
    { protocols: Record<string, string[]>; attributes: Record<string, unknown> },
  ];
  assert.ok(schema.protocols.href.includes("tel"), "streamdown's tel: widening must survive");
  assert.ok(
    (schema.attributes.code as unknown[]).includes("metastring"),
    "streamdown's code/metastring widening must survive",
  );
});
