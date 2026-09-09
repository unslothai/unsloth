// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Streamdown sanitizes with its default schema before hardening, and that schema allows only
 * http(s) image sources: a `data:image/...` src is stripped before the harden stage, which does
 * allow data images (`allowDataImages: true`), so the message renders "[Image blocked: …]" instead
 * of the image. Streamdown extends its own schema with caller `allowedTags` only when it receives
 * its default pipeline (identity check), so callers that pass one must carry that merge themselves.
 */
import type { Pluggable, Plugin } from "unified";
import { defaultRehypePlugins } from "streamdown";

interface SanitizeSchema {
  tagNames?: string[];
  attributes?: Record<string, string[]>;
  protocols?: Record<string, string[]>;
}

/** Streamdown's default raw/sanitize/harden pipeline with `data` added to image source protocols. */
export function withDataImageSupport(allowedTags: Record<string, string[]>): Pluggable[] {
  const sanitize = defaultRehypePlugins.sanitize as [Plugin<[SanitizeSchema]>, SanitizeSchema];
  const [sanitizePlugin, schema] = sanitize;
  // Positional by design: Streamdown itself builds its default pipeline as `Object.values` of this
  // same object, so spreading it in the same order reproduces that pipeline exactly. Naming the keys
  // would pin OUR order instead of theirs.
  const [raw, , harden] = Object.values(defaultRehypePlugins);
  return [
    raw,
    [
      sanitizePlugin,
      {
        ...schema,
        tagNames: [...(schema.tagNames ?? []), ...Object.keys(allowedTags)],
        attributes: { ...schema.attributes, ...allowedTags },
        protocols: {
          ...schema.protocols,
          // Harden still gates the scheme on `allowDataImages` and only honors `data:image/*`.
          src: [...(schema.protocols?.src ?? []), "data"],
        },
      },
    ],
    harden,
  ];
}
