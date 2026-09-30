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

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
  position?: unknown;
}

const HTML_TAG_NAME = /^\s*<\/?([a-z][a-z0-9-]*)/i;
const INNER_TAG = /<(\/?)([a-z][^\s/>]*)/gi;

function rehypeLiteralUnknownTags(tagNames: string[]) {
  const known = new Set(tagNames);
  return function walk(node: HastNode): void {
    const children = node.children ?? [];
    children.forEach((child, index) => {
      if (child.type !== "raw") {
        walk(child);
        return;
      }
      const tag = HTML_TAG_NAME.exec(child.value ?? "")?.[1];
      if (!tag) {
        return;
      }
      if (known.has(tag.toLowerCase())) {
        child.value = child.value?.replace(INNER_TAG, (match, slash, name) =>
          known.has(name.toLowerCase()) ? match : `&lt;${slash}${name}`,
        );
        return;
      }
      // Streamdown's memoised components only re-render when the node position changes.
      const { position } = child;
      const text = { type: "text", value: child.value, position };
      children[index] =
        node.type === "root"
          ? {
              type: "element",
              tagName: "p",
              properties: {},
              children: [text],
              position,
            }
          : text;
    });
  };
}

/** Keep data images and resolve sandbox paths before URL hardening. */
export function withDataImageSupport(
  allowedTags: Record<string, string[]>,
  beforeHarden: Pluggable[] = [],
): Pluggable[] {
  const sanitize = defaultRehypePlugins.sanitize as [Plugin<[SanitizeSchema]>, SanitizeSchema];
  const [sanitizePlugin, schema] = sanitize;
  // Positional by design: Streamdown itself builds its default pipeline as `Object.values` of this
  // same object, so spreading it in the same order reproduces that pipeline exactly. Naming the keys
  // would pin OUR order instead of theirs.
  const [raw, , harden] = Object.values(defaultRehypePlugins);
  const tagNames = [...(schema.tagNames ?? []), ...Object.keys(allowedTags)];
  return [
    [rehypeLiteralUnknownTags, tagNames],
    raw,
    [
      sanitizePlugin,
      {
        ...schema,
        tagNames,
        attributes: { ...schema.attributes, ...allowedTags },
        protocols: {
          ...schema.protocols,
          // Harden still gates the scheme on `allowDataImages` and only honors `data:image/*`.
          src: [...(schema.protocols?.src ?? []), "data"],
        },
      },
    ],
    ...beforeHarden,
    harden,
  ];
}
