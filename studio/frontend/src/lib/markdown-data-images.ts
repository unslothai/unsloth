// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Streamdown sanitizes with its default schema before hardening, and that schema allows only
 * http(s) image sources: a `data:image/...` src is stripped before the harden stage, which does
 * allow data images (`allowDataImages: true`), so the message renders "[Image blocked: …]" instead
 * of the image. Streamdown extends its own schema with caller `allowedTags` only when it receives
 * its default pipeline (identity check), so callers that pass one must carry that merge themselves.
 *
 * Sanitize also drops every tag outside the schema, which erases `<placeholder>` and `Vec<T>`, so
 * those become text before `raw`. Sanitize and harden still decide every element.
 */
import type { Element, Root, RootContent } from "hast";
import type { Pluggable, Plugin } from "unified";
import { defaultRehypePlugins } from "streamdown";

interface SanitizeSchema {
  tagNames?: string[];
  attributes?: Record<string, string[]>;
  protocols?: Record<string, string[]>;
}

const HTML_TAG_NAME = /^\s*<\/?([a-z][^\s/<>]*)/i;
const INNER_TAG = /<(\/?)([a-z][^\s/<>]*)/gi;
// Formatting tags outside the schema that documents use as markup. Where a pipeline opts in, they are
// left for sanitize to unwrap so their text shows: always in a raw HTML block, which closes them
// implicitly, and in prose only when matched, so "the <small> tag" stays text like any unknown tag.
const UNWRAPPED_TAGS = new Set([
  "abbr",
  "big",
  "center",
  "cite",
  "dfn",
  "figcaption",
  "figure",
  "font",
  "mark",
  "nobr",
  "small",
  "u",
]);

/** `child:offset` of every matched formatting-tag opener and closer among the raw children. */
function matchedFormattingTags(children: RootContent[]): Set<string> {
  const matched = new Set<string>();
  const open = new Map<string, string[]>();
  children.forEach((child, index) => {
    if (child.type !== "raw") return;
    for (const tag of child.value.matchAll(INNER_TAG)) {
      const name = tag[2].toLowerCase();
      if (!UNWRAPPED_TAGS.has(name)) continue;
      const key = `${index}:${tag.index}`;
      const openers = open.get(name) ?? [];
      open.set(name, openers);
      if (!tag[1]) {
        openers.push(key);
        continue;
      }
      const opener = openers.pop();
      if (opener) matched.add(opener).add(key);
    }
  });
  return matched;
}

interface LiteralTagOptions {
  tagNames: string[];
  unwrapFormatting: boolean;
}

// One options object: Streamdown keys its processor cache on a plugin's first option only.
const rehypeLiteralUnknownTags: Plugin<[LiteralTagOptions], Root> =
  function rehypeLiteralUnknownTags({ tagNames, unwrapFormatting }) {
    const schemaTags = new Set(tagNames);
    return (tree) => {
      function walk(node: Root | Element): void {
        const children: RootContent[] = node.children;
        const matched = unwrapFormatting
          ? matchedFormattingTags(children)
          : new Set<string>();
        const isSchemaTag = (name: string) =>
          schemaTags.has(name.toLowerCase());
        children.forEach((child, index) => {
          if (child.type === "element") {
            walk(child);
            return;
          }
          if (child.type !== "raw") return;
          const tag = HTML_TAG_NAME.exec(child.value)?.[1];
          if (!tag) return;
          if (
            isSchemaTag(tag) ||
            (unwrapFormatting &&
              node.type === "root" &&
              UNWRAPPED_TAGS.has(tag.toLowerCase())) ||
            matched.has(`${index}:${child.value.indexOf("<")}`)
          ) {
            child.value = child.value.replace(
              INNER_TAG,
              (match, slash, name) =>
                isSchemaTag(name) ||
                (unwrapFormatting && UNWRAPPED_TAGS.has(name.toLowerCase()))
                  ? match
                  : `&lt;${slash}${name}`,
            );
            return;
          }
          // Streamdown's memoised components only re-render when the node position changes.
          const { position } = child;
          const text = { type: "text" as const, value: child.value, position };
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
      }
      walk(tree);
    };
  };

function literalTagPipeline(
  allowedTags: Record<string, string[]>,
  srcProtocols: string[],
  beforeHarden: Pluggable[],
  unwrapFormatting: boolean,
): Pluggable[] {
  const sanitize = defaultRehypePlugins.sanitize as [
    Plugin<[SanitizeSchema]>,
    SanitizeSchema,
  ];
  const [sanitizePlugin, schema] = sanitize;
  // Positional by design: Streamdown itself builds its default pipeline as `Object.values` of this
  // same object, so spreading it in the same order reproduces that pipeline exactly. Naming the keys
  // would pin OUR order instead of theirs.
  const [raw, , harden] = Object.values(defaultRehypePlugins);
  const tagNames = [...(schema.tagNames ?? []), ...Object.keys(allowedTags)];
  return [
    [rehypeLiteralUnknownTags, { tagNames, unwrapFormatting }],
    raw,
    [
      sanitizePlugin,
      {
        ...schema,
        tagNames,
        attributes: { ...schema.attributes, ...allowedTags },
        protocols: {
          ...schema.protocols,
          src: [...(schema.protocols?.src ?? []), ...srcProtocols],
        },
      },
    ],
    ...beforeHarden,
    harden,
  ];
}

/** Streamdown's default pipeline plus `allowedTags` for documents: tags outside the schema stay text,
 * formatting tags used as markup are unwrapped. */
export function withLiteralUnknownTags(
  allowedTags: Record<string, string[]> = {},
): Pluggable[] {
  return literalTagPipeline(allowedTags, [], [], true);
}

/** Keep data images, keep tags outside the schema as text, resolve sandbox paths before URL hardening. */
export function withDataImageSupport(
  allowedTags: Record<string, string[]>,
  beforeHarden: Pluggable[] = [],
): Pluggable[] {
  // Harden still gates the scheme on `allowDataImages` and only honors `data:image/*`.
  // Replies mention tags in prose more often than they format with them, so none are unwrapped.
  return literalTagPipeline(allowedTags, ["data"], beforeHarden, false);
}
