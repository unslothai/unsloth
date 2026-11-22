// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Root, RootContent } from "hast";
import type { Plugin } from "unified";

const MEDIA_TAGS = new Set(["img", "video", "audio", "source", "track"]);
// `/{datasets|spaces}/owner/repo/blob/rev/...` is the HTML file viewer; `resolve` serves the bytes.
const HF_BLOB_PATH_RE = /^(\/(?:datasets\/|spaces\/)?[^/]+\/[^/]+\/)blob\//;
const HF_HOSTS = new Set(["huggingface.co", "www.huggingface.co", "hf.co"]);

export function resolveReadmeUrl(
  url: string,
  baseUrl: string,
  media: boolean,
): string {
  const trimmed = url.trim();
  if (!trimmed || /^(?:#|data:|mailto:|tel:)/i.test(trimmed)) return url;
  let parsed: URL;
  try {
    parsed = new URL(trimmed, baseUrl);
  } catch {
    return url;
  }
  // Other schemes pass through untouched for harden to judge.
  if (!/^https?:$/.test(parsed.protocol)) return url;
  const base = new URL(baseUrl);
  if (
    media &&
    (parsed.origin === base.origin || HF_HOSTS.has(parsed.hostname)) &&
    HF_BLOB_PATH_RE.test(parsed.pathname)
  ) {
    parsed.pathname = parsed.pathname.replace(HF_BLOB_PATH_RE, "$1resolve/");
  }
  return parsed.toString();
}

/**
 * Harden blocks a relative URL unless it starts with `/`, `./` or `../`, so a card's
 * `<img src="assets/plot.png">` showed "[Image blocked]": `urlTransform` resolves it only after
 * harden has run. This resolves against the repo first; harden still decides every URL.
 */
export const rehypeReadmeUrls: Plugin<[{ baseUrl: string }], Root> =
  function rehypeReadmeUrls({ baseUrl }) {
    return (tree) => {
      function walk(node: Root | RootContent): void {
        if (node.type === "element") {
          const media = MEDIA_TAGS.has(node.tagName);
          for (const key of ["src", "href", "poster"]) {
            const value = node.properties[key];
            if (typeof value === "string") {
              node.properties[key] = resolveReadmeUrl(
                value,
                baseUrl,
                media || key === "poster",
              );
            }
          }
        }
        if ("children" in node) node.children.forEach(walk);
      }
      walk(tree);
    };
  };
