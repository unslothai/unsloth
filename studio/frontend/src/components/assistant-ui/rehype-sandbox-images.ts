// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Root, RootContent } from "hast";
import type { Plugin } from "unified";

import { safeMarkdownUrl } from "../../lib/safe-markdown-url.ts";
import { markdownSandboxImageSrc } from "./sandbox-files.ts";

type SandboxScope = Parameters<typeof markdownSandboxImageSrc>[1];

export const rehypeSandboxImages: Plugin<[SandboxScope], Root> = function rehypeSandboxImages(scope) {
  return (tree) => {
    function walk(node: Root | RootContent): void {
      if (node.type === "element" && node.tagName === "img") {
        const src = node.properties.src;
        if (typeof src === "string") {
          // Check URLs before harden changes relative paths or removes origins.
          const safe = safeMarkdownUrl(src, "src", node);
          node.properties.src = safe
            ? (markdownSandboxImageSrc(safe, scope) ?? safe)
            : undefined;
        }
      }
      if ("children" in node) node.children.forEach(walk);
    }
    walk(tree);
  };
};
