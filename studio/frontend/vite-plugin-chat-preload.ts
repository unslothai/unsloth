// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Preloads the chat chunks before the entry on a start that lands on Chat. A file, not inline: script-src is 'self'.

import { createHash } from "node:crypto";
import type { Plugin } from "vite";

const CHAT_PAGE_MODULE = "/src/features/chat/chat-page.tsx";
const ENTRY_TAG = /<script type="module"[^>]*\ssrc="[^"]+"[^>]*><\/script>/;

type ChunkLike = {
  type: "chunk";
  isEntry: boolean;
  isDynamicEntry: boolean;
  imports: string[];
  moduleIds: string[];
};

function staticClosure(
  bundle: Record<string, { type: string }>,
  roots: string[],
): Set<string> {
  const seen = new Set<string>();
  const stack = [...roots];
  while (stack.length > 0) {
    const name = stack.pop() as string;
    if (seen.has(name)) {
      continue;
    }
    seen.add(name);
    const chunk = bundle[name] as ChunkLike | undefined;
    if (chunk?.type === "chunk") {
      stack.push(...chunk.imports);
    }
  }
  return seen;
}

export function chatPreloadScript(files: string[]): string {
  return `(function () {
  try {
    var path = location.pathname;
    // Mirrors main.tsx: hasAuthToken() and getPostAuthRoute() === "/chat", where the single-user
    // Desktop app goes to Chat even with a password change pending.
    var singleUserDesktop =
      ("__TAURI__" in window ||
        "__TAURI_INTERNALS__" in window ||
        location.protocol === "tauri:") &&
      localStorage.getItem("unsloth.auth-login-mode.v1") !== "multi";
    var landsOnChat =
      path === "/chat" ||
      (path === "/" &&
        Boolean(localStorage.getItem("unsloth_auth_token")) &&
        (singleUserDesktop ||
          localStorage.getItem("unsloth_auth_must_change_password") === null));
    if (!landsOnChat) return;
    var files = ${JSON.stringify(files)};
    for (var i = 0; i < files.length; i++) {
      var link = document.createElement("link");
      link.rel = "modulepreload";
      link.crossOrigin = "";
      link.href = files[i];
      document.head.appendChild(link);
    }
  } catch (error) {}
})();
`;
}

export function chatPagePreload(): Plugin {
  let base = "/";
  return {
    name: "chat-page-preload",
    apply: "build",
    enforce: "post",
    configResolved(config) {
      base = config.base;
    },
    generateBundle(_options, bundle) {
      const html = bundle["index.html"];
      if (!html || html.type !== "asset") {
        return;
      }
      const chunks = Object.entries(bundle).filter(
        (entry): entry is [string, ChunkLike & (typeof bundle)[string]] =>
          entry[1].type === "chunk",
      );
      const page = chunks.find(
        ([, chunk]) =>
          chunk.isDynamicEntry &&
          chunk.moduleIds.some((id) =>
            id.replaceAll("\\", "/").endsWith(CHAT_PAGE_MODULE),
          ),
      );
      if (!page) {
        this.error(
          `no lazy chunk holds ${CHAT_PAGE_MODULE}; Chat would start without its preload`,
        );
      }
      const eager = staticClosure(
        bundle,
        chunks.filter(([, chunk]) => chunk.isEntry).map(([name]) => name),
      );
      const files = [...staticClosure(bundle, [page[0]])]
        .filter((name) => !eager.has(name))
        .sort()
        .map((name) => `${base}${name}`);
      const source = chatPreloadScript(files);
      const hash = createHash("sha256")
        .update(source)
        .digest("hex")
        .slice(0, 8);
      const fileName = `assets/chat-preload-${hash}.js`;
      this.emitFile({ type: "asset", fileName, source });
      const text = String(html.source);
      if (!ENTRY_TAG.test(text)) {
        this.error(
          "index.html has no module entry script to put the Chat preload before",
        );
      }
      html.source = text.replace(
        ENTRY_TAG,
        (tag) => `<script src="${base}${fileName}"></script>\n    ${tag}`,
      );
    },
  };
}
