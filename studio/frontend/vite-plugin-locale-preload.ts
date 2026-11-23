// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Preload the non-English catalog before the entry (a file, not inline: script-src is 'self').
// A wrong or failed hint costs one wasted fetch: the app still imports the same URL.

import { createHash } from "node:crypto";
import type { Plugin } from "vite";
import { LOCALES, matchLocaleTag } from "./src/i18n/locale-tags";

const LOCALE_MODULE = /\/src\/i18n\/locales\/([^/]+)\.ts$/;
const ENTRY_TAG = /<script type="module"[^>]*\ssrc="[^"]+"[^>]*><\/script>/;
// Mirror locale-store.ts: LOCALE_STORAGE_KEY and AUTO_LOCALE (tests/locale-preload.test.ts).
const STORAGE_KEY = "unsloth_locale";
const AUTO = "auto";

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
): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  const stack = [...roots];
  while (stack.length > 0) {
    const name = stack.pop() as string;
    if (seen.has(name)) continue;
    seen.add(name);
    order.push(name);
    const chunk = bundle[name] as ChunkLike | undefined;
    if (chunk?.type === "chunk") stack.push(...chunk.imports);
  }
  return order;
}

// Ships matchLocaleTag's own source so the boot script and app cannot drift. crossOrigin ""
// matches Vite's own modulepreload credentials mode.
export function localePreloadScript(files: Record<string, string[]>): string {
  return `(function () {
  try {
    var locales = ${JSON.stringify(Object.keys(LOCALES))};
    var files = ${JSON.stringify(files)};
    var matchLocaleTag = ${matchLocaleTag.toString()};
    // Mirrors initializeLocale() in locale-store.ts: a stored known code wins, else detect.
    var stored = null;
    try {
      stored = localStorage.getItem(${JSON.stringify(STORAGE_KEY)});
    } catch (error) {}
    var locale = null;
    if (stored !== ${JSON.stringify(AUTO)} && locales.indexOf(stored) !== -1) {
      locale = stored;
    } else {
      var tags =
        navigator.languages && navigator.languages.length
          ? navigator.languages
          : navigator.language
            ? [navigator.language]
            : [];
      for (var i = 0; i < tags.length && locale === null; i++) {
        locale = matchLocaleTag(tags[i], locales);
      }
    }
    var hrefs = locale !== null && Object.prototype.hasOwnProperty.call(files, locale) ? files[locale] : [];
    for (var j = 0; j < hrefs.length; j++) {
      var link = document.createElement("link");
      link.rel = "modulepreload";
      link.crossOrigin = "";
      link.href = hrefs[j];
      document.head.appendChild(link);
    }
  } catch (error) {}
})();
`;
}

export function localeCatalogPreload(): Plugin {
  let base = "/";
  return {
    name: "locale-catalog-preload",
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
      const eager = new Set(
        staticClosure(
          bundle,
          chunks.filter(([, chunk]) => chunk.isEntry).map(([name]) => name),
        ),
      );
      const catalogs = new Map<string, string>();
      for (const [name, chunk] of chunks) {
        if (!chunk.isDynamicEntry) continue;
        for (const id of chunk.moduleIds) {
          const match = id.replaceAll("\\", "/").match(LOCALE_MODULE);
          if (match) catalogs.set(match[1].toLowerCase(), name);
        }
      }
      const files: Record<string, string[]> = {};
      for (const locale of Object.keys(LOCALES)) {
        if (locale === "en") continue;
        const name = catalogs.get(locale.toLowerCase());
        if (!name) {
          this.error(
            `no lazy chunk holds the "${locale}" catalog; its first render would start without a preload`,
          );
        }
        files[locale] = staticClosure(bundle, [name])
          .filter((file) => !eager.has(file))
          .map((file) => `${base}${file}`);
      }
      const source = localePreloadScript(files);
      const hash = createHash("sha256")
        .update(source)
        .digest("hex")
        .slice(0, 8);
      const fileName = `assets/locale-preload-${hash}.js`;
      this.emitFile({ type: "asset", fileName, source });
      const text = String(html.source);
      if (!ENTRY_TAG.test(text)) {
        this.error(
          "index.html has no module entry script to put the locale preload before",
        );
      }
      html.source = text.replace(
        ENTRY_TAG,
        (tag) => `<script src="${base}${fileName}"></script>\n    ${tag}`,
      );
    },
  };
}
