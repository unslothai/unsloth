// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
Object.assign(globalThis, { document: { documentElement: { lang: "" } } });
const navigatorState: { language?: string; languages?: string[] } = {};
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: navigatorState,
});

const { initializeLocale, LOCALE_STORAGE_KEY, AUTO_LOCALE } = await import(
  "../src/i18n/locale-store.ts"
);
const { LOCALES } = await import("../src/i18n/messages.ts");
const { localeCatalogPreload, localePreloadScript } = await import(
  "../vite-plugin-locale-preload.ts"
);

const LAZY = Object.keys(LOCALES).filter((locale) => locale !== "en");
const FILES = Object.fromEntries(
  LAZY.map((locale) => [locale, [`/assets/${locale}-hash.js`]]),
);

type Browser = {
  languages?: string[];
  language?: string;
  stored?: string | null;
  storageThrows?: boolean;
};

function appLocale({ languages, language, stored }: Browser): string {
  navigatorState.languages = languages;
  navigatorState.language = language;
  store.clear();
  if (stored !== undefined && stored !== null) {
    store.set(LOCALE_STORAGE_KEY, stored);
  }
  return initializeLocale({ loadMessages: () => undefined }) as string;
}

function preloads(
  { languages, language, stored = null, storageThrows = false }: Browser,
  script = localePreloadScript(FILES),
): { rel?: string; href?: string; crossOrigin?: string }[] {
  const appended: { rel?: string; href?: string; crossOrigin?: string }[] = [];
  vm.runInNewContext(script, {
    navigator: { languages, language },
    localStorage: {
      getItem: (key: string) => {
        if (storageThrows) throw new Error("SecurityError");
        return key === LOCALE_STORAGE_KEY ? stored : null;
      },
    },
    document: {
      createElement: () => ({}),
      head: {
        appendChild: (link: (typeof appended)[number]) => {
          appended.push(link);
        },
      },
    },
  });
  return appended;
}

function expectedHrefs(browser: Browser): string[] {
  return FILES[appLocale(browser)] ?? [];
}

const TAGS = [
  ...Object.keys(LOCALES),
  ...Object.keys(LOCALES).map((locale) => locale.toUpperCase()),
  ...Object.keys(LOCALES).map((locale) => `${locale.split("-")[0]}-XX`),
  "zh",
  "zh-TW",
  "zh-HK",
  "zh-MO",
  "zh-SG",
  "zh-Hant",
  "zh-Hans",
  "zh-Hans-HK",
  "zh-Hant-CN",
  "es-MX",
  "es-AR",
  "es-US",
  "es-ES",
  "es-GQ",
  "es-Latn-UY",
  "es-419",
  "pt",
  "pt-PT",
  "pt-br",
  "en-US",
  "en-GB",
  "fr-CA",
  "de-CH",
  "ja-JP",
  "nb-NO",
  "xx",
  "",
];

test("the boot script preloads the catalog the app loads for every browser language", () => {
  for (const tag of TAGS) {
    for (const browser of [
      { languages: [tag] },
      { languages: [tag, "ja"] },
      { languages: ["xx", tag] },
      { languages: [], language: tag },
    ]) {
      const hrefs = preloads(browser).map((link) => link.href);
      assert.deepEqual(hrefs, expectedHrefs(browser), JSON.stringify(browser));
    }
  }
  assert.deepEqual(preloads({}), []);
  assert.equal(appLocale({}), "en");
});

test("a stored choice wins over the browser, and only a known code counts as one", () => {
  for (const stored of [
    ...Object.keys(LOCALES),
    AUTO_LOCALE,
    "pt-br",
    "JA",
    "garbage",
    "",
    "toString",
    "__proto__",
  ]) {
    for (const languages of [["ja-JP"], ["en-US"], ["de-DE", "fr"]]) {
      const browser = { languages, stored };
      const hrefs = preloads(browser).map((link) => link.href);
      assert.deepEqual(hrefs, expectedHrefs(browser), JSON.stringify(browser));
    }
  }
  assert.deepEqual(
    preloads({ languages: ["en-US"], stored: "ja" }).map((link) => link.href),
    ["/assets/ja-hash.js"],
  );
});

test("English needs no catalog, so it gets no preload", () => {
  assert.deepEqual(preloads({ languages: ["en-US"] }), []);
  assert.deepEqual(preloads({ languages: ["ja"], stored: "en" }), []);
  assert.deepEqual(preloads({ languages: ["xx-YY"] }), []);
});

test("unreadable storage falls back to the browser languages, like the app", () => {
  const hrefs = preloads({ languages: ["de-DE"], storageThrows: true });
  assert.deepEqual(
    hrefs.map((link) => link.href),
    ["/assets/de-hash.js"],
  );
});

test("each hint is a module preload in the same CORS mode as the app's import()", () => {
  const [link] = preloads({ languages: ["fr-FR"] });
  assert.deepEqual(link, {
    rel: "modulepreload",
    crossOrigin: "",
    href: "/assets/fr-hash.js",
  });
});

test("every lazy locale has a catalog file the plugin can find", () => {
  const files = readdirSync(new URL("../src/i18n/locales/", import.meta.url))
    .filter((name) => name.endsWith(".ts"))
    .map((name) => name.slice(0, -3).toLowerCase());
  const loaders = readFileSync(
    new URL("../src/i18n/messages.ts", import.meta.url),
    "utf8",
  );
  for (const locale of LAZY) {
    assert.ok(files.includes(locale.toLowerCase()), locale);
    const loader = loaders.match(
      new RegExp(`"?${locale}"?: \\(\\) => import\\("\\./locales/([^"]+)"\\)`),
    );
    assert.equal(loader?.[1].toLowerCase(), locale.toLowerCase(), locale);
  }
});

type FakeChunk = {
  type: "chunk";
  isEntry: boolean;
  isDynamicEntry: boolean;
  imports: string[];
  moduleIds: string[];
};

function fakeBundle(skip: string | null = null): Record<string, unknown> {
  const bundle: Record<string, unknown> = {
    "index.html": {
      type: "asset",
      source: `<head>
    <script src="/theme-boot.js"></script>
    <script type="module" crossorigin src="/assets/index-abc.js"></script>
    <link rel="modulepreload" crossorigin href="/assets/react-abc.js">
  </head>`,
    },
    "assets/index-abc.js": {
      type: "chunk",
      isEntry: true,
      isDynamicEntry: false,
      imports: ["assets/react-abc.js"],
      moduleIds: ["/repo/src/main.tsx"],
    } satisfies FakeChunk,
    "assets/react-abc.js": {
      type: "chunk",
      isEntry: false,
      isDynamicEntry: false,
      imports: [],
      moduleIds: ["/repo/node_modules/react/index.js"],
    } satisfies FakeChunk,
    "assets/shared-abc.js": {
      type: "chunk",
      isEntry: false,
      isDynamicEntry: false,
      imports: [],
      moduleIds: ["/repo/src/i18n/shared.ts"],
    } satisfies FakeChunk,
  };
  for (const locale of LAZY) {
    if (locale === skip) continue;
    const file = locale === "pt-BR" ? "pt-br" : locale;
    bundle[`assets/${file}-abc.js`] = {
      type: "chunk",
      isEntry: false,
      isDynamicEntry: true,
      imports:
        locale === "ja" ? ["assets/shared-abc.js", "assets/react-abc.js"] : [],
      moduleIds: [`C:\\repo\\src\\i18n\\locales\\${file}.ts`],
    } satisfies FakeChunk;
  }
  return bundle;
}

function runPlugin(bundle: Record<string, unknown>): {
  fileName: string;
  source: string;
}[] {
  const plugin = localeCatalogPreload();
  (plugin.configResolved as unknown as (config: { base: string }) => void)({
    base: "/",
  });
  const emitted: { fileName: string; source: string }[] = [];
  (
    plugin.generateBundle as unknown as (
      this: unknown,
      options: unknown,
      bundle: Record<string, unknown>,
    ) => void
  ).call(
    {
      emitFile: (file: { fileName: string; source: string }) => {
        emitted.push(file);
      },
      error: (message: string) => {
        throw new Error(message);
      },
    },
    {},
    bundle,
  );
  return emitted;
}

test("the build emits a hashed classic script before the module entry", () => {
  const bundle = fakeBundle();
  const emitted = runPlugin(bundle);
  assert.equal(emitted.length, 1);
  const [{ fileName, source }] = emitted;
  assert.match(fileName, /^assets\/locale-preload-[0-9a-f]{8}\.js$/);
  const html = String((bundle["index.html"] as { source: string }).source);
  assert.ok(
    html.includes(
      `<script src="/${fileName}"></script>\n    <script type="module" crossorigin src="/assets/index-abc.js"></script>`,
    ),
    html,
  );
  assert.equal(html.match(/locale-preload-/g)?.length, 1);
  assert.equal(
    runPlugin(fakeBundle())[0].fileName,
    fileName,
    "the name follows the content",
  );

  new vm.Script(source);
  const hrefs = (languages: string[]) =>
    preloads({ languages }, source).map((link) => link.href);
  assert.deepEqual(hrefs(["pt-BR"]), ["/assets/pt-br-abc.js"]);
  assert.deepEqual(hrefs(["zh-HK"]), ["/assets/zh-TW-abc.js"]);
  // A catalog's own imports ride along unless the entry already loads them.
  assert.deepEqual(hrefs(["ja-JP"]), [
    "/assets/ja-abc.js",
    "/assets/shared-abc.js",
  ]);
  assert.deepEqual(hrefs(["en-US"]), []);
});

test("the build fails rather than ship a locale without its preload", () => {
  assert.throws(() => runPlugin(fakeBundle("de")), /"de" catalog/);
});

test("the vite build registers the locale preload", () => {
  const config = readFileSync(
    new URL("../vite.config.ts", import.meta.url),
    "utf8",
  );
  assert.match(config, /localeCatalogPreload\(\)/);
});
