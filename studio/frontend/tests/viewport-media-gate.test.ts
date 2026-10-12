// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";

import {
  gateSelector,
  gateViewportMedia,
  viewportBootScript,
  viewportGate,
  viewportMediaGate,
} from "../vite-plugin-viewport-media.ts";
import { readText } from "./helpers/kit.ts";

const SM = "data-mq-min-width-40rem";
const MD = "data-mq-min-width-48rem";
const LG = "data-mq-min-width-64rem";

test("a viewport media block keeps its place, gated on <html> for screens and unchanged for print", () => {
  const css =
    ".a{display:block}@media (min-width:40rem){.sm\\:flex{display:flex}.x .y{color:red}}.b{display:grid}";
  const { css: next, queries } = gateViewportMedia(css);
  assert.deepEqual(queries, [{ attribute: SM, query: "(min-width:40rem)" }]);
  assert.equal(
    next,
    ".a{display:block}" +
      `@media screen{:where(html[${SM}]) .sm\\:flex,.sm\\:flex:where(html[${SM}]){display:flex}` +
      `:where(html[${SM}]) .x .y,.x:where(html[${SM}]) .y{color:red}}` +
      "@media print{@media (min-width:40rem){.sm\\:flex{display:flex}.x .y{color:red}}}" +
      ".b{display:grid}",
  );
});

test("the second form anchors the leftmost compound on <html>, so rules keyed on its classes still match", () => {
  const root = `html[${SM}]`;
  assert.equal(
    gateSelector(".dark .md\\:x", root),
    `:where(${root}) .dark .md\\:x,.dark:where(${root}) .md\\:x`,
  );
  assert.equal(gateSelector(":root", root), `:where(${root}) :root,:root:where(${root})`);
  assert.equal(
    gateSelector(".x>.y+.z~.w", root),
    `:where(${root}) .x>.y+.z~.w,.x:where(${root})>.y+.z~.w`,
  );
});

test("pseudo-elements stay last in their compound, and escaped colons are not pseudo-elements", () => {
  const root = `html[${SM}]`;
  assert.equal(
    gateSelector(".sm\\:after\\:hidden:after", root),
    `:where(${root}) .sm\\:after\\:hidden:after,.sm\\:after\\:hidden:where(${root}):after`,
  );
  assert.equal(
    gateSelector(".a::-webkit-scrollbar-thumb:hover", root),
    `:where(${root}) .a::-webkit-scrollbar-thumb:hover,.a:where(${root})::-webkit-scrollbar-thumb:hover`,
  );
  assert.equal(
    gateSelector("::selection", root),
    `:where(${root}) ::selection,:where(${root})::selection`,
  );
  assert.equal(
    gateSelector(".a:before-ish", root),
    `:where(${root}) .a:before-ish,.a:before-ish:where(${root})`,
  );
});

test("escapes, strings and nested selector lists are not combinators or separators", () => {
  const { css } = gateViewportMedia(
    '@media (min-width:96rem){.\\32 xl\\:max-w-6xl{max-width:72rem}.a:is(.b,.c) .d,[data-x="a, b"]{content:"{}"}}',
  );
  const root = "html[data-mq-min-width-96rem]";
  assert.ok(
    css.startsWith(
      `@media screen{:where(${root}) .\\32 xl\\:max-w-6xl,.\\32 xl\\:max-w-6xl:where(${root}){max-width:72rem}` +
        `:where(${root}) .a:is(.b,.c) .d,.a:is(.b,.c):where(${root}) .d,` +
        `:where(${root}) [data-x="a, b"],[data-x="a, b"]:where(${root}){content:"{}"}}`,
    ),
    css,
  );
});

test("negated, nested and range queries combine into one gate, with one print copy at the top", () => {
  const { css, queries } = gateViewportMedia(
    "@media (min-width:48rem){.md\\:a{order:1}@media not all and (min-width:64rem){.md\\:max-lg\\:b{order:2}}@media (hover:hover){.md\\:hover\\:c:hover{order:3}}@supports (display:grid){.md\\:d{order:4}}}",
  );
  assert.deepEqual(
    queries.map(({ attribute }) => attribute),
    [MD, LG],
  );
  assert.ok(css.includes(`:where(html[${MD}]:not([${LG}])) .md\\:max-lg\\:b`));
  assert.ok(css.includes(`@media (hover:hover){:where(html[${MD}]) .md\\:hover\\:c:hover`));
  assert.ok(css.includes(`@supports (display:grid){:where(html[${MD}]) .md\\:d`));
  assert.equal(css.match(/@media print/g)?.length, 1);
  assert.ok(css.endsWith("@media print{@media (min-width:48rem){.md\\:a{order:1}@media not all and (min-width:64rem){.md\\:max-lg\\:b{order:2}}@media (hover:hover){.md\\:hover\\:c:hover{order:3}}@supports (display:grid){.md\\:d{order:4}}}}"));

  assert.deepEqual(viewportGate("not all and (width>=40rem)"), { query: "(width>=40rem)", present: false });
  assert.deepEqual(
    gateViewportMedia("@media (width<=1023px){.a{order:1}}@media (width>=40rem){.b{order:2}}").queries.map(
      ({ attribute }) => attribute,
    ),
    ["data-mq-width-lte-1023px", "data-mq-width-gte-40rem"],
  );
});

test("selector comments cannot swallow gates or supply selector syntax", () => {
  const root = `html[${SM}]`;
  const { css } = gateViewportMedia(
    "@media (min-width:40rem){.a, /* x, > ::before ( [ */ .b/* ::after, */::before{content:'x'}}",
  );
  assert.ok(css.startsWith(
    `@media screen{:where(${root}) .a,.a:where(${root}),` +
    `:where(${root}) .b/* ::after, */::before,.b/* ::after, */:where(${root})::before{content:'x'}}`,
  ), css);
  assert.equal(
    gateSelector(" /* x */ /* y */ html.dark .b", root),
    `:where(${root}) html.dark .b,html.dark:where(${root}) .b`,
  );
});

test("what a selector gate cannot express is left as it was", () => {
  for (const css of [
    "@media (min-width:40rem){@keyframes k{to{opacity:1}}.a{animation:k 1s}}",
    "@media (min-width:40rem){@scope (.card){.title{order:1}}}",
    "@media screen and (min-width:40rem){.a{order:1}}",
    "@media (min-width:40rem) and (max-width:60rem){.a{order:1}}",
    "@media (hover:hover){.a:hover{order:1}}",
    "@media print{.a{order:1}}",
    "/* (min-width:40rem) */.a{order:1}",
  ]) {
    assert.deepEqual(gateViewportMedia(css), { css, queries: [] });
  }
});

test("a block with a selector list the parser cannot rebuild stays native, and its neighbours are still gated", () => {
  for (const invalid of [".b,/* x */", ".b,", ".a, ,.b", ",.a"]) {
    const native = `@media (min-width:64rem){.c{order:1}${invalid}{order:2}}`;
    const { css, queries } = gateViewportMedia(`@media (min-width:40rem){.a{order:1}}${native}`);
    assert.ok(css.endsWith(`@media print{@media (min-width:40rem){.a{order:1}}}${native}`), css);
    assert.deepEqual(queries, [{ attribute: SM, query: "(min-width:40rem)" }]);
  }
});

test("a stylesheet that cannot be parsed is left ungated with a warning, never a failed build", () => {
  const [gate] = viewportMediaGate();
  const warnings: string[] = [];
  const hook = gate.transform as {
    handler: (this: { warn: (message: string) => void }, code: string, id: string) => unknown;
  };
  assert.equal(
    hook.handler.call({ warn: (message) => warnings.push(message) }, "@media (min-width:40rem){.a{order:1}", "/app/broken.css"),
    null,
  );
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /\/app\/broken\.css.*Unclosed block/);
});

test("with build sourcemaps on, the rewrite returns its own map", () => {
  const [gate] = viewportMediaGate();
  (gate.configResolved as (config: { build: { sourcemap: boolean } }) => void)({ build: { sourcemap: true } });
  const hook = gate.transform as {
    handler: (this: unknown, code: string, id: string) => { code: string; map: string };
  };
  const result = hook.handler.call({}, "@media (min-width:40rem){.a{order:1}}", "/app/src/index.css");
  const map: { sources: string[]; mappings: string } = JSON.parse(result.map);
  assert.ok(result.code.startsWith("@media screen{"));
  assert.ok(map.sources.includes("index.css"));
  assert.notEqual(map.mappings, "");
});

test("the boot script mirrors each query onto <html> and follows its changes", () => {
  const attributes = new Map<string, boolean>();
  const lists = new Map<string, { matches: boolean; listeners: (() => void)[] }>();
  const context = {
    document: {
      documentElement: {
        toggleAttribute: (name: string, force: boolean) => attributes.set(name, force),
      },
    },
    window: {
      addEventListener: () => {},
      matchMedia: (query: string) => {
        const list = { matches: query === "(min-width:40rem)", listeners: [] as (() => void)[] };
        lists.set(query, list);
        return {
          get matches() {
            return list.matches;
          },
          addEventListener: (_type: string, listener: () => void) => list.listeners.push(listener),
        };
      },
    },
  };
  vm.runInNewContext(
    viewportBootScript([
      { attribute: SM, query: "(min-width:40rem)" },
      { attribute: LG, query: "(min-width:64rem)" },
    ]),
    context,
  );
  assert.deepEqual([...attributes], [[SM, true], [LG, false]]);
  const large = lists.get("(min-width:64rem)")!;
  large.matches = true;
  for (const listener of large.listeners) listener();
  assert.equal(attributes.get(LG), true);
});

test("boot script data cannot close an inline script and retains its exact runtime values", () => {
  const values = [
    "(width<=1023px)",
    "(width>=40rem)",
    '</script><script>globalThis.injected=true</script>',
    '</ScRiPt><script>globalThis.injected=true</script>',
    '<!--<script>',
    '"];globalThis.injected=true;//',
    '\\";globalThis.injected=true;//',
    '` ${globalThis.injected=true}',
    "quotes: \" ' \\; literal escapes: \\u003c \\u2028",
    String.fromCharCode(...Array.from({ length: 32 }, (_, i) => i)),
    "line\u2028paragraph\u2029end",
    "surrogates: \ud800 \udfff; emoji: \ud83d\ude00",
  ];
  const queries = values.map((value) => ({ attribute: value, query: value }));
  const script = viewportBootScript(queries);
  assert.doesNotMatch(script, /[<>\u2028\u2029]/u);

  const observedQueries: string[] = [];
  const attributes: [string, boolean][] = [];
  const context = {
    injected: false,
    document: {
      documentElement: {
        toggleAttribute: (name: string, force: boolean) => attributes.push([name, force]),
      },
    },
    window: {
      addEventListener: () => {},
      matchMedia: (query: string) => {
        observedQueries.push(query);
        return { matches: true, addEventListener: () => {} };
      },
    },
  };
  vm.runInNewContext(script, context, { timeout: 1000 });
  assert.equal(context.injected, false);
  assert.deepEqual(observedQueries, values);
  assert.deepEqual(attributes, values.map((value) => [value, true]));
});

test("resize refreshes gates before app listeners even while existing media lists are stale", () => {
  const attributes = new Map<string, boolean>();
  const listeners: (() => void)[] = [];
  const lists: { matches: boolean }[] = [];
  let matches = false;
  vm.runInNewContext(viewportBootScript([{ attribute: SM, query: "(min-width:40rem)" }]), {
    document: { documentElement: { toggleAttribute: (name: string, value: boolean) => attributes.set(name, value) } },
    window: {
      addEventListener: (type: string, listener: () => void) => {
        assert.equal(type, "resize");
        listeners.push(listener);
      },
      matchMedia: () => {
        const list = { matches, addEventListener: () => {} };
        lists.push(list);
        return list;
      },
    },
  });
  const observed: boolean[] = [];
  listeners.push(() => observed.push(attributes.get(SM)!));
  for (const next of [true, false, true]) {
    matches = next;
    listeners.forEach((listener) => listener());
  }
  assert.deepEqual(observed, [true, false, true]);
  assert.equal(lists[0].matches, false);
});

test("the build gates every main-document stylesheet and loads the boot script before the others", () => {
  const [gate, boot] = viewportMediaGate();
  assert.equal(gate.apply, "build");
  assert.equal(gate.enforce, undefined, "before vite:css-post turns the stylesheet into a module");
  const hook = gate.transform as {
    filter: { id: { include: RegExp; exclude: RegExp }; code: string };
    handler: (code: string) => { code: string; map: null } | null;
  };
  assert.ok(hook.filter.id.include.test("/app/src/index.css"));
  assert.ok(hook.filter.id.exclude.test("/app/src/frame.css?inline"));
  assert.ok(!hook.filter.id.exclude.test("/app/src/index.css"));
  assert.equal(hook.handler(".a{order:1}"), null);
  assert.ok(hook.handler("@media (min-width:40rem){.a{order:1}}")?.code.startsWith("@media screen{"));

  assert.equal(boot.enforce, "post", "after vite:build-html has emitted the page");
  const emitted: { name: string; source: string }[] = [];
  const bundle: Record<string, { type: string; fileName: string; source: string }> = {
    "index.html": {
      type: "asset",
      fileName: "index.html",
      source: '<html>\n  <head>\n    <title>x</title>\n    <script src="/crypto-boot.js"></script>\n  </head>\n</html>',
    },
  };
  (boot.generateBundle as (this: unknown, options: unknown, bundle: unknown) => void).call(
    {
      emitFile: (file: { name: string; source: string }) => {
        emitted.push(file);
        return "ref";
      },
      getFileName: () => "assets/viewport-media-abc.js",
    },
    {},
    bundle,
  );
  assert.equal(emitted.length, 1);
  assert.ok(emitted[0].source.includes(`"${SM}"`));
  assert.match(
    bundle["index.html"].source,
    /<title>x<\/title>\n {4}<script src="\/assets\/viewport-media-abc\.js"><\/script>\n {4}<script src="\/crypto-boot\.js">/,
  );
});

test("the build config and the reload snapshot use the gate", () => {
  assert.match(
    readText("../vite.config.ts"),
    /plugins: \[react\(\), tailwindcss\(\), childSelectorOrder\(\), viewportMediaGate\(\),/,
  );
  assert.match(
    readText("../public/reload-snapshot.js"),
    /applyAppearanceAttributes\(shellRoot, snapshot\.appearance\);\s*Array\.prototype\.forEach\.call\(\s*document\.documentElement\.attributes,\s*function \(attribute\) \{\s*if \(attribute\.name\.indexOf\("data-mq-"\) === 0\) \{\s*shellRoot\.setAttribute\(attribute\.name, attribute\.value\);/,
    "the restored shell sits in a shadow tree under its own <html>, which needs the same gates",
  );
});

test("the boot asset honors absolute and relative Vite bases on nested pages", () => {
  for (const [base, expected] of [
    ["/", "/assets/viewport-media-abc.js"],
    ["/studio/", "/studio/assets/viewport-media-abc.js"],
    ["https://cdn.example/studio/", "https://cdn.example/studio/assets/viewport-media-abc.js"],
    ["./", "./../assets/viewport-media-abc.js"],
    ["", "./../assets/viewport-media-abc.js"],
  ]) {
    const [gate, boot] = viewportMediaGate();
    (gate.transform as { handler: (code: string) => unknown }).handler(
      "@media (min-width:40rem){.a{order:1}}",
    );
    (boot.configResolved as (config: { base: string }) => void)({ base });
    const asset = { type: "asset", fileName: "nested/index.html", source: "<head></head>" };
    (boot.generateBundle as (this: unknown, options: unknown, bundle: unknown) => void).call(
      { emitFile: () => "ref", getFileName: () => "assets/viewport-media-abc.js" },
      {},
      { "nested/index.html": asset },
    );
    assert.ok(asset.source.includes(`<script src="${expected}">`), asset.source);
  }
});
