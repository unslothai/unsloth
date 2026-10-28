// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a fence body becomes skippable only once it carries its measured height", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.match(
    defer,
    /data-unsloth-fence-windowed=\{lineWindow === null \|\| !measured \? undefined : "true"\}/,
    "a fence that upgrades off screen is first laid out as its plain lines; skipped before its height is written, it collapses, and WebKit, with no scroll anchoring, clamps a following reader's scrollTop and detaches them",
  );
  assert.match(defer, /const NO_WINDOW = \{ window: null, pins: null, measured: false \} as const;\s*const EMPTY_WINDOW = \{ window: EMPTY_LINE_WINDOW, pins: null, measured: false \} as const;/);
  const measure = defer.slice(defer.indexOf("measure.current = () => {"), defer.indexOf("setState({ measured: true, window: next, pins });"));
  assert.ok(measure.indexOf("region.style.setProperty(FENCE_HEIGHT_PROPERTY, declared);") >= 0, "the measure that first marks a fence measured writes its height before it does");
  const autoscroll = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
  assert.match(autoscroll, /const onMutation = \(\): void => \{\s*if \(el\.ownerDocument\.hidden\) \{\s*onLayoutChange\(\);\s*return;\s*\}\s*layoutChanged = true;\s*if \(!parkIfHeld\(\)\) \{\s*extendFollow\(\);\s*\}\s*requestTick\(\);\s*\};/, "a mutation on a shown page never forces layout, following stays the frame loop's job; a hidden page runs no frames, so it pins at once and a reply that ends there is at the bottom when the tab comes back");
  assert.match(autoscroll, /const adjustForContentInsertedAbove = useCallback\(\(deltaPx: number\) => \{\s*adjustImplRef\.current\(deltaPx\);\s*return userDetachedRef\.current;\s*\}, \[\]\);/, "the hold learns whether its correction was written, even when the engine rounded it to nothing");
});

test("only engines without requestIdleCallback shrink the widening step after a long commit", () => {
  const transcript = readSrc("components/assistant-ui/reasoning-transcript.tsx");
  assert.match(transcript, /if \(grew && withoutIdleCallback\)\s*widenCommitMs = Math\.max\(widenCommitMs, performance\.now\(\) - commitStartedAt\);/, "Blink commits a full step inside a frame, so it keeps filling at full speed");
  assert.match(transcript, /widenScale \* Math\.min\(2, WIDEN_COMMIT_TARGET_MS \/ widenCommitMs\)/);
  assert.match(transcript, /const MIN_WIDEN_SCALE = 1 \/ 8;/);
});

test("a desktop zoom shows skipped content until WebKit has measured it at the new zoom", () => {
  const store = readSrc("features/settings/stores/interface-scale-store.ts");
  assert.match(store, /holdContainmentAcrossZoom\(\);\s*try \{\s*await getCurrentWebview\(\)\.setZoom\(zoom\);\s*\} finally \{\s*releaseContainmentAfterZoom\(\);\s*\}/);
  assert.match(store, /const holdsContainmentAcrossZoom = !\(\s*typeof navigator !== "undefined" && navigator\.userAgent\.includes\("Windows"\)\s*\);/, "WebView2 rescales remembered sizes itself");
  assert.match(readSrc("features/settings/lib/interface-scale-runtime.ts"), /const ZOOMING_RELEASE_MS = 300;/);
  const css = readSrc("index.css");
  assert.match(css, /html\[data-interface-zooming\] \.aui-thread-root \[data-slot="reasoning-transcript"\] > \[data-reasoning-chunk\],\s*html\[data-interface-zooming\] \.aui-thread-root \[data-unsloth-fence-windowed\],\s*html\[data-interface-zooming\] \.aui-thread-root \.aui-math-block,\s*html\[data-interface-zooming\] \.aui-thread-root \.aui-math-display \{\s*content-visibility: visible !important;\s*\}/, "WebKit keeps contain-intrinsic-size: auto sizes in the old zoom's units, maths blocks included");
  assert.match(store, /interfaceScaleApplicationQueue = Promise\.resolve\(\);\s*releaseContainmentAfterZoom\(\);\s*resolve\(\);/, "a setZoom that never settles must not leave every skipped block rendered");
});

test("the Windows thread gutter follows the native thin bar, which neither browser nor desktop zoom scales", () => {
  const provider = readSrc("app/provider.tsx");
  assert.match(provider, /if \(!root\.classList\.contains\("client-windows"\)\) return;/);
  assert.match(provider, /if \(CSS\.supports\("selector\(::-webkit-scrollbar\)"\)\) \{[\s\S]*?watchThreadScrollbarGutter\(\{\s*source,\s*measure: measureThinScrollbar,\s*style: rule\.style,\s*\}\)/, "the measurement updates a scoped stylesheet rule before paint");
  assert.equal(readSrc("index.css").includes("--studio-interface-zoom"), false);
  assert.equal(readSrc("features/settings/lib/interface-scale-runtime.ts").includes("--studio-interface-zoom"), false);
});

test("below 1.5x device scale the Windows thread scrolls on the compositor and keeps LCD text", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /@media screen and \(forced-colors: none\) \{\s*:root\.client-windows\[data-low-device-scale\] \.aui-thread-viewport:not\(\.chat-full-view-dock \*\) \{\s*scrollbar-gutter: auto;\s*overflow-y: scroll;\s*border-inline-start: var\(--thread-scrollbar-gutter\) solid transparent;\s*background-color: var\(--background\);\s*\}\s*\}/,
    "Chromium composites a scroller below 1.5x only when its scrolling background is opaque, which it paints that way only with an auto gutter; the always-on bar and the start border keep the stable both-edges geometry. A contrast theme paints that transparent border in its text colour, so forced colours keep the stable gutter",
  );
  const supports = css.lastIndexOf("@supports selector(::-webkit-scrollbar)", css.indexOf("[data-low-device-scale]"));
  assert.ok(supports > css.indexOf(".aui-thread-viewport {\n\t/* Reserve scrollbar space"), "the Chromium-only scrollbar block holds it, so Firefox keeps its own scrolling");
  const provider = readSrc("app/provider.tsx");
  assert.match(provider, /const stopFlag = watchLowDeviceScale\(\{\s*source,\s*interfaceZoom: getAppliedInterfaceZoom,\s*subscribeInterfaceZoom: subscribeAppliedInterfaceZoom,\s*root,\s*\}\);/, "the desktop divides its own page zoom out, since Chromium decides on the display's scale, not the zoomed ratio");
  assert.match(provider, /<AppearanceCustomizationEffect \/>\s*<LowDeviceScaleEffect \/>/);
});

test("grammar work waits for a held pointer and the reader, and without requestIdleCallback for the stream", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.match(defer, /const warmMustWait = \(\): boolean =>\s*inputPointerHeld\(\) \|\| inputQuietIn\(\) > 0\s*\|\| \(typeof \(globalThis as Record<string, unknown>\)\.requestIdleCallback !== "function"\s*&& streamActive\(\)\);/);
  assert.ok(defer.indexOf("if (warmMustWait()) {") < defer.indexOf("grammarsWarmed.add(language);"), "a warm that has to wait does not mark its grammar warmed");
});

test("settled fences go to the worker where the main thread's regex engine is slow", () => {
  const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(markdown, /const slowMainThreadRegex =\s*typeof window !== "undefined"\s*&& typeof window\.requestIdleCallback !== "function";/);
  assert.match(markdown, /if \(!tokenizesOffThread\(body, streaming\)\) \{\s*if \(streaming \|\| !slowMainThreadRegex\) return false;\s*const state = highlightWorkerState\(\);\s*if \(state === "unavailable" \|\| state === "stalled"\) return false;\s*\}\s*return code\.cover\(options\)\.uncovered > MAIN_THREAD_TAIL_CHARS;/, "JavaScriptCore interprets every lookbehind pattern, so WebKit tokenizes settled fences off the main thread, and a fence too long for the main thread goes to the worker, unless the main thread's own cache leaves only a short tail, as it does for the fence that just streamed");
  assert.match(markdown, /const MAIN_THREAD_TAIL_CHARS = 512;/);
  const warm = markdown.slice(markdown.indexOf("const warm = useCallback("), markdown.indexOf("const speculate = useCallback("));
  assert.match(warm, /!settlesOffThread\(body, Boolean\(isIncomplete\), options\)/, "the reach latch's warm routes a settled fence the same way the render does, or a scroll tokenizes on the main thread a fence WebKit sends to the worker");
  assert.equal(markdown.includes("fenceJustStreamed"), false, "routing reads the cache itself, not a guess about which fence streamed last");
  const plugin = readSrc("components/assistant-ui/code-plugin.ts");
  assert.match(plugin, /return defaultJavaScriptRegexConstructor\(pattern, \{\s*target: "ES2018",\s*accuracy: "strict",\s*\}\);\s*\} catch \{\s*return defaultJavaScriptRegexConstructor\(pattern\);\s*\}/, "u-flag patterns build several times faster in JavaScriptCore, and anything ES2018 cannot express keeps today's constructor");
  assert.match(plugin, /const javaScriptCore =\s*typeof navigator !== "undefined"\s*&& navigator\.userAgent\.includes\("AppleWebKit\/"\)\s*&& !\/Chrom\(\?:e\|ium\)\\\/\/\.test\(navigator\.userAgent\);/, "workers have a navigator too, so the worker's highlighter makes the same choice");
  assert.match(plugin, /: \{ forgiving: true \},\s*\);/, "V8 matches the emulated ES2018 patterns about 7% slower, so Blink keeps the engine's own target");
});

test("find in page defers rebuilds during scrolling with a maximum wait", () => {
  const find = readSrc("features/find-in-page/hooks/use-find-in-page.ts");
  assert.match(find, /reindexBy = performance\.now\(\) \+ 4 \* REINDEX_INTERVAL_MS;/);
  assert.match(find, /const flush = \(\) => \{\s*const now = performance\.now\(\);\s*const wait = Math\.min\(\s*REINDEX_INTERVAL_MS - \(now - scrolledAt\),\s*reindexBy - now,\s*\);\s*if \(wait > 0\) \{\s*timerRef\.current = setTimeout\(flush, wait\);\s*return;\s*\}/);
  assert.match(find, /window\.addEventListener\("wheel", noteScroll, \{ capture: true, passive: true \}\);/);
  assert.match(find, /window\.removeEventListener\("wheel", noteScroll, \{ capture: true \}\);/);
  assert.equal(/addEventListener\("scroll", noteScroll/.test(find), false, "autoscroll's own scrolling must not starve a followed stream of rebuilds");
});
