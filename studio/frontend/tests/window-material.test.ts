import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	MATERIALS,
	sanitizeMaterial,
	sanitizeStrength,
} from "../src/features/chat-background/model.ts";

test("glass material repairs malformed persisted data", () => {
	assert.equal(sanitizeMaterial(undefined), "solid");
	assert.equal(sanitizeMaterial("glass"), "solid");
	assert.equal(sanitizeMaterial({}), "solid");
	for (const material of MATERIALS) assert.equal(sanitizeMaterial(material), material);
});

test("every glass material has its stylesheet tokens", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	for (const material of MATERIALS.filter((m) => m !== "solid")) {
		assert.match(
			css,
			new RegExp(`html\\[data-glass="${material}"\\] \\{ --glass-tint-base: \\d+%; --glass-blur-base: \\d+px; --glass-saturate: \\d+%; \\}`),
			material,
		);
	}
	// The page only clears its fills once the shell has put the system material under it.
	assert.match(css, /html\[data-glass="macos"\],\nhtml\[data-glass="macos"\] body \{\n  background: transparent;/);
});

test("glass keeps the unprefixed backdrop-filter that Chromium reads", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const glass = css.slice(css.indexOf("/* Glass materials (html[data-glass])"));
	assert.doesNotMatch(glass, /\n\s*backdrop-filter:[^;]+;\n\s*-webkit-backdrop-filter:/);
});

test("glass menus only apply on top of an applied material", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const rules = [...css.matchAll(/^([^\n{]*\[data-glass-menus\][^\n{]*)\{/gm)].map((m) => m[1]);
	assert.ok(rules.length > 0);
	// Without data-glass the tint variables are unset and color-mix would drop the menu fill.
	for (const rule of rules) assert.match(rule, /^html\[data-glass(\^?="[a-z0-9-]+")?\]\[data-glass-menus\]/);
});

test("turning the wallpaper fade off drops the band behind the composer", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(
		css,
		/\.chat-wallpaper-image\[data-fade="false"\]\) \.thread-bottom-fade \{\n  display: none;/,
	);
});

test("glass strength repairs malformed persisted data", () => {
	assert.equal(sanitizeStrength(undefined), 50);
	assert.equal(sanitizeStrength(Number.NaN), 50);
	assert.equal(sanitizeStrength(-4), 0);
	assert.equal(sanitizeStrength(240), 100);
	assert.equal(sanitizeStrength(33.6), 34);
});

test("composer outline off also drops the glass ring and bevel", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] [^{]*\[data-outline="false"\]\) \.unsloth-composer-surface,[^{]*\{\n  --composer-shadow: none;/);
	assert.match(css, /\[data-outline="false"\]\) \.unsloth-composer-surface::before,[^{]*\{\n  content: none;/);
});

test("glass adds no outer shadows, only inset light", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const glass = css.slice(css.indexOf("/* Glass materials (html[data-glass])")).replace(/\/\*[\s\S]*?\*\//g, "");
	const split = (value: string) => {
		const parts: string[] = [];
		let depth = 0;
		let start = 0;
		for (let i = 0; i < value.length; i++) {
			if (value[i] === "(") depth++;
			else if (value[i] === ")") depth--;
			else if (value[i] === "," && depth === 0) {
				parts.push(value.slice(start, i));
				start = i + 1;
			}
		}
		return [...parts, value.slice(start)].map((p) => p.trim());
	};
	const decls = [...glass.matchAll(/(box-shadow|--composer-shadow|--glass-[a-z-]*rim):\s*([^;]+);/g)];
	assert.ok(decls.length > 5);
	for (const [, prop, value] of decls) {
		for (const part of split(value.replace(/\s*!important$/, ""))) {
			assert.ok(/^(inset\b|var\(|none$)/.test(part), `${prop}: ${part}`);
		}
	}
});

test("glass panes thicken over images that would wash out their text", async () => {
	const { glassReadability, toneOf } = await import("../src/features/chat-background/model.ts");
	const light = toneOf(Array.from({ length: 100 }, (_, i) => 0.6 + i * 0.003));
	const dark = toneOf(Array.from({ length: 100 }, () => 0.02));
	// Dark mode: light text over a light image needs a nearly opaque pane; a dark image needs none.
	assert.ok(glassReadability(light, true).floor > 0.75);
	assert.equal(glassReadability(dark, true).floor, 0);
	// Light mode: dark text over a dark image needs a lighter pane; a light image shades it instead.
	assert.ok(glassReadability(dark, false).floor > 0.5);
	assert.equal(glassReadability(light, false).floor, 0);
	assert.ok(glassReadability(light, false).shade > 0);
	assert.equal(glassReadability(toneOf([1, 1, 1]), false).floor, 0);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /var\(--glass-sidebar-fill\) min\(96%, calc\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) \* var\(--glass-opacity, 1\)\)\)/);
	assert.match(css, /var\(--glass-panel-fill\) min\(96%, calc\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) \* var\(--glass-opacity, 1\)\)\)/);
});
