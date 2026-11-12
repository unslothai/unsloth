// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1].trim(), body: m[2] }));
const GLASS_SIDEBAR = [
	'html[data-glass][data-chat-wallpaper-workspace-ready] [data-slot="sidebar-inner"]',
	'html[data-glass="macos"] [data-slot="sidebar-inner"]',
	// Transparent: the sidebar is its own colour at the glass strength, so see-through too.
	'html[data-glass-clear] [data-slot="sidebar-inner"]',
];

test("the sidebar footer fade paints nothing over a glass sidebar", () => {
	const sidebar = readFileSync(new URL("../src/components/app-sidebar.tsx", import.meta.url), "utf8");
	const [, slot, classes] = /ref=\{attachFade\}\s*data-slot="([^"]+)"[\s\S]*?className=\{cn\(\s*(?:\/\/[^\n]*\n\s*)*"([^"]+)"/.exec(sidebar) ?? [];
	assert.match(classes ?? "", /from-\[var\(--sidebar-surface\)\]/);
	const fade = `[data-slot="${slot}"]`;
	for (const scope of GLASS_SIDEBAR) {
		const hidden = rules.find((r) => r.selector.split(",").some((s) => s.trim() === `${scope} ${fade}`));
		assert.ok(hidden && /display:\s*none/.test(hidden.body), scope);
		const masked = rules.find((r) =>
			r.selector.split(",").some((s) => s.trim().startsWith(`${scope}:has(${fade}[data-visible="true"])`) && s.includes(".sidebar-scroll-fade")),
		);
		assert.ok(masked && /(^|[^-])mask-image:[^;]*transparent calc\(100% - 8px\)/.test(masked.body), scope);
	}
	// Glass over an opaque sidebar (no image on that route) keeps the fade.
	for (const rule of rules.filter((r) => r.selector.includes(fade) && /display:\s*none/.test(r.body))) {
		for (const s of rule.selector.split(",")) assert.ok(GLASS_SIDEBAR.some((scope) => s.trim().startsWith(scope)), s);
	}
});

test("the account button hover keeps the glass tint", () => {
	const opaque = rules.filter((r) => r.selector.includes(".app-user-trigger") && /background(-color)?:/.test(r.body) && /var\(--card\)/.test(r.body));
	assert.ok(opaque.length > 0);
	for (const rule of opaque) {
		assert.match(rule.selector, /:not\(\[data-glass\]\[data-chat-wallpaper-workspace-ready\], \[data-glass="macos"\]\)/);
	}
});
