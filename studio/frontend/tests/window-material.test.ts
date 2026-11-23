import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	MATERIALS,
	LENS_SCALE,
	inScope,
	LENS_CLEAR_HOSTS,
	LENS_HOSTS,
	glassStrengthVars,
	lensDrawn,
	sanitizeScope,
	sanitizeLens,
	sanitizeMaterial,
	sanitizeStrength,
	sanitizeStrengths,
} from "../src/features/chat-background/model.ts";

test("glass material repairs malformed persisted data", () => {
	assert.equal(sanitizeMaterial(undefined), "solid");
	assert.equal(sanitizeMaterial("glass"), "solid");
	assert.equal(sanitizeMaterial({}), "solid");
	for (const material of MATERIALS) assert.equal(sanitizeMaterial(material), material);
});

test("the glass Show on setting keeps old records on every page and maps routes like the background", () => {
	assert.equal(sanitizeScope(undefined, "all"), "all");
	assert.equal(sanitizeScope("pages", "all"), "all");
	assert.equal(sanitizeScope("chats", "all"), "chats");
	assert.deepEqual(
		(["newChat", "chat", "page"] as const).map((route) => [inScope("newChat", route), inScope("chats", route), inScope("all", route)]),
		[[true, true, true], [false, true, true], [false, false, true]],
	);
	const material = readFileSync(new URL("../src/features/chat-background/material.ts", import.meta.url), "utf8");
	assert.match(material, /scope: sanitizeScope\(stored\?\.scope, "all"\)/);
	// Out of scope the page draws solid: no glass attribute, no glass menus, no lens or plain backdrop.
	assert.match(material, /const applied = shown === "solid" \|\| !here \? null : shown;/);
	assert.match(material, /menus && material !== "solid" && here,/);
	for (const file of ["lens.tsx", "mount.tsx", "background.tsx"]) {
		const source = readFileSync(new URL(`../src/features/chat-background/${file}`, import.meta.url), "utf8");
		assert.match(source, /useAppliedMaterial\(\)/, file);
	}
	// Pages without the image still get the plain glass canvas; only macOS draws its own.
	const mount = readFileSync(new URL("../src/features/chat-background/mount.tsx", import.meta.url), "utf8");
	assert.match(mount, /return backdrop && ready && material !== "solid" && material !== "macos" \? \(/);
	const root = readFileSync(new URL("../src/app/routes/__root.tsx", import.meta.url), "utf8");
	assert.match(root, /<WindowMaterialMount route=\{hideNavbar \? undefined : isChatRoute \? "chat" : "page"\} \/>/);
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
	assert.match(css, /html\[data-glass="macos"\],\nhtml\[data-glass="macos"\] body \{\n  background: transparent;/);
});

test("glass keeps the unprefixed backdrop-filter that Chromium reads", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const glass = css.slice(css.indexOf("/* Glass materials (html[data-glass])"));
	assert.doesNotMatch(glass, /\n\s*backdrop-filter:[^;]+;\n\s*-webkit-backdrop-filter:/);
});

test("glass menus only apply on top of an applied material", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const rules = [...css.matchAll(/^([^\n{]*\[data-glass-menus\][^\n{]*)\{/gm)].map((m) => m[1].trim());
	assert.ok(rules.length > 0);
	for (const rule of rules) assert.match(rule, /^html(?:\[data-glass(?:\^?="[a-z0-9-]+")?\]|\.tauri\.client-windows\[data-glass-clear\])\[data-glass-menus\]/);
});

test("turning the wallpaper fade off drops the band behind the composer", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(
		css,
		/\[data-wallpaper-frame-flat\] \.thread-bottom-fade \{\n  display: none;/,
	);
});

test("the thread fades out under the see-through header of a background or macOS glass", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(
		css,
		/\[data-wallpaper-ready\]\[data-wallpaper-ready\] \.aui-thread-viewport:not\(\.chat-full-view-dock \*\),\nhtml\[data-glass="macos"\] \.aui-thread-viewport:not\(\.chat-full-view-dock \*\) \{\n  mask-image:\n    linear-gradient\(to bottom, transparent var\(--thread-header-offset, -1\.5rem\), #000 calc\(var\(--thread-header-offset, -1\.5rem\) \+ 1\.5rem\)\),/,
	);
});

test("the header fade covers both thread edges, unmasking only a classic scrollbar's strip", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(
		css,
		/\+ 1\.5rem\)\),\n    linear-gradient\(#000, #000\);\n  mask-size: 100% 100%, min\(var\(--thread-scrollbar-gutter, 10px\), var\(--classic-scrollbar-width, 0px\)\) 100%;\n  mask-position: 0 0, right;/,
	);
});

test("pinned Run settings rows frost a glass menu instead of painting its fill twice", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(
		css,
		/html\[data-glass\]\[data-glass-menus\] \.composer-run-settings :is\(\[data-config-header-row\], \[data-pinned-footer\]\) \{\n  background: none;\n  -webkit-backdrop-filter: none;\n  backdrop-filter: none;\n\}/,
	);
	assert.match(
		css,
		/html\[data-glass\]\[data-glass-menus\] \.composer-run-settings :is\(\[data-config-header-row\], \[data-pinned-footer\]\)::before \{[^}]*background: none;[^}]*backdrop-filter: blur\(16px\);/,
	);
	assert.match(
		css,
		/\[data-glass-motion\] \[data-glass-moving\] \.sticky,\nhtml:is\(\[data-glass\],\[data-glass-clear\]\)\[data-glass-motion\] \[data-glass-moving\] \.composer-run-settings :is\(\[data-config-header-row\], \[data-pinned-footer\]\)::before,/,
	);
	assert.match(
		css,
		/\[data-glass-motion\] \[data-glass-moving\] \.composer-run-settings :is\(\[data-config-header-row\], \[data-pinned-footer\]\)::before \{\n  background: color-mix\(in srgb, var\(--glass-page-popover\) 96%, transparent\);/,
	);
	assert.match(
		css,
		/\[data-glass-clear\]\[data-glass-menus\] [^{]*:is\(\.sticky, \[data-sticky-bar\], \.composer-run-settings \[data-config-header-row\]\) \{\n  --background: var\(--glass-page-bg\);/,
	);
});

test("glass strength repairs malformed persisted data", () => {
	assert.equal(sanitizeStrength(undefined), 50);
	assert.equal(sanitizeStrength(Number.NaN), 50);
	assert.equal(sanitizeStrength(-4), 0);
	assert.equal(sanitizeStrength(240), 100);
	assert.equal(sanitizeStrength(33.6), 34);
	assert.deepEqual(sanitizeStrengths(50), { light: 40, dark: 15 });
	assert.deepEqual(sanitizeStrengths({ light: 70, dark: "x" }), { light: 70, dark: 15 });
});

test("composer outline off also drops the glass ring and bevel", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] [^{]*\[data-wallpaper-outline-off\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel\),[^{]*\{\n  --composer-shadow: none;/);
	assert.match(css, /\[data-wallpaper-outline-off\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel\)::before,[^{]*\{\n  content: none;/);
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

test("the pinned Windows glass frame rounds the corner with either an image or a plain backdrop", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const titlebarRules = css.split("\n").filter((line) =>
		line.startsWith("html[data-chat-wallpaper-ready]") && line.includes('[data-slot="window-titlebar'),
	);
	assert.ok(titlebarRules.length > 0);
	for (const selector of titlebarRules) {
		assert.ok(selector.includes("[data-glass-clear]"), selector);
	}
	const glassFrame = 'html:is([data-glass], [data-glass-clear])[data-chat-wallpaper-workspace-ready]';
	assert.ok(css.includes(glassFrame + ' [data-slot="sidebar-container"][data-windows-glass-frame]::before'));
	assert.ok(css.includes(glassFrame + ' [data-windows-glass-frame] > [data-slot="sidebar-inner"] {\n  position: relative;\n  z-index: 1;\n  background: transparent;'));
});

test("other pages under glass read over one tinted pane with the sidebar's floor", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\][^{]*> \[data-slot="sidebar-inset"\] \{\n  --glass-page-wash: color-mix\(in srgb, var\(--glass-page-bg\) min\(96%, calc\(\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) \+ 6%\)/);
	const pane = /html\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\][^{]*> \[data-slot="sidebar-inset"\] \{([^}]*)\}/.exec(css)?.[1] ?? "";
	assert.doesNotMatch(pane, /backdrop-filter/);
	assert.ok(pane.includes("background-color: var(--glass-page-wash)"));
	assert.ok(css.includes("inset: var(--studio-custom-titlebar-height) 0 0 anchor(--windows-page-sidebar right, 0px)"));
	assert.doesNotMatch(css, /\[data-wallpaper-route="page"\][^{]*\.chat-wallpaper-artwork \{/);
});

test("glass lens modes repair malformed data and scale the right way", () => {
	assert.equal(sanitizeLens("fisheye"), "off");
	assert.equal(sanitizeLens(undefined), "off");
	for (const mode of ["off", "magnify", "shrink"]) assert.equal(sanitizeLens(mode), mode);
	assert.equal(LENS_SCALE.off, 1);
	assert.ok(LENS_SCALE.magnify > 1 && LENS_SCALE.shrink < 1);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /\.glass-lens \{\n  position: absolute;\n  inset: 0;\n  z-index: -1;/);
	assert.match(css, /\[data-wallpaper-outline-off\] \.glass-lens \.glass-lens-rim,/);
});

test("a glass lens set to off draws no wallpaper copies", () => {
	for (const material of MATERIALS) assert.equal(lensDrawn(material, "off"), false);
	for (const lens of ["magnify", "shrink"] as const) {
		assert.equal(lensDrawn("solid", lens), false);
		for (const material of MATERIALS.filter((m) => m !== "solid"))
			assert.equal(lensDrawn(material, lens), true);
	}
});

test("glass panes thicken over images that would wash out their text", async () => {
	const { glassReadability, toneOf } = await import("../src/features/chat-background/model.ts");
	const light = toneOf(Array.from({ length: 100 }, (_, i) => 0.6 + i * 0.003));
	const dark = toneOf(Array.from({ length: 100 }, () => 0.02));
	assert.ok(glassReadability(light, true).floor > 0.75);
	assert.equal(glassReadability(dark, true).floor, 0);
	assert.ok(glassReadability(dark, false).floor > 0.5);
	assert.ok(glassReadability(light, false).floor < 0.2);
	assert.ok(glassReadability(dark, false).floor > 0.7);
	assert.ok(glassReadability(light, false).shade > 0);
	assert.equal(glassReadability(toneOf([1, 1, 1]), false).floor, 0);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /var\(--glass-sidebar-fill\) min\(96%, calc\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) \* var\(--glass-opacity, 1\)\)\)/);
	assert.match(css, /var\(--glass-panel-fill\) min\(96%, calc\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) \* var\(--glass-opacity, 1\)\)\)/);
});

test("resetting local preferences also resets the glass theme", () => {
	const tab = readFileSync(new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url), "utf8");
	const material = readFileSync(new URL("../src/features/chat-background/material.ts", import.meta.url), "utf8");
	const key = /MATERIAL_STORAGE_KEY = "([^"]+)"/.exec(material)?.[1];
	assert.ok(key);
	assert.ok(tab.includes(`"${key}",`));
});

test("the lens refreshes its tint when the composer's fill changes", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /opacity,\n\s*fill\.backgroundColor,\n\s*fill\.backgroundImage,\n\s*blur,\n\s*\]\.join\(\)/);
});

test("glass off writes no strength or tone to the page", () => {
	const material = readFileSync(new URL("../src/features/chat-background/material.ts", import.meta.url), "utf8");
	const background = readFileSync(new URL("../src/features/chat-background/background.tsx", import.meta.url), "utf8");
	assert.match(material, /if \(material === "solid" \|\| strength === 50\) \{\n\s*root\.style\.removeProperty\("--glass-strength"\);/);
	assert.match(background, /if \(!\(frameUrl && glass\)\) return;\n\s*let cancelled = false;\n\s*measureTone\(frameUrl\)/);
	assert.match(background, /if \(!\(appWallpaper && tone && glass\)\) return;/);
});

test("the glass preference survives an account switch, and sign-in waits for the server before a personal image", async () => {
	const { ACCOUNT_CHROME_KEYS } = await import("../src/lib/account-transition.ts");
	const material = readFileSync(new URL("../src/features/chat-background/material.ts", import.meta.url), "utf8");
	const key = /MATERIAL_STORAGE_KEY = "([^"]+)"/.exec(material)?.[1];
	assert.ok(key && ACCOUNT_CHROME_KEYS.has(key));
	for (const page of ["login-page.tsx", "change-password-page.tsx"]) {
		const source = readFileSync(new URL(`../src/features/auth/${page}`, import.meta.url), "utf8");
		assert.match(source, /const personal = useConfirmedLoginMode\(\) === "single";/);
	}
	const client = readFileSync(new URL("../src/features/auth/login-client.ts", import.meta.url), "utf8");
	assert.match(client, /getConfirmedLoginMode = \(\): LoginMode \| null =>\n\s*statusKnown \? loginMode : null;/);
});

test("glass over an open conversation dims the image and thickens the composer and files panel", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	// Welcome comes from a root flag (welcome.ts) and the image from a host flag (host-flags.ts), not a :has().
	const conversation = String.raw`html\[data-glass\]:not\(\[data-chat-welcome\]\) :is\(\[data-wallpaper-workspace\], \[data-chat-content-root\]\)\[data-wallpaper-ready\]\[data-wallpaper-ready\]:not\(\[data-wallpaper-route="page"\]\)`;
	assert.match(css, new RegExp(`${conversation} > \\.chat-wallpaper,[^{]*\\{\\n  opacity: calc\\(\\.7 \\* `));
	assert.match(css, new RegExp(`${conversation} :is\\(\\.unsloth-composer-surface, \\.unsloth-files-panel\\),[^{]*\\{\\n  --glass-composer-boost: 9%;`));
	assert.match(css, /calc\(\(max\(calc\(var\(--glass-tint\) \+ 14%\), var\(--glass-read-floor, 0%\), var\(--glass-composer-floor\)\) \+ var\(--glass-composer-boost, 0%\)\)/);
});

test("glass frosts the chat model controls and gives the account row the row hover", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const model = String.raw`html\[data-glass\]\[data-chat-wallpaper-ready\] \[data-chat-model-controls\] > :is\(\.unsloth-model-selector-trigger, \[data-chat-model-notice\]\)`;
	const glass = new RegExp(`${model},\\n[^{]*\\{([^}]*)\\}`).exec(css)?.[1] ?? "";
	assert.match(css, new RegExp(`${model},\\nhtml\\[data-glass\\]\\[data-chat-wallpaper-workspace-ready\\] \\[data-wallpaper-route="page"\\] \\.unsloth-model-selector-trigger \\{`));
	assert.match(glass, /var\(--glass-composer-fill\) min\(96%, calc\(max\(calc\(var\(--glass-tint\) \+ 14%\)/);
	assert.match(glass, /backdrop-filter: blur\(var\(--glass-blur\)\) saturate/);
	assert.doesNotMatch(glass, /padding/);
	assert.match(css, new RegExp(`${model}:is\\(:hover[^{]*\\{\\n  background-image: linear-gradient\\(var\\(--glass-hover\\), var\\(--glass-hover\\)\\), var\\(--glass-sheen`));
	assert.match(css, /\[data-slot="sidebar-inner"\] \.app-user-trigger:is\(:hover, \[data-state="open"\]\)[^{]*\{\n  background-color: var\(--glass-hover\);\n  box-shadow: var\(--glass-hover-rim\);/);
	const page = readFileSync(new URL("../src/features/chat/chat-page.tsx", import.meta.url), "utf8");
	assert.match(page, /data-wallpaper-chrome-controls="" data-chat-model-controls=""/);
});

test("the narrow window's Run settings sheet takes the docked panel's glass", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const sheet = readFileSync(new URL("../src/features/chat/chat-settings-sheet.tsx", import.meta.url), "utf8");
	const compact = sheet.slice(sheet.indexOf("if (isCompact) {"), sheet.indexOf("</Sheet>"));
	assert.match(compact, /<SheetContent side="right" data-chat-settings-sheet=""/);
	assert.match(compact, /data-slot="chat-settings-panel" className="relative flex h-full flex-col"/);
	assert.match(css, /html\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-chat-settings-sheet\],\nhtml\[data-glass="macos"\] \[data-chat-settings-sheet\] \{\n  background-color: transparent;/);
	// The floating monitor measures whichever panel is shown, so it docks by the sheet's real width too.
	const monitor = readFileSync(new URL("../src/components/floating-monitor.tsx", import.meta.url), "utf8");
	assert.match(monitor, /document\.querySelector\('\[data-slot="chat-settings-panel"\]'\)/);
	assert.match(monitor, /if \(!attach\(\)\) \{\n\s*setPaintedSettingsWidth\(0\);\n\s*frame = requestAnimationFrame/);
});

test("glass frosts sticky bars in panels and popups, floating buttons, and softens shared shadows", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] :is\(\[data-slot="chat-settings-panel"\], \[data-slot\$="popover-content"\][^)]*\) \.sticky \{\n  -webkit-backdrop-filter: blur\(12px\)/);
	assert.match(css, /html\[data-glass\] :is\(\[data-glass-float\], \.aui-thread-scroll-to-bottom\) \{\n  -webkit-backdrop-filter: blur\(max\(var\(--glass-read-blur\), calc\(8px \* var\(--glass-strength, 1\)\)\)\);/);
	// Floating controls keep the readable floor and scale their fill with strength like every other glass surface.
	assert.match(css, /html\[data-glass\] \[data-glass-float\] \{\n  background-color: color-mix\(in srgb, color-mix\(in srgb, var\(--popover\), var\(--foreground\) 12%\) min\(96%, calc\(82% \* var\(--glass-opacity, 1\)\)\), transparent\) !important;/);
	assert.match(css, /html\.dark\[data-glass\] \{\n  --popup-shadow: 0 16px 40px -12px rgb\(0 0 0 \/ \.52\);/);
	const pickers = readFileSync(new URL("../src/features/model-picker/components/model-selector/pickers.tsx", import.meta.url), "utf8");
	assert.match(pickers, /data-glass-float=""\n\s*onClick=\{\(\) => \(ejectsAll/);
});

test("glass surfaces holding a frosted child move their own blur behind their content", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /\[data-slot="dialog-content"\]\):has\(\.sticky, \[data-glass-float\]\),\n[^{]*\{\n  -webkit-backdrop-filter: none !important;\n  backdrop-filter: none !important;\n  isolation: isolate;/);
	assert.match(css, /:has\(\.sticky, \[data-glass-float\]\)::before \{\n  content: "";\n  position: absolute;\n  inset: 0;\n  z-index: -1;/);
});

test("glass dims the wallpaper on other pages like an open chat, and blurs it nowhere", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\] > \.chat-wallpaper\[data-ready\],[^{]*\{\n  opacity: calc\(\.7 \* /);
	assert.doesNotMatch(css, /\.chat-wallpaper-artwork[^{]*\{\n  filter: blur\(/);
	assert.match(css, /\[data-wallpaper-route="page"\]\[data-wallpaper-media\] > \.chat-wallpaper\[data-ready\] \{\n  opacity: calc\(\.56 \* /);
	const root = readFileSync(new URL("../src/app/routes/__root.tsx", import.meta.url), "utf8");
	assert.match(root, /data-wallpaper-media=\{isImagesRoute \|\| isVideoRoute \|\| isAudioRoute \|\| undefined\}/);
});

test("glass leaves the unstuck library header clear", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /:is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\][^)]*\)\) \{\n  -webkit-backdrop-filter: none;\n  backdrop-filter: none;/);
	assert.match(css, /\[data-page-model-header\] \[data-slot="api-toolbar-fade"\] \{\n  display: none;/);
	const api = readFileSync(new URL("../src/features/api-monitor/api-monitor-page.tsx", import.meta.url), "utf8");
	assert.match(api, /data-page-model-header=""\n\s*className=\{cn\(\n\s*"pointer-events-none sticky top-0/);
	assert.doesNotMatch(css, /data-hub-pane/);
});

test("every non-chat page takes the chat model button's glass on its controls and cards", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const scope = String.raw`html\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-route="page"\] > \[data-slot="sidebar-inset"\] `;
	assert.match(css, new RegExp(`${scope}:where\\(\\.elevated-card, \\[class~="bg-card"\\][^{]*\\{\\n  background-color: var\\(--page-glass-fill, color-mix\\(in srgb, var\\(--glass-composer-fill\\)[^}]*backdrop-filter: blur\\(var\\(--glass-blur\\)\\)`));
	assert.match(css, new RegExp(`${scope}:where\\(button:not\\(:disabled, \\[data-slot="checkbox"\\], \\[data-slot="switch"\\]\\):is\\(\\[class\\*="hover:bg-"\\]:hover, \\[class\\*="open:bg-"\\]\\[data-state="open"\\]\\)\\) \\{[^}]*backdrop-filter: blur\\(var\\(--glass-blur\\)\\)`));
	assert.match(css, new RegExp(`${scope}\\.elevated-card:has\\(> \\[class~="bg-card"\\]\\) \\{\\n  background-color: transparent;`));
});

test("dark glass chat row menus keep the glass rim over the flat sidebar-menu shadow", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\^="liquid"\]\[data-glass-menus\]\.dark \.unsloth-plus-menu\.sidebar-menu\[data-slot\] \{\n  box-shadow: var\(--glass-rim\), var\(--glass-lift\) !important;/);
});

test("the browser menu takes the glass fill instead of its pinned dark surface", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\]\[data-glass-menus\] \.browser-menu\[data-slot\] \{\n  --browser-menu-surface: var\(--glass-menu-fill\);/);
});

test("menus, popovers and dialogs point every hover token at the glass hover", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const at = css.indexOf('[data-slot="alert-dialog-content"], [data-slot="sheet-content"]) {\n  --nav-surface-hover');
	assert.notEqual(at, -1);
	const rule = css.slice(at, css.indexOf("}", at));
	for (const token of ["nav-surface-hover", "panel-surface-hover", "chat-icon-bg-hover", "panel-input-surface-hover"]) {
		assert.match(rule, new RegExp(`--${token}: var\\(--glass-hover\\);`));
		assert.match(rule, new RegExp(`--color-${token}: var\\(--glass-hover\\);`));
	}
});

test("glass sheen runs at 90% except on the composer and files panel", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\^="liquid"\] \{\n  --glass-sheen: linear-gradient\(170deg, rgb\(255 255 255 \/ \.047\)/);
	assert.match(css, /html\[data-glass\^="liquid"\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel, \.wallpaper-preview-composer\) \{\n  --glass-sheen: linear-gradient\(170deg, rgb\(255 255 255 \/ \.053\)/);
	assert.match(css, /html\.dark\[data-glass="liquid-3d"\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel, \.wallpaper-preview-composer\) \{\n  --glass-edge:/);
});

test("grouped glass cards get more blur and a lighter fill than single controls", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /:is\(\.elevated-card, \.elevated-card > \[class~="bg-card"\], \.library-card\) \{\n  background-color: var\(--page-glass-card-fill, color-mix\(in srgb, color-mix\(in srgb, var\(--glass-composer-fill\), #fff 6%\)[^}]*backdrop-filter: blur\(calc\(var\(--glass-blur\) \+ 8px \* var\(--glass-strength, 1\)\)\)/);
	assert.ok(css.indexOf(".library-card) {") < css.indexOf(".elevated-card:has(> [class~=\"bg-card\"]) {"));
	const surface = readFileSync(new URL("../src/features/library/surface.ts", import.meta.url), "utf8");
	assert.match(surface, /CARD_SHADOW =\n  "library-card /);
});

test("light glass pages skip the extra wallpaper dim that hid the image under the reading wash", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html:not\(\.dark\)\[data-glass\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\] > \.chat-wallpaper\[data-ready\] \{\n  opacity: 1;/);
	assert.match(css, /html:not\(\.dark\)\[data-glass\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\]\[data-wallpaper-media\] > \.chat-wallpaper\[data-ready\] \{\n  opacity: \.8;/);
});

test("light glass pages ease the white reading wash so the wallpaper shows through", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html:not\(\.dark\)\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-workspace\]\[data-wallpaper-route="page"\] > \[data-slot="sidebar-inset"\] \{\n  --glass-page-wash: color-mix\(in srgb, var\(--glass-page-bg\) min\(96%, calc\(\(max\(var\(--glass-tint\), var\(--glass-pane-floor\)\) - 12%\)/);
});

test("light glass New chat lifts the wallpaper along a curve that still tracks the whole slider", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /\.chat-wallpaper-frame:not\(\[data-entering\], \[data-departing\]\) \{\n  opacity: min\(1, calc\(var\(--wallpaper-visibility\) \* 2\)\);\n\}/);
	assert.match(css, /@supports \(opacity: sqrt\(0\.5\)\) \{\n[^}]*\.chat-wallpaper-frame:not\(\[data-entering\], \[data-departing\]\) \{\n    opacity: sqrt\(var\(--wallpaper-visibility\)\);/);
	const lift = (v: number) => Math.sqrt(v);
	assert.ok(Math.abs(lift(0.24) - 0.5) < 0.02);
	assert.ok(lift(0.9) - lift(0.75) > 0.07 && lift(1) === 1 && lift(0) === 0);
});

test("light glass pages use the white control glass and a lighter card sheet, edged with an outline and inner light", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const light = String.raw`html\[data-glass\]:not\(\.dark\)\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-route="page"\] > \[data-slot="sidebar-inset"\]`;
	assert.match(css, new RegExp(`${light} \\{\\n  --page-glass-fill: rgb\\(255 255 255 / min\\(\\.96, calc\\(\\.3 \\* var\\(--glass-opacity, 1\\)\\)\\)\\);\\n  --page-glass-card-fill: rgb\\(255 255 255 / min\\(\\.96, calc\\(\\.45 \\* var\\(--glass-opacity, 1\\)\\)\\)\\);\\n  --page-glass-edge: var\\(--glass-light-edge\\);`));
	assert.match(css, new RegExp(`${light} \\.elevated-card > \\[class~="bg-card"\\] \\{\\n  background-color: transparent;\\n  box-shadow: inset -1px 0 0 var\\(--page-glass-divider\\), inset 0 -1px 0 var\\(--page-glass-divider\\);`));
	assert.match(css, new RegExp(`${light} :is\\(\\.elevated-card, \\.library-card\\) \\{\\n  box-shadow: var\\(--page-glass-card-edge\\);`));
	assert.match(css, new RegExp(`${light} \\.glass-divider \\{\\n  border-color: var\\(--page-glass-divider\\);`));
	assert.match(css, new RegExp(`${light} \\.unsloth-model-selector-trigger \\{\\n  background-color: var\\(--glass-control\\) !important;\\n  box-shadow: var\\(--glass-light-edge\\);`));
	const api = readFileSync(new URL("../src/features/api-monitor/api-monitor-page.tsx", import.meta.url), "utf8");
	assert.match(api, /const DIVIDER = "glass-divider /);
	assert.match(api, /const INSET =\n  "glass-inset /);
});

test("light glass API cards use a thinner fill with the outlined glass edge", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\]:not\(\.dark\)\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-route="page"\] > \[data-slot="sidebar-inset"\] \.api-monitor-page \{\n  --page-glass-card-fill: rgb\(255 255 255 \/ min\(\.96, calc\(\.3 \* var\(--glass-opacity, 1\)\)\)\);\n  --page-glass-card-edge: var\(--glass-light-edge\);/);
	assert.match(css, /\.api-monitor-page \.elevated-card:has\(> \[class~="bg-card"\]\) \{\n  clip-path: inset\(0 1px 1px 0 round var\(--radius-2xl\)\);/);
});

test("light glass outlines model buttons and the status chip, but not the sidebar", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /  --glass-light-inner: inset 0 1px 0 rgb\(255 255 255 \/ \.7\);/);
	assert.match(css, /html:not\(\.dark\)\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-slot="sidebar-inner"\] \{\n  border-color: transparent;\n  box-shadow: none;/);
	assert.match(css, /html:not\(\.dark\)\[data-glass\]\[data-chat-wallpaper-ready\] \[data-chat-model-controls\] > :is\(\.unsloth-model-selector-trigger, \[data-chat-model-notice\]\) \{\n  box-shadow: var\(--glass-light-edge\);/);
	assert.match(css, /\.glass-inset \{\n  box-shadow: var\(--glass-light-edge\);/);
	assert.match(css, /  --glass-light-edge: inset 0 0 0 1px color-mix\(in srgb, var\(--foreground\) var\(--element-outline, 12%\), transparent\), var\(--glass-light-inner\);/);
});

test("light glass draws no grey outlines outside the composer", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const block = (selector: string) => {
		const start = css.indexOf(`\n${selector} {\n`);
		assert.ok(start >= 0, selector);
		return css.slice(start, css.indexOf("\n}", start + 1));
	};
	for (const selector of ['html[data-glass^="liquid"]', 'html[data-glass="liquid-3d"]', "html[data-glass]:not(.dark)"]) {
		assert.doesNotMatch(block(selector), /--glass-(rim|control-rim): [^;]*inset 0 0 0 1px/, selector);
	}
	assert.match(css, /html\[data-glass\] \{\n  --glass-hover: [^;]*;\n  --glass-hover-rim: inset 0 0 0 0 transparent;/);
	assert.match(css, /html\.dark\[data-glass\] \{\n[^}]*--glass-hover-rim: inset 0 0 0 1px rgb\(255 255 255 \/ \.06\);/);
	assert.match(css, /html\[data-glass="liquid-3d"\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel, \.wallpaper-preview-composer\) \{\n[^}]*--glass-rim: [^;]*inset 0 0 0 1px/);
});

test("glass decoration has no blurred inset shadows or radial sheens", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const glass = css.slice(css.indexOf('html[data-glass="frosted"]'), css.indexOf(".glass-lens {"));
	assert.ok(glass.length > 10000);
	assert.deepEqual(glass.match(/inset 0 -?[\d.]+px [1-9][\d.]*px/g), null);
	assert.deepEqual(glass.match(/--glass-sheen: radial-gradient/g), null);
});

test("light glass pages keep the borders of boxed panels and select triggers", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.doesNotMatch(css, /:where\(\[class~="border"\]\[class\*="rounded"\]\):not\(\[data-slot="checkbox"\], \[role="checkbox"\], \[role="radio"\], \[role="switch"\], :focus-visible, \[aria-invalid="true"\]\) \{\n  border-color: transparent;/);
});

test("light glass border clearing leaves focused and invalid fields their ring border", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const cleared = [...css.matchAll(/([^{}]*)\{\n  border-color: transparent;\n\}/g)].map(([, sel]) => sel).filter((sel) => sel.includes('[data-wallpaper-route="page"]') && sel.includes(":not(.dark)"));
	assert.ok(cleared.length >= 1);
	for (const sel of cleared) assert.match(sel, /:focus-visible, \[aria-invalid="true"\]\)/, sel);
});

test("focused glass surfaces on pages show their Tailwind focus ring over the glass edge", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /\[data-slot="sidebar-inset"\] :focus-visible:is\(\[class\*="focus-visible:ring"\]\) \{\n  box-shadow: var\(--tw-inset-shadow, 0 0 #0000\), var\(--tw-inset-ring-shadow, 0 0 #0000\), var\(--tw-ring-offset-shadow, 0 0 #0000\), var\(--tw-ring-shadow, 0 0 #0000\), var\(--tw-shadow, 0 0 #0000\) !important;/);
	const cards = readFileSync(new URL("../src/features/library/components/library-cards.tsx", import.meta.url), "utf8");
	assert.match(cards, /focus-visible:ring-2/);
});

test("transparent glass keeps each surface's colour at the strength's opacity, with no shadow or blur", () => {
	const material = readFileSync(new URL("../src/features/chat-background/material.ts", import.meta.url), "utf8");
	// Its own attribute, so none of the frosted html[data-glass] rules apply on top.
	assert.match(material, /if \(applied && applied !== "transparent"\) root\.setAttribute\("data-glass", applied\);/);
	assert.match(material, /root\.toggleAttribute\("data-glass-clear", applied === "transparent"\);/);
	assert.match(material, /root\.style\.setProperty\("--glass-fill-alpha", `\$\{strength\}%`\);/);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const block = css.slice(css.indexOf("/* Transparent glass: the plain theme"));
	// Shared surface tokens resolve from the opaque html palette.
	for (const [token, copy] of [["background", "bg"], ["card", "card"], ["popover", "popover"], ["muted", "muted"], ["secondary", "secondary"], ["sidebar", "sidebar"]]) {
		assert.match(block, new RegExp(`html\\[data-glass-clear\\] #root,\\nhtml\\[data-glass-clear\\] #root :is\\(\\[data-sonner-toaster\\], \\.toast-surface\\) \\{[^}]*--${token}: color-mix\\(in srgb, var\\(--glass-page-${copy}\\) var\\(--clear-alpha\\), transparent\\);`), token);
	}
	// Sonner renders inside #root (no portal), so toasts re-resolve the tokens at full opacity.
	assert.match(block, /html\[data-glass-clear\] #root \[class\*="text-background"\] \{\n  --background: var\(--glass-page-bg\);\n  --color-background: var\(--glass-page-bg\);\n\}/);
	assert.match(css, /--glass-page-sidebar: var\(--sidebar\);/);
	assert.match(block, /html\[data-glass-clear\] #root :is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\][^)]*\)\) \{\n\s*background-color: transparent;/);
	assert.match(block, /@layer base \{\n\s*html\[data-glass-clear\] body :not\(:focus-visible\),[^{]*\{\n\s*box-shadow: none !important;\n\s*text-shadow: none !important;/);
	assert.match(block, /html\[data-glass-clear\] #root \* \{\n\s*-webkit-backdrop-filter: none !important;\n\s*backdrop-filter: none !important;/);
});

test("Windows Transparent glass shares its palette with opted-in popups and paints panel headers once", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const clear = css.slice(css.indexOf("/* Transparent glass: the plain theme"));
	const palette = clear.slice(0, clear.indexOf("--clear-alpha:"));
	assert.ok(palette.includes("html.tauri.client-windows[data-glass-clear][data-glass-menus]"));
	for (const slot of ['[data-slot$="menu-content"]', '[data-slot="dialog-content"]']) {
		assert.ok(palette.includes(slot), slot);
	}
	for (const target of ['[data-slot="chat-settings-panel"] .bg-panel-surface', '[data-settings-dialog]']) {
		const rule = clear.split(target + " {")[1]?.split("}")[0];
		assert.ok(rule?.includes("background-color: transparent;"), target);
	}
});

test("the model page's Back to Hub bar paints no strip over a background image until content scrolls under it", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const resting = String.raw`\.hub-detail-bar:not\(\[data-scrolled\]\)`;
	assert.match(css, new RegExp(String.raw`\[data-wallpaper-workspace\]\[data-wallpaper-ready\]\[data-wallpaper-ready\] :is\([^)]*\.hub-page \.hub-canvas[^{]*${resting}\) \{ background: transparent; \}`));
	assert.match(css, new RegExp(String.raw`:is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\], ${resting}\) \{\n\s*-webkit-backdrop-filter: none;`));
	assert.match(css, new RegExp(String.raw`html\[data-glass-clear\] #root :is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\], ${resting}\) \{\n\s*background-color: transparent;`));
});

test("macOS glass and a plain background image leave the resting page headers clear", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const block = css.slice(css.indexOf("/* macOS glass sees the window through every page"));
	assert.match(block, /^\/\*[^]*?\*\/\nhtml\[data-glass="macos"\] \[data-slot="sidebar-inset"\] :is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\], \.hub-page \.hub-detail-bar:not\(\[data-scrolled\]\)\),\nhtml:not\(\[data-glass\]\) \[data-wallpaper-workspace\]\[data-wallpaper-ready\]\[data-wallpaper-ready\] > \[data-slot="sidebar-inset"\] :is\(\.library-header-row\.sticky:not\(\[data-stuck\]\), \[data-page-model-header\]\) \{\n  background-color: transparent;/);
	assert.match(block, /html\[data-glass="macos"\] \[data-page-model-header\] \[data-slot="api-toolbar-fade"\],\nhtml:not\(\[data-glass\]\) [^{]*\[data-page-model-header\] \[data-slot="api-toolbar-fade"\] \{\n  display: none;/);
});

test("the update cards that float like toasts take the glass toast surface", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\]\[data-glass-menus\] :is\(\[data-sonner-toast\]\[data-styled="true"\], \.toast-surface\) \{[^}]*backdrop-filter: blur\(max\(var\(--glass-blur\), calc\(18px \* var\(--glass-strength, 1\)\)\)\)/);
	assert.match(css, /html\[data-glass\^="liquid"\]\[data-glass-menus\] :is\(\[data-sonner-toast\]\[data-styled="true"\], \.toast-surface\) \{/);
	// Transparent keeps toasts opaque; the cards re-resolve the palette at full alpha like the toaster.
	assert.match(css, /html\[data-glass-clear\] #root :is\(\[data-sonner-toaster\], \.toast-surface\) \{\n\s*--clear-alpha: 100%;/);
	for (const file of ["components/llama-update-banner.tsx", "components/tauri/update-banner.tsx", "components/web/update-banner.tsx"]) {
		const source = readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");
		assert.match(source, /className="toast-surface /, file);
	}
});

test("the glass lens draws into every glass surface, not only the composer", async () => {
	const { LENS_HOSTS } = await import("../src/features/chat-background/model.ts");
	assert.equal(typeof LENS_HOSTS, "string");
	for (const host of [
		".unsloth-composer-surface",
		'[data-slot="sidebar-inner"]',
		"[data-windows-glass-frame]",
		'[data-slot="chat-settings-panel"]',
		'[data-slot="sheet-content"]',
		'[data-slot="dialog-content"]',
		"[data-chat-model-controls] > .unsloth-model-selector-trigger",
		'[data-slot$="menu-content"]',
		'[data-slot="popover-content"]',
		".tooltip-compact",
		".tooltip-rich",
		"[data-glass-float]",
		".aui-thread-scroll-to-bottom",
	])
		assert.ok(LENS_HOSTS.includes(host), host);
	// A card grid would clone the wallpaper twice per card.
	assert.doesNotMatch(LENS_HOSTS, /elevated-card|library-card/);
	// Every surface the Glass menus blur reaches (menus, pickers, toasts) is a lens host too.
	const glassCss = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	for (const rule of [
		/^html\[data-glass\]\[data-glass-menus\] :is\((\[data-slot\$="menu-content"\].*?)\) \{$/m,
		/^html\[data-glass\]\[data-glass-menus\] :is\((\[data-sonner-toast\].*?)\) \{$/m,
	]) {
		const list = glassCss.match(rule)?.[1];
		assert.ok(list, String(rule));
		for (const selector of list.split(", ")) assert.ok(LENS_HOSTS.includes(selector), selector);
	}
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	// Menus and sheets are portaled to <body>; hosts without a backdrop filter (Glass menus off, opaque sheets) are skipped.
	assert.match(lens, /\(preview \? PREVIEW_HOSTS : LENS_HOSTS\),\n\s*\]\.filter\(\(h\) => drawsGlass\(h\) && \(!clear \|\| h\.matches\(LENS_CLEAR_HOSTS\)\)\)/);
	// Switching glass style (liquid-only rules) changes which hosts draw glass without a DOM mutation.
	assert.match(lens, /\}, \[workspace, on, material, menus, preview, clear\]\);/);
	assert.match(lens, /observer\.observe\(preview \? workspace : document\.body, \{ childList: true, subtree: true \}\);/);
	// A host that was only moving drops its lens when the motion flag clears.
	assert.match(lens, /motion\.observe\(preview \? workspace : document\.body, \{ attributes: true, subtree: true, attributeFilter: \[GLASS_MOVING_ATTR\] \}\);/);
	// The Desktop sidebar frame and layered panels draw their glass (and fill) on ::before.
	assert.match(lens, /for \(const pseudo of \[null, "::before"\]\)/);
	assert.match(lens, /const fill = !glass \|\| painted\(own\) \? own : glass;/);
	// The copy blurs as much as the surface it sits in, not the shared --glass-blur.
	assert.match(lens, /if \(fillCopy\) fillCopy\.style\.filter = blur \|\| \(host\.hasAttribute\(GLASS_MOVING_ATTR\) \? "none" : ""\);/);
	// Outside the workspace the wash stops at <body>, whose opaque fill would hide the copy.
	assert.match(lens, /node && node !== workspace && node !== document\.body;/);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	// The lens paints at z-index -1, so each host is its own stacking context.
	assert.match(css, /\[data-glass-lens-host\] \{\n  isolation: isolate;\n\}\n\[data-glass-lens-host="static"\] \{\n  position: relative;\n\}/);
	// The frame's ::before (z-index 1) draws its glass, so its lens sits above that and below the sidebar, without isolation.
	assert.match(css, /\[data-windows-glass-frame\]\[data-glass-lens-host\] \{\n  isolation: auto;\n\}\n\[data-windows-glass-frame\] > \.glass-lens \{\n  z-index: 1;\n\}/);
	assert.match(css, /\[data-windows-glass-frame\]\[data-glass-lens-host\] > \[data-slot="sidebar-inner"\] \{\n  z-index: 2;\n\}/);
	assert.doesNotMatch(css, /:has\(> \.glass-lens\)/);
	for (const file of ["background.tsx", "backdrop.tsx"]) {
		const source = readFileSync(new URL(`../src/features/chat-background/${file}`, import.meta.url), "utf8");
		assert.match(source, /<GlassLenses wallpaper=\{appWallpaper\} \/>/, file);
	}
	// The Appearance previews show the lens on their sample composers, over their own wallpaper.
	const background = readFileSync(new URL("../src/features/chat-background/background.tsx", import.meta.url), "utf8");
	assert.match(background, /\{previewWallpaper && <GlassLenses wallpaper=\{previewWallpaper\} preview \/>\}/);
	assert.match(lens, /\(preview \? workspace : document\)\.querySelectorAll<HTMLElement>\(preview \? PREVIEW_HOSTS : LENS_HOSTS\)/);
});

test("glass strength scales the blur and the fill of every glass surface, with no fixed floors", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
	// Fixed blur floors and add-ons left the menus' low end and the open composer's extra blur deaf to the slider.
	assert.doesNotMatch(css, /max\(var\(--glass-blur\), \d+px\)/);
	assert.doesNotMatch(css, /\+ 5px \+ var\(--glass-repaint/);
	assert.doesNotMatch(css, /calc\(var\(--glass-blur\) \+ 8px\)/);
	// Every backdrop blur on a glass surface reads the strength, except the clear (Transparent) theme's,
	// and sticky headers, which frost their own panel's scrolled content rather than the wallpaper.
	const off: string[] = [];
	for (const [, selector, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
		if (!/data-glass(?!-clear)/.test(selector) || /data-glass-clear|\.sticky|data-config-header-row|data-pinned-footer|data-glass-motion/.test(selector)) continue;
		for (const [, filter] of body.matchAll(/(?<!-webkit-)backdrop-filter: blur\(([^;]*)\)/g))
			if (!/--glass-blur|--glass-strength/.test(filter)) off.push(selector.trim().slice(0, 120));
	}
	assert.deepEqual(off, []);
	// Light pages and the macOS strip had fixed fills: the slider moved their blur but not their tint.
	assert.match(css, /--page-glass-card-fill: rgb\(255 255 255 \/ min\(\.96, calc\(\.45 \* var\(--glass-opacity, 1\)\)\)\);/);
	assert.match(css, /--page-glass-fill: rgb\(255 255 255 \/ min\(\.96, calc\(\.3 \* var\(--glass-opacity, 1\)\)\)\);/);
	assert.match(css, /html\[data-glass="macos"\] :is\(\[data-slot="sidebar-inset"\], \[data-slot="window-titlebar"\]\) \{\n  background-color: color-mix\(in srgb, var\(--background\) min\(96%, calc\(52% \* var\(--glass-opacity, 1\)\)\), transparent\);/);
});

test("every glass surface keeps a readable blur floor at any strength, lens copy included", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] \{\n  --glass-tint: var\(--glass-tint-base\);\n  --glass-on: min\(1, var\(--glass-strength, 1\) \* 100\);\n  --glass-read-blur: calc\(12px \* var\(--glass-on\)\);\n  --glass-blur: calc\(max\(var\(--glass-read-blur\), var\(--glass-blur-base\) \* var\(--glass-strength, 1\)\) \+ var\(--glass-repaint, 0\) \* \.01px\);/);
	// The open chat's composer redefines --glass-blur, so it carries the floor too.
	assert.match(css, /--glass-blur: calc\(max\(var\(--glass-read-blur\), \(var\(--glass-blur-base\) \+ 5px\) \* var\(--glass-strength, 1\)\) \+ var\(--glass-repaint, 0\) \* \.01px\);/);
	// Every other --glass-blur definition goes through the floor.
	for (const [, value] of css.matchAll(/--glass-blur: ([^;]+);/g)) assert.match(value, /var\(--glass-read-blur\)/, value);
	// The lens copy blurs with the same variable.
	assert.match(css, /\.glass-lens-fill \.glass-lens-copy \{\n  filter: blur\(var\(--glass-blur, 12px\)\)/);
});

// Evaluates a glass blur() argument from index.css at a given --glass-strength (Liquid Glass base 10px, no repaint nudge).
function glassBlurPx(expr: string, strength: number): number {
	const on = Math.min(1, strength * 100);
	const read = 12 * on;
	const js = expr
		.replaceAll("var(--glass-on)", String(on))
		.replaceAll("var(--glass-strength, 1)", String(strength))
		.replaceAll("var(--glass-read-blur)", `${read}px`)
		.replaceAll("var(--glass-blur)", `max(${read}px, 10px * ${strength})`)
		.replaceAll("var(--glass-blur-base)", "10px")
		.replaceAll("var(--glass-repaint, 0)", "0")
		.replaceAll("calc(", "(")
		.replaceAll("max(", "Math.max(")
		.replaceAll("min(", "Math.min(")
		.replaceAll("px", "");
	assert.match(js, /^[\d\s.+*(),Mathmxin-]+$/, `unexpected token in ${expr}`);
	return Function(`return ${js}`)() as number;
}

test("glass strength 0% is clear on every surface, and 1% keeps the 12px read floor", () => {
	assert.deepEqual(glassStrengthVars(0), { blur: 0, opacity: 0.35 });
	assert.ok(glassStrengthVars(1).blur > 0);
	assert.equal(glassStrengthVars(50).blur, 1);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /--glass-on: min\(1, var\(--glass-strength, 1\) \* 100\);\n  --glass-read-blur: calc\(12px \* var\(--glass-on\)\);/);
	assert.match(css, /--glass-blur: calc\(max\(var\(--glass-read-blur\), var\(--glass-blur-base\) \* var\(--glass-strength, 1\)\) \+ var\(--glass-repaint, 0\) \* \.01px\);/);
	const blurs = [...css.matchAll(/(?<!-webkit-)backdrop-filter: blur\(((?:[^()]|\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))*)\)/g)]
		.map((m) => m[1])
		.filter((arg) => arg.includes("var(--glass-"));
	assert.ok(blurs.length >= 10, `only ${blurs.length} glass blurs found`);
	const at1 = glassStrengthVars(1).blur;
	for (const arg of blurs) {
		assert.equal(glassBlurPx(arg, 0), 0, `not clear at 0%: blur(${arg})`);
		assert.ok(glassBlurPx(arg, at1) >= 12, `under the read floor at 1%: blur(${arg})`);
	}
});

test("at glass strength 0% the lens stays on the composer only, as on main", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /const clear = useGlassStrength\(\) === 0;/);
	assert.match(lens, /\.filter\(\(h\) => drawsGlass\(h\) && \(!clear \|\| h\.matches\(LENS_CLEAR_HOSTS\)\)\)/);
	assert.ok(LENS_HOSTS.split(", ").includes(".unsloth-composer-surface"));
	assert.equal(LENS_CLEAR_HOSTS, ".unsloth-composer-surface, .wallpaper-preview-composer");
});

test("each lens portal is keyed by its host, so a host joining the list does not remount the others", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /`glass-lens-\$\{hostKey\(host\)\}`/);
	assert.doesNotMatch(lens, /`glass-lens-\$\{i\}`/);
});

test("the lens tint copies the host's fill when only its ::before carries the blur", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /const fill = !glass \|\| painted\(own\) \? own : glass;/);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /:not\(\[data-glass-layers\]\) \[data-slot="dialog-content"\]\[data-settings-dialog\] \{\s*-webkit-backdrop-filter: none;/);
});

test("a lens stays on the visible box of a host that scrolls itself", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /if \(el\.style\.translate\) el\.style\.translate = "";\s*const sx = host\.scrollLeft;\s*const sy = host\.scrollTop;\s*el\.style\.translate = sx \|\| sy \? `\$\{sx\}px \$\{sy\}px` : "";/);
	assert.match(lens, /host\.addEventListener\("scroll", place, \{ passive: true \}\);/);
	assert.match(lens, /host\.removeEventListener\("scroll", place\);/);
});

test("legacy WebKitGTK's sidebar glass takes the lens instead of the masked frame, never both", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /const FRAME_SIDEBAR = '\[data-windows-glass-frame\] > \[data-slot="sidebar-inner"\]';/);
	assert.match(lens, /if \(framed\.size\) found = found\.filter\(\(h\) => !framed\.has\(h\)\);/);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /webkit-legacy-compositor\[data-glass\]\[data-chat-wallpaper-workspace-ready\] \[data-windows-glass-frame\] > \[data-slot="sidebar-inner"\] \{[^}]*backdrop-filter: blur/);
});

test("a lens never portals into a text-only host, and the title tooltip wraps its text", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /if \(width < 8 \|\| height < 8 \|\| textOnly\(el\)\) return false;/);
	const tip = readFileSync(new URL("../src/components/ui/title-tooltip.tsx", import.meta.url), "utf8");
	assert.match(tip, /<span>\{shown\.text\}<\/span>/);
});

test("the composer keeps its lens without a backdrop filter, as on main (Transparent)", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /return el\.matches\(LENS_CLEAR_HOSTS\) \|\| el\.hasAttribute\(GLASS_MOVING_ATTR\) \|\| glassLayer\(el\) !== null;/);
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass-clear\] #root \* \{\s*-webkit-backdrop-filter: none !important;/);
});

test("the Appearance preview composers wrap their prompt, so a lens can mount in them", () => {
	const controls = readFileSync(new URL("../src/features/chat-background/controls.tsx", import.meta.url), "utf8");
	assert.match(controls, /<div className="wallpaper-preview-composer rounded-xl p-3">\s*<span>\{t\("chatBackground\.prompt"\)\}<\/span>/);
});

test("a lens copy drops its blur while its host is moving, instead of falling back to the CSS blur", () => {
	const lens = readFileSync(new URL("../src/features/chat-background/lens.tsx", import.meta.url), "utf8");
	assert.match(lens, /if \(fillCopy\) fillCopy\.style\.filter = blur \|\| \(host\.hasAttribute\(GLASS_MOVING_ATTR\) \? "none" : ""\);/);
	assert.match(lens, /motion\.observe\(host, \{ attributes: true, attributeFilter: \[GLASS_MOVING_ATTR, GLASS_SETTLING_ATTR, "style"\] \}\);/);
	assert.match(lens, /for \(let node = host\.parentElement; node && node !== document\.body; node = node\.parentElement\)\s*motion\.observe\(node, \{ attributes: true, attributeFilter: \["style"\] \}\);/);
	assert.match(lens, /motion\.disconnect\(\);/);
});

test("only the composer draws a lens rim; other glass hosts keep their own edge instead of a doubled outline", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	const rule = css.match(/\[data-glass-lens-host\]:not\(([^)]+)\) > \.glass-lens \.glass-lens-rim \{\s*display: none;\s*\}/);
	assert.ok(rule, "lens rim is hidden on non-composer hosts");
	assert.deepEqual(rule[1].split(",").map((s) => s.trim()).sort(), LENS_CLEAR_HOSTS.split(",").map((s) => s.trim()).sort());
});
