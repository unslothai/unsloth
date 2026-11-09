import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	MATERIALS,
	LENS_SCALE,
	inScope,
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
	assert.match(lens, /opacity,\n\s*fill\.backgroundColor,\n\s*fill\.backgroundImage,\n\s*\]\.join\(\)/);
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

test("glass frosts sticky bars in panels and popups, floating buttons, and softens shared shadows", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /html\[data-glass\] :is\(\[data-slot="chat-settings-panel"\], \[data-slot\$="popover-content"\][^)]*\) \.sticky \{\n  -webkit-backdrop-filter: blur\(12px\)/);
	assert.match(css, /html\[data-glass\] :is\(\[data-glass-float\], \.aui-thread-scroll-to-bottom\) \{\n  -webkit-backdrop-filter: blur\(8px\);/);
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
	assert.match(css, /html\[data-glass\^="liquid"\] \{\n  --glass-sheen: radial-gradient\(130% 90% at 18% 0%, rgb\(255 255 255 \/ \.059\)/);
	assert.match(css, /html\[data-glass\^="liquid"\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel, \.wallpaper-preview-composer\) \{\n  --glass-sheen: radial-gradient\(130% 90% at 18% 0%, rgb\(255 255 255 \/ \.066\)/);
	assert.match(css, /html\.dark\[data-glass="liquid-3d"\] :is\(\.unsloth-composer-surface, \.unsloth-files-panel, \.wallpaper-preview-composer\) \{\n  --glass-edge:/);
});

test("grouped glass cards get more blur and a lighter fill than single controls", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /:is\(\.elevated-card, \.elevated-card > \[class~="bg-card"\], \.library-card\) \{\n  background-color: var\(--page-glass-card-fill, color-mix\(in srgb, color-mix\(in srgb, var\(--glass-composer-fill\), #fff 6%\)[^}]*backdrop-filter: blur\(calc\(var\(--glass-blur\) \+ 8px\)\)/);
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
	assert.match(css, new RegExp(`${light} \\{\\n  --page-glass-fill: var\\(--glass-control\\);\\n  --page-glass-card-fill: rgb\\(255 255 255 / \\.45\\);\\n  --page-glass-edge: var\\(--glass-light-edge\\);`));
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
	assert.match(css, /html\[data-glass\]:not\(\.dark\)\[data-chat-wallpaper-workspace-ready\] \[data-wallpaper-route="page"\] > \[data-slot="sidebar-inset"\] \.api-monitor-page \{\n  --page-glass-card-fill: rgb\(255 255 255 \/ \.3\);\n  --page-glass-card-edge: var\(--glass-light-edge\);/);
	assert.match(css, /\.api-monitor-page \.elevated-card:has\(> \[class~="bg-card"\]\) \{\n  clip-path: inset\(0 1px 1px 0 round var\(--radius-2xl\)\);/);
});

test("light glass outlines model buttons and the status chip, but not the sidebar", () => {
	const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
	assert.match(css, /  --glass-light-inner: inset 0 1px 1px rgb\(255 255 255 \/ \.7\), inset 0 16px 30px -20px rgb\(255 255 255 \/ \.55\), inset 0 -18px 34px -20px rgb\(255 255 255 \/ \.4\);/);
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
		assert.match(block, new RegExp(`html\\[data-glass-clear\\] #root,\\nhtml\\[data-glass-clear\\] #root \\[data-sonner-toaster\\] \\{[^}]*--${token}: color-mix\\(in srgb, var\\(--glass-page-${copy}\\) var\\(--clear-alpha\\), transparent\\);`), token);
	}
	// Sonner renders inside #root (no portal), so toasts re-resolve the tokens at full opacity.
	assert.match(block, /html\[data-glass-clear\] #root \[data-sonner-toaster\] \{\n  --clear-alpha: 100%;\n\}/);
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
