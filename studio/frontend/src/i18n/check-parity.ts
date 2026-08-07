// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Parity check between en.ts and every non-English locale:
// - Locale files may be partial; required release surfaces must be translated.
// - All non-English keys must exist in en (no extras).
// - Placeholder set must match per leaf between en and the overlay.
// --strict also fails on missing keys, naming them (see studio-frontend-ci.yml).
// Run: npm run i18n:check   (or: npx tsx src/i18n/check-parity.ts [--strict])

import { ar } from "./locales/ar.ts";
import { de } from "./locales/de.ts";
import { en } from "./locales/en.ts";
import { es } from "./locales/es.ts";
import { fr } from "./locales/fr.ts";
import { hi } from "./locales/hi.ts";
import { it } from "./locales/it.ts";
import { ja } from "./locales/ja.ts";
import { ko } from "./locales/ko.ts";
import { ptBR } from "./locales/pt-br.ts";
import { ru } from "./locales/ru.ts";
import { zhCN } from "./locales/zh-CN.ts";

type Tree = { readonly [k: string]: string | Tree };

function isTree(v: unknown): v is Tree {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function placeholders(s: string): string[] {
  const out: string[] = [];
  const re = /\{([a-zA-Z0-9_]+)\}/g;
  let m: RegExpExecArray | null;
  while (true) {
    m = re.exec(s);
    if (m === null) {
      break;
    }
    out.push(m[1]);
  }
  return out.sort();
}

function checkOverlay(
  enNode: Tree,
  overlay: Tree | undefined,
  path: string,
  errors: string[],
  missing: string[],
): void {
  for (const [k, v] of Object.entries(enNode)) {
    const subPath = path ? `${path}.${k}` : k;
    if (typeof v === "string") {
      if (overlay === undefined) {
        missing.push(subPath);
        continue;
      }
      const overlayV = overlay[k];
      if (overlayV === undefined) {
        missing.push(subPath);
        continue;
      }
      if (typeof overlayV !== "string") {
        errors.push(`${subPath} should be string, got ${typeof overlayV}`);
        continue;
      }
      const enP = placeholders(v);
      const ovP = placeholders(overlayV);
      if (JSON.stringify(enP) !== JSON.stringify(ovP)) {
        errors.push(
          `${subPath}: placeholder mismatch en={${enP.join(",")}} overlay={${ovP.join(",")}}`,
        );
      }
    } else if (isTree(v)) {
      const overlaySub = overlay === undefined ? undefined : overlay[k];
      if (overlaySub !== undefined && !isTree(overlaySub)) {
        errors.push(`${subPath} should be an object, got ${typeof overlaySub}`);
        continue;
      }
      checkOverlay(v, overlaySub, subPath, errors, missing);
    }
  }
}

function checkExtras(
  overlay: Tree,
  enNode: Tree,
  path: string,
  errors: string[],
): void {
  for (const [k, v] of Object.entries(overlay)) {
    const subPath = path ? `${path}.${k}` : k;
    if (!(k in enNode)) {
      errors.push(`${subPath} exists in overlay but not in en`);
      continue;
    }
    const enV = enNode[k];
    if (isTree(v) && isTree(enV)) {
      checkExtras(v, enV, subPath, errors);
    } else if (isTree(v) !== isTree(enV)) {
      errors.push(
        `${subPath}: shape mismatch (en=${typeof enV}, overlay=${typeof v})`,
      );
    }
  }
}

const overlays: Record<string, Tree> = {
  "zh-CN": zhCN as unknown as Tree,
  "pt-BR": ptBR as unknown as Tree,
  ja: ja as unknown as Tree,
  es: es as unknown as Tree,
  hi: hi as unknown as Tree,
  ar: ar as unknown as Tree,
  fr: fr as unknown as Tree,
  ru: ru as unknown as Tree,
  de: de as unknown as Tree,
  ko: ko as unknown as Tree,
  it: it as unknown as Tree,
};

const requiredOverlayPrefixes = ["picker.", "studio."] as const;

function requiresOverlay(key: string): boolean {
  return requiredOverlayPrefixes.some((prefix) => key.startsWith(prefix));
}

const strict = process.argv.includes("--strict");
let anyError = false;
let anyMissing = false;

for (const [locale, overlay] of Object.entries(overlays)) {
  const errors: string[] = [];
  const missing: string[] = [];
  checkOverlay(en as unknown as Tree, overlay, "", errors, missing);
  checkExtras(overlay, en as unknown as Tree, "", errors);
  for (const key of missing) {
    if (requiresOverlay(key)) {
      errors.push(`${key} must be localized`);
    }
  }

  console.log(`\n=== ${locale} ===`);
  console.log(`Missing keys (will fall back to en): ${missing.length}`);
  if (missing.length > 0) {
    anyMissing = true;
    // Name them either way: the count alone does not say what to translate.
    for (const k of missing) {
      console.log(`  - ${k}`);
    }
  }
  if (errors.length > 0) {
    anyError = true;
    console.error(`Errors (${errors.length}):`);
    for (const e of errors) {
      console.error(`  - ${e}`);
    }
  } else {
    console.log("No errors.");
  }
}

if (anyError) {
  process.exit(1);
}
if (strict && anyMissing) {
  console.error(
    "\nMissing keys above. Add them to every overlay, or run without --strict " +
      "to accept the English fallback.",
  );
  process.exit(1);
}
console.log("\nAll locale overlays pass parity.");
