// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Checks whether merging adjacent same-styled Shiki tokens removes spans. It does not (0.0% on
 * the studiobench corpus); kept so the result is checkable. Usage: node <this> [--theme dark] f.md
 */

import { readFileSync } from "node:fs";
import oneDarkPro from "@shikijs/themes/one-dark-pro";
import oneLight from "@shikijs/themes/one-light";
import { createHighlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

const withTransparentBg = (t) => ({
  ...t,
  bg: "transparent",
  colors: { ...t.colors, "editor.background": "transparent" },
});
const light = { ...withTransparentBg(oneLight), name: "unsloth-light" };
const dark = { ...withTransparentBg(oneDarkPro), name: "unsloth-dark" };

// `htmlAttrs` blocks a merge: it may carry ids or data attributes something depends on.
const sameHtmlStyle = (a, b) => {
  if (a === b) return true;
  const ka = a ? Object.keys(a) : [];
  const kb = b ? Object.keys(b) : [];
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (a?.[k] !== b?.[k]) return false;
  return true;
};
const hasAttrs = (t) => t.htmlAttrs !== undefined && Object.keys(t.htmlAttrs).length > 0;
const mergeable = (a, b) =>
  a.color === b.color &&
  a.bgColor === b.bgColor &&
  a.fontStyle === b.fontStyle &&
  !hasAttrs(a) &&
  !hasAttrs(b) &&
  sameHtmlStyle(a.htmlStyle, b.htmlStyle);

const coalesceLine = (line) => {
  if (line.length < 2) return line;
  const out = [];
  for (const tok of line) {
    const last = out.length ? out[out.length - 1] : null;
    if (last !== null && mergeable(last, tok)) {
      out[out.length - 1] = { ...last, content: last.content + tok.content };
      continue;
    }
    out.push(tok);
  }
  return out;
};

// All CommonMark fence forms, scanned line by line: an optional-closer regex double counts.
const OPEN_RE = /^ {0,3}(`{3,}|~{3,})([^\n]*)$/;

const readFences = (paths) => {
  const out = [];
  for (const path of paths) {
    const lines = readFileSync(path, "utf8").split("\n");
    let open = null;
    let body = [];
    for (const line of lines) {
      const m = OPEN_RE.exec(line);
      if (open === null) {
        if (m && !(m[1][0] === "`" && m[2].includes("`"))) {
          open = { marker: m[1], lang: m[2].trim().split(/\s+/)[0] || "text" };
          body = [];
        }
        continue;
      }
      if (m && m[1][0] === open.marker[0] && m[1].length >= open.marker.length
          && m[2].trim() === "") {
        out.push({ lang: open.lang, code: body.join("\n") });
        open = null;
        continue;
      }
      body.push(line);
    }
    // An unterminated fence stays open to the end of the document.
    if (open !== null) out.push({ lang: open.lang, code: body.join("\n") });
  }
  return out;
};

const ALIAS = {
  py: "python", js: "javascript", ts: "typescript", rs: "rust", rb: "ruby",
  sh: "shellscript", bash: "shellscript", zsh: "shellscript", shell: "shellscript",
  yml: "yaml", golang: "go", "c++": "cpp", "c#": "csharp", kt: "kotlin",
};

const argv = process.argv.slice(2);
const themeAt = argv.indexOf("--theme");
const mode = themeAt === -1 ? "dual" : argv[themeAt + 1];
const paths = argv.filter((a, i) => i !== themeAt && (themeAt === -1 || i !== themeAt + 1));
if (paths.length === 0) {
  console.error("usage: coal-span-census.mjs [--theme dual|dark|light] <markdown> [...]");
  process.exit(2);
}

const fences = readFences(paths);
if (fences.length === 0) {
  console.error("no fenced code found in those files");
  process.exit(1);
}

const engine = createJavaScriptRegexEngine({ forgiving: true });
const themeArg =
  mode === "dark" ? { theme: "unsloth-dark" }
    : mode === "light" ? { theme: "unsloth-light" }
      : { themes: { light: "unsloth-light", dark: "unsloth-dark" } };

let before = 0;
let after = 0;
let lines = 0;
let chars = 0;
const perLang = {};
const highlighters = new Map();

for (const f of fences) {
  const lang = ALIAS[f.lang.toLowerCase()] ?? f.lang.toLowerCase();
  let hl = highlighters.get(lang);
  if (!hl) {
    try {
      hl = await createHighlighter({ themes: [light, dark], langs: [lang], engine });
    } catch {
      hl = await createHighlighter({ themes: [light, dark], langs: ["text"], engine });
    }
    highlighters.set(lang, hl);
  }
  const use = hl.getLoadedLanguages().includes(lang) ? lang : "text";
  const res = hl.codeToTokens(f.code, { lang: use, ...themeArg });
  let b = 0;
  let a = 0;
  for (const line of res.tokens) {
    b += line.length;
    const c = coalesceLine(line);
    a += c.length;
    // If the text ever changes the census is meaningless, so stop.
    if (line.map((t) => t.content).join("") !== c.map((t) => t.content).join("")) {
      throw new Error(`TEXT CHANGED in a ${use} fence`);
    }
  }
  lines += res.tokens.length;
  chars += f.code.length;
  before += b;
  after += a;
  const k = perLang[use] ?? (perLang[use] = { fences: 0, before: 0, after: 0 });
  k.fences += 1;
  k.before += b;
  k.after += a;
}

const pct = (from, to) => `${(100 * (1 - to / from)).toFixed(1)}%`;
console.log(`theme mode        ${mode}`);
console.log(`fences            ${fences.length}`);
console.log(`code characters   ${chars}`);
console.log(`fence lines       ${lines}   (one <span> each, unchanged by the merge)`);
console.log(`token spans       ${before} -> ${after}   ${pct(before, after)} fewer`);
console.log(`total spans       ${before + lines} -> ${after + lines}   ${pct(before + lines, after + lines)} fewer`);
console.log("\nper language:");
for (const [k, v] of Object.entries(perLang).sort((x, y) => y[1].before - x[1].before)) {
  console.log(
    `  ${k.padEnd(14)} fences ${String(v.fences).padStart(3)}  ` +
      `${String(v.before).padStart(7)} -> ${String(v.after).padStart(7)}  ${pct(v.before, v.after)}`,
  );
}
