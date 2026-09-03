// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Would merging adjacent same-styled Shiki tokens remove any spans?
 *
 * WHY THIS EXISTS. The renderer emits one <span> per themed token, so a thread's span census is
 * its token census, and Shiki splits on GRAMMAR boundaries rather than on rendered appearance:
 * `foo`, `.`, `bar` in a member expression look like three tokens carrying one identical colour.
 * Merging runs whose rendered style is byte-identical therefore looks like a pure reduction with
 * no viewport gating, no state machine and nothing a reader can do to undo it.
 *
 * It removes nothing. Over the whole studiobench corpus, 728 fences and 1,335,897 code
 * characters in typescript, go, rust, python and c:
 *
 *   dual        tokens 537013 -> merged 537013   0.0% fewer
 *   darkonly    tokens 535981 -> merged 535981   0.0% fewer
 *
 * and over the 100K rung's 99 assembled fences, 180,902 characters, 72,550 -> 72,550 dual,
 * 72,408 -> 72,408 dark only, 62,098 -> 62,098 light only. Not one adjacent pair in half a million
 * shares a rendered style, in any theme mode. Shiki already emits maximally coalesced tokens.
 *
 * This script is kept so the result is checkable rather than quoted, and so the next person to
 * have the idea can spend a minute on it instead of a day.
 *
 * USAGE. Point it at any markdown that contains fenced code; a saved thread export, a docs page,
 * a scratch file. It extracts the fences itself.
 *
 *   node studio/frontend/scripts/coal-span-census.mjs thread.md [more.md ...]
 *   node studio/frontend/scripts/coal-span-census.mjs --theme dark thread.md
 *
 * The configuration below is the one `code-plugin.ts` uses: the JavaScript regex engine, one-light
 * and one-dark-pro with transparent backgrounds, dual-theme `themes:` mode.
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

// The merge predicate. `htmlAttrs` blocks a merge outright rather than being reconciled: it can
// carry ids, titles or data attributes something downstream depends on, and two spans that differ
// there are not interchangeable however identical they look.
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

// Every fence form CommonMark allows, not just the unindented triple backtick: three or more
// backticks or tildes, up to three spaces of indent, closed by at least as many of the same
// character. Scanned line by line rather than by one regex, because a regex that treats the
// closing delimiter as optional will also match every CLOSING line as a fresh opener and double
// the fence count, which is what the first version of this did.
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
      // A closer is the same character, at least as long, and carries nothing else.
      if (m && m[1][0] === open.marker[0] && m[1].length >= open.marker.length
          && m[2].trim() === "") {
        out.push({ lang: open.lang, code: body.join("\n") });
        open = null;
        continue;
      }
      body.push(line);
    }
    // Markdown leaves an unterminated fence open to the end of the document, and so does a
    // thread that was still streaming when it was saved.
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
    // The whole point is that the text is untouched. If it ever is not, the census is meaningless
    // and the run should stop rather than print a number.
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
