// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Drafts src/components/update/releases/<tag>/notes.json from a published GitHub release,
// so the app's What's New reads the same highlights as the release page.
// Usage: node scripts/draft-release-notes.mjs v0.1.923-beta [--force]
// The draft has no feature cards: add one or two with artwork, then trim to the brief in
// releases/README.md.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const RELEASES =
  process.env.RELEASE_NOTES_DIR ??
  join(
    dirname(fileURLToPath(import.meta.url)),
    "../src/components/update/releases",
  );
const REPO = "unslothai/unsloth";
const MAX_CHANGES = 4;
const SKIPPED_SECTIONS =
  /^(Highlights|What's Changed|New Contributors|Thanks|Contributors|Download)/i;

const [tag, flag] = process.argv.slice(2);
if (!tag?.startsWith("v")) {
  console.error(
    "usage: draft-release-notes.mjs <tag, e.g. v0.1.923-beta> [--force]",
  );
  process.exit(2);
}
const out = join(RELEASES, tag, "notes.json");
if (existsSync(out) && flag !== "--force") {
  console.error(`${out} exists; pass --force to overwrite`);
  process.exit(1);
}

const headers = new Headers({ accept: "application/vnd.github+json" });
if (process.env.GITHUB_TOKEN) {
  headers.set("authorization", `Bearer ${process.env.GITHUB_TOKEN}`);
}
const res = await fetch(
  `https://api.github.com/repos/${REPO}/releases/tags/${tag}`,
  {
    headers,
  },
);
if (!res.ok) {
  console.error(`GitHub answered ${res.status} for ${tag}`);
  process.exit(1);
}
const release = await res.json();
const body = String(release.body ?? "").replaceAll("\r\n", "\n");

function section(name) {
  const match = new RegExp(
    `^## ${name}\\n([\\s\\S]*?)(?=^## |(?![\\s\\S]))`,
    "m",
  ).exec(body);
  return match ? match[1] : "";
}

// "- **Label** - text", "- **Label:** text" and "- **Label.** text" all become [label, text].
function highlight(line) {
  const match = /^- \*\*(.+?)[:.]?\*\*[:.]?\s*(?:-\s*)?(.+)$/.exec(line.trim());
  if (!match) {
    return null;
  }
  const text = match[2]
    // Trailing "[Guide](...)"-style links point outside the app; other links keep their text.
    .replace(
      /\s*[-(]?\s*\[(?:See )?(?:Guide|Blog|Analysis|GGUF|FP8)\]\([^)]+\)\)?/gi,
      "",
    )
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\s+/g, " ")
    .replace(/[\s-]+$/, "")
    .trim();
  const sentence = text.charAt(0).toUpperCase() + text.slice(1);
  return [match[1].trim(), /[.!?]$/.test(sentence) ? sentence : `${sentence}.`];
}

const changes = section("Highlights")
  .split("\n")
  .map(highlight)
  .filter(Boolean)
  .slice(0, MAX_CHANGES);

// The rest: merged PRs under What's Changed, else bullets in the themed sections.
let total = (section("What's Changed").match(/^\* /gm) ?? []).length;
if (total === 0) {
  for (const part of body.split(/^## /m).slice(1)) {
    if (!SKIPPED_SECTIONS.test(part)) {
      total += (part.match(/^- /gm) ?? []).length;
    }
  }
}

const notes = {
  version: tag.replace(/^v/, ""),
  title: String(release.name ?? tag).replace(/\s*\+\s*/g, " and "),
  date: String(release.published_at ?? "").slice(0, 10),
  features: [],
  changes,
  more: Math.max(0, total - changes.length),
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${JSON.stringify(notes, null, 2)}\n`);

const index = join(RELEASES, "index.ts");
const source = readFileSync(index, "utf8");
const name = `release${notes.version.replace(/\D/g, "")}`;
if (!source.includes(`./${tag}/notes.json`)) {
  const withImport = source.replace(
    /^(import release\d+ from "\.\/v[^"]+\/notes\.json";\n)/m,
    `import ${name} from "./${tag}/notes.json";\n$1`,
  );
  writeFileSync(
    index,
    withImport.replace(
      "export const releases = [\n",
      `export const releases = [\n  ${name},\n`,
    ),
  );
}
console.log(`wrote ${out} (${changes.length} highlights, ${notes.more} more)`);
