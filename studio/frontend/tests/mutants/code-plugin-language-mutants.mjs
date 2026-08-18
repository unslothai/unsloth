// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One mutant per assertion in tests/code-plugin-language-fallback.test.ts.
//
// Same contract as code-token-cache-mutants.mjs: each entry breaks one property of the language
// resolution Studio took over from `@streamdown/code`, and NAMES the test that must go red. The
// run fails if that test stays green, or if any test in the file is nobody's named victim.
//
// It MUTATES THE WORKING TREE and restores the file afterwards, so run it in a scratch checkout.
// Never point it at a tree a benchmark is running in. It is not wired into `npm test` or into CI.
//
//   node tests/mutants/code-plugin-language-mutants.mjs

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, "..", "..");
const source = path.join(frontend, "src/components/assistant-ui/code-plugin.ts");
const testFile = path.join(frontend, "tests/code-plugin-language-fallback.test.ts");

const MUTANTS = [
  {
    // `createHighlighter` REJECTS on a grammar it does not have, and the rejection is swallowed by
    // the catch, so without the fallback an unknown fence tag is silently never highlighted.
    name: "unknown tags are not folded to plaintext",
    test: "an unknown fence tag still produces tokens instead of an unstyled block",
    from: "  return BUNDLED_LANGUAGE_IDS.has(resolved)\n    ? (resolved as BundledLanguage)\n    : \"text\";",
    to: "  return resolved as BundledLanguage;",
  },
  {
    // Folding EVERYTHING to plaintext still returns tokens, so only an assertion that the grammar
    // actually ran can see this one.
    name: "every tag is folded to plaintext",
    test: "a Shiki alias resolves to the same grammar as its canonical id",
    from: "  const resolved = SHIKI_ALIASES[normalized] ?? normalized;",
    to: "  const resolved = \"definitely-not-a-grammar\";",
  },
];

const original = readFileSync(source, "utf8");

function failingTests() {
  try {
    execFileSync(
      process.execPath,
      ["--experimental-strip-types", "--test", "--test-reporter=tap", testFile],
      { cwd: frontend, encoding: "utf8", stdio: "pipe" },
    );
    return [];
  } catch (error) {
    const out = `${error.stdout ?? ""}${error.stderr ?? ""}`;
    return [...out.matchAll(/^not ok \d+ - (.+)$/gm)].map((m) => m[1].trim());
  }
}

let bad = 0;
const killed = new Set();
try {
  const clean = failingTests();
  if (clean.length > 0) {
    console.error(`unmutated tree is already failing: ${clean.join(", ")}`);
    process.exit(1);
  }
  for (const mutant of MUTANTS) {
    if (!original.includes(mutant.from)) {
      console.error(`MUTANT NOT APPLICABLE: ${mutant.name} (anchor text is gone)`);
      bad += 1;
      continue;
    }
    writeFileSync(source, original.replace(mutant.from, mutant.to), "utf8");
    const failed = failingTests();
    if (!failed.includes(mutant.test)) {
      console.error(`SURVIVED: ${mutant.name} -> "${mutant.test}" stayed green`);
      bad += 1;
    } else {
      killed.add(mutant.test);
      const collateral = failed.filter((name) => name !== mutant.test);
      const extra = collateral.length > 0 ? ` (also reddened ${collateral.length})` : "";
      console.log(`killed: ${mutant.name} -> "${mutant.test}"${extra}`);
    }
  }
} finally {
  writeFileSync(source, original, "utf8");
}

const declared = [
  ...readFileSync(testFile, "utf8").matchAll(/^test\("(.+?)", async \(\) => \{$/gm),
].map((m) => m[1]);
const unguarded = declared.filter((name) => !killed.has(name));
if (unguarded.length > 0) {
  console.error(`no mutant targets: ${unguarded.join(", ")}`);
  bad += unguarded.length;
}
if (bad > 0) {
  console.error(`${bad} problem(s)`);
  process.exit(1);
}
console.log(
  `all ${MUTANTS.length} mutants killed, and all ${declared.length} tests are somebody's target`,
);
