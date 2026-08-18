// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One mutant per assertion in tests/code-token-cache.test.ts.
//
// A test suite that stays green when the thing it guards is broken is worse than no suite, and a
// suite-wide "some test failed" does not prove that MY test caught it. So each entry below breaks
// exactly one property of the cache and NAMES the test that must go red; the run fails if that
// test stays green, and it also fails if any test in the file is nobody's named victim, which is
// what a vacuous test looks like. Collateral failures are printed but not failed on: breaking a
// primitive that every test reads through reddens everything and says nothing either way.
//
// It MUTATES THE WORKING TREE and restores the file after every mutant, so run it in a scratch
// checkout. Never point it at a tree a benchmark is running in. It is not wired into `npm test`
// or into CI: it only runs when somebody invokes it.
//
//   node tests/mutants/code-token-cache-mutants.mjs

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, "..", "..");
const source = path.join(frontend, "src/components/assistant-ui/code-token-cache.ts");
const testFile = path.join(frontend, "tests/code-token-cache.test.ts");

const MUTANTS = [
  {
    name: "get never finds anything",
    test: "a stored tokenisation is served back for the same source",
    from: "      const entry = entries.get(key);\n      if (!entry) return null;",
    to: "      const entry = undefined as TokenCacheEntry<T> | undefined;\n      if (!entry) return null;",
  },
  {
    name: "no prefix eviction",
    test: "a streamed fence occupies one entry, not one per refresh window",
    from: "        if (code.startsWith(other.code)) {\n          drop(otherKey, other);\n        }",
    to: "        if (false) {\n          drop(otherKey, other);\n        }",
  },
  {
    // The shipped bug: evicting in BOTH directions live-locks two fences that are on screen at
    // once when one extends the other.
    name: "prefix eviction runs in both directions",
    test: "two fences on screen at once, one extending the other, both stay cached",
    from: "        if (code.startsWith(other.code)) {",
    to: "        if (code.startsWith(other.code) || other.code.startsWith(code)) {",
  },
  {
    name: "prefix eviction ignores the group",
    test: "prefix eviction does not reach across groups",
    from: "        if (other.group !== group) continue;",
    to: "        if (false) continue;",
  },
  {
    name: "no character budget",
    test: "the character budget bounds what the cache holds",
    from: "      (chars > options.maxChars && entries.size > 1)",
    to: "      (false && entries.size > 1)",
  },
  {
    name: "no entry count cap",
    test: "the entry count is capped for a thread of tiny fences",
    from: "      entries.size > options.maxEntries ||",
    to: "      false ||",
  },
  {
    name: "reads do not refresh recency",
    test: "reading an entry protects it from the next eviction",
    from: "      entries.delete(key);\n      entries.set(key, entry);\n      return entry.result;",
    to: "      return entry.result;",
  },
  {
    name: "dropping an entry forgets to refund its characters",
    test: "the accounted character total tracks what is actually held",
    from: "    entries.delete(key);\n    chars -= entry.code.length;",
    to: "    entries.delete(key);",
  },
];

const original = readFileSync(source, "utf8");

// node --test prints "not ok N - <name>" per failing test, so the failures can be attributed.
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
    const caught = failed.includes(mutant.test);
    const collateral = failed.filter((name) => name !== mutant.test);
    if (!caught) {
      console.error(`SURVIVED: ${mutant.name} -> "${mutant.test}" stayed green`);
      bad += 1;
    } else {
      killed.add(mutant.test);
      // Collateral is reported, not failed on. Breaking a primitive every test reads through
      // reddens every test, which says nothing about whether the named one is vacuous; what says
      // that is the coverage check below, that EVERY test is somebody's named victim.
      const extra = collateral.length > 0 ? ` (also reddened ${collateral.length})` : "";
      console.log(`killed: ${mutant.name} -> "${mutant.test}"${extra}`);
    }
  }
} finally {
  writeFileSync(source, original, "utf8");
}

// A test nobody breaks is a test that proves nothing, so the names in the file have to be
// covered by the mutant list, not just the other way round.
const declared = [...readFileSync(testFile, "utf8").matchAll(/^test\("(.+?)", \(\) => \{$/gm)].map(
  (m) => m[1],
);
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
