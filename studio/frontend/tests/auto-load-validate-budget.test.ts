// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The auto-load cascade preflights every candidate with a POST to /validate, and a candidate the
// preflight refuses (trust-remote-code, security review, a transformers upgrade) returns before
// loadAttempts is incremented. So MAX_AUTO_LOAD_ATTEMPTS bounds /load only: on a device whose cached
// repos are all refused the sweep walked the whole inventory, one /validate per repo, and never hit
// a cap. These drive the cascade's own loop conditions, lifted from the shipped source the way
// tests/auto-load-target-key.test.ts lifts normalizeTarget, so a guard that stops naming the
// validate budget goes red here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

/** The text inside the parentheses that open at or after `from`. */
function parenthesized(from: number, what: string): string {
  const open = source.indexOf("(", from);
  assert.ok(open >= 0, `no condition found for ${what}`);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "(") {
      depth += 1;
    } else if (source[i] === ")") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(open + 1, i);
      }
    }
  }
  throw new Error(`unbalanced condition for ${what}`);
}

function numericConstant(name: string): number | null {
  const match = new RegExp(`const ${name} = (\\d+);`).exec(source);
  return match ? Number(match[1]) : null;
}

const loadCap = numericConstant("MAX_AUTO_LOAD_ATTEMPTS");
assert.ok(loadCap, "MAX_AUTO_LOAD_ATTEMPTS is no longer defined");
const MAX_AUTO_LOAD_ATTEMPTS = loadCap;
// A missing cap is not a broken test: it is the defect, and the replay below reports it as the
// unbounded sweep it is rather than as a failed lift.
const MAX_AUTO_VALIDATE_ATTEMPTS =
  numericConstant("MAX_AUTO_VALIDATE_ATTEMPTS") ?? Number.POSITIVE_INFINITY;

type CascadeState = {
  autoLoadCancelled: boolean;
  loadAttempts: number;
  validateAttempts: number;
};

/** Compile one lifted loop condition into a predicate over the counters it reads. */
function predicate(condition: string): (state: CascadeState) => boolean {
  const compiled = new Function(
    "MAX_AUTO_LOAD_ATTEMPTS",
    "MAX_AUTO_VALIDATE_ATTEMPTS",
    "autoLoadCancelled",
    "loadAttempts",
    "validateAttempts",
    `return Boolean(${condition});`,
  ) as (...args: unknown[]) => boolean;
  return (state) =>
    compiled(
      MAX_AUTO_LOAD_ATTEMPTS,
      MAX_AUTO_VALIDATE_ATTEMPTS,
      state.autoLoadCancelled,
      state.loadAttempts,
      state.validateAttempts,
    );
}

const sourceLoopStart = source.indexOf("for (const source of sources) {");
assert.ok(sourceLoopStart >= 0, "the cascade's source loop is no longer there");
// The `if (...) break;` that opens the body, not the `for` header itself.
const stopSweep = predicate(
  parenthesized(
    source.indexOf("if (", sourceLoopStart),
    "the source loop's stop guard",
  ),
);
const keepTryingCandidates = predicate(
  parenthesized(
    source.indexOf("while (", sourceLoopStart),
    "the per-source candidate loop",
  ),
);

/**
 * Replay the cascade over `candidates` cached repos. Each source yields one candidate, whose
 * preflight POSTs /validate once; a refused one is marked skipped, so the next pass of the inner
 * loop resolves nothing and the sweep moves to the next source. Returns the /validate and /load
 * counts a browser would issue.
 */
function runCascade(
  candidates: number,
  preflight: (index: number) => boolean,
): { validates: number; loads: number } {
  const state: CascadeState = {
    autoLoadCancelled: false,
    loadAttempts: 0,
    validateAttempts: 0,
  };
  let validates = 0;
  for (let index = 0; index < candidates; index += 1) {
    if (stopSweep(state)) {
      break;
    }
    // One candidate per source, then resolveAutoLoadCandidate returns null and the loop ends.
    if (!keepTryingCandidates(state)) {
      break;
    }
    validates += 1;
    state.validateAttempts += 1;
    if (!preflight(index)) {
      continue;
    }
    state.loadAttempts += 1;
    return { validates, loads: state.loadAttempts };
  }
  return { validates, loads: state.loadAttempts };
}

const REFUSED_REPOS = 40;
// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const MODULE_LEVEL_COUNTER = /^let validateAttempts = 0;$/m;

test("a cache full of refused models cannot POST /validate once per repo", () => {
  const cascade = runCascade(REFUSED_REPOS, () => false);
  assert.equal(cascade.loads, 0, "a refused preflight never reaches /load");
  assert.ok(
    cascade.validates < REFUSED_REPOS,
    `the cascade issued one /validate per cached repo (${cascade.validates} of ` +
      `${REFUSED_REPOS}), so nothing bounds it but the size of the inventory`,
  );
  assert.ok(
    cascade.validates <= MAX_AUTO_VALIDATE_ATTEMPTS,
    `the cascade issued ${cascade.validates} /validate calls, over the ` +
      `${MAX_AUTO_VALIDATE_ATTEMPTS} budget`,
  );
});

test("the validate budget still leaves room for the whole load budget", () => {
  // A refused candidate must keep costing no load attempt, so the validate cap has to clear the
  // load cap: every successful load spends a validate of its own first.
  assert.ok(MAX_AUTO_VALIDATE_ATTEMPTS >= MAX_AUTO_LOAD_ATTEMPTS);
  const cascade = runCascade(REFUSED_REPOS, (index) => index === 0);
  assert.equal(cascade.loads, 1, "the first loadable model still loads");
  assert.equal(cascade.validates, 1, "and it costs exactly one preflight");
});

test("refusals ahead of a loadable model do not consume its load attempt", () => {
  const refusalsFirst = MAX_AUTO_VALIDATE_ATTEMPTS - 1;
  const cascade = runCascade(REFUSED_REPOS, (index) => index === refusalsFirst);
  assert.equal(
    cascade.loads,
    1,
    "a model reached inside the validate budget still loads",
  );
});

test("the validate budget is per cascade, not module-global", () => {
  // A counter beside MAX_AUTO_VALIDATE_ATTEMPTS would leave the session's second auto-load with a
  // spent budget and no way to ever load anything.
  assert.doesNotMatch(source, MODULE_LEVEL_COUNTER);
  const cascadeStart = source.indexOf("async function autoLoadSmallestModel(");
  assert.ok(cascadeStart >= 0, "autoLoadSmallestModel is no longer defined");
  const declaration = source.indexOf("let validateAttempts = 0;", cascadeStart);
  assert.ok(
    declaration > cascadeStart,
    "validateAttempts is not declared inside the cascade",
  );
});

test("every /validate the cascade sends is counted", () => {
  // Counting at the call sites instead of at the request would miss whichever branch is added next.
  const guard = source.indexOf("async function canAutoLoad(");
  assert.ok(guard >= 0, "canAutoLoad is no longer defined");
  const post = source.indexOf("await validateModel(", guard);
  const counted = source.indexOf("validateAttempts += 1;", guard);
  assert.ok(
    counted >= 0 && counted < post,
    "canAutoLoad must count the attempt before it POSTs /validate",
  );
});
