// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rejected /validate preflights never reach loadAttempts, so a separate budget counts dead-end
// preflights (refusals and rejections), not passing ones. Loop conditions are lifted from source.

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

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const VALIDATE_CAP_RE = /const (MAX_AUTO_VALIDATE_[A-Z_]+) = (\d+);/;
const LOAD_CAP_RE = /const MAX_AUTO_LOAD_ATTEMPTS = (\d+);/;

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

function braced(from: number, what: string): string {
  const open = source.indexOf("{", from);
  assert.ok(open >= 0, `no block found for ${what}`);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "{") {
      depth += 1;
    } else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(open + 1, i);
      }
    }
  }
  throw new Error(`unbalanced block for ${what}`);
}

/** Function body at `from`; skips a `{` opened by a return type annotation like Promise<{...}>. */
function functionBody(from: number, what: string): string {
  for (
    let i = source.indexOf("{", from);
    i >= 0;
    i = source.indexOf("{", i + 1)
  ) {
    const before = source.slice(0, i).trimEnd().slice(-1);
    if (before === ")" || before === ">") {
      return braced(i, what);
    }
  }
  throw new Error(`no body found for ${what}`);
}

const loadCapMatch = LOAD_CAP_RE.exec(source);
assert.ok(loadCapMatch, "MAX_AUTO_LOAD_ATTEMPTS is no longer defined");
const MAX_AUTO_LOAD_ATTEMPTS = Number(loadCapMatch[1]);

// A missing cap is the defect, reported as an unbounded sweep; the cap name is read from source too.
const validateCapMatch = VALIDATE_CAP_RE.exec(source);
const VALIDATE_CAP_NAME =
  validateCapMatch?.[1] ?? "MAX_AUTO_VALIDATE_UNDEFINED";
const MAX_AUTO_VALIDATE_FAILURES = validateCapMatch
  ? Number(validateCapMatch[2])
  : Number.POSITIVE_INFINITY;

const cascadeStart = source.indexOf("async function autoLoadSmallestModel(");
assert.ok(cascadeStart >= 0, "autoLoadSmallestModel is no longer defined");
const cascadeBody = functionBody(cascadeStart, "autoLoadSmallestModel");

const counterMatch = new RegExp(
  `(\\w+)\\s*(?:>=|<)\\s*${VALIDATE_CAP_NAME}`,
).exec(source);
const COUNTER_NAME = counterMatch?.[1] ?? "validateFailures";

type CascadeState = {
  autoLoadCancelled: boolean;
  loadAttempts: number;
  validateFailures: number;
};

function predicate(condition: string): (state: CascadeState) => boolean {
  const compiled = new Function(
    "MAX_AUTO_LOAD_ATTEMPTS",
    VALIDATE_CAP_NAME,
    "autoLoadCancelled",
    "loadAttempts",
    COUNTER_NAME,
    `return Boolean(${condition});`,
  ) as (...args: unknown[]) => boolean;
  return (state) =>
    compiled(
      MAX_AUTO_LOAD_ATTEMPTS,
      MAX_AUTO_VALIDATE_FAILURES,
      state.autoLoadCancelled,
      state.loadAttempts,
      state.validateFailures,
    );
}

const sourceLoopStart = source.indexOf("for (const source of sources) {");
assert.ok(sourceLoopStart >= 0, "the cascade's source loop is no longer there");
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

type Outcome =
  | "refuse" // requires_trust_remote_code / security review / transformers upgrade
  | "reject" // validateModel threw: dead backend, transport failure
  | "load-fails"
  | "loads";

type CascadeResult = {
  validates: number;
  loads: number;
  loaded: boolean;
  validateFailures: number;
};

type SweepStatus = "next-source" | "aborted" | "loaded";

type ValidateBudget = { validates: number; abortAt: number };

function sweepSource(
  quants: readonly Outcome[],
  state: CascadeState,
  budget: ValidateBudget,
): SweepStatus {
  let next = 0;
  while (keepTryingCandidates(state) && next < quants.length) {
    const outcome = quants[next];
    next += 1;
    if (budget.validates >= budget.abortAt) {
      return "aborted";
    }
    budget.validates += 1;
    if (outcome === "refuse" || outcome === "reject") {
      state.validateFailures += 1;
      continue;
    }
    state.loadAttempts += 1;
    if (outcome === "loads") {
      return "loaded";
    }
  }
  return "next-source";
}

function runCascade(
  inventory: readonly (readonly Outcome[])[],
  options?: { cancelAfterValidates?: number },
): CascadeResult {
  const state: CascadeState = {
    autoLoadCancelled: false,
    loadAttempts: 0,
    validateFailures: 0,
  };
  const budget: ValidateBudget = {
    validates: 0,
    abortAt: options?.cancelAfterValidates ?? Number.POSITIVE_INFINITY,
  };
  let loaded = false;
  for (const quants of inventory) {
    if (stopSweep(state)) {
      break;
    }
    const status = sweepSource(quants, state, budget);
    loaded = status === "loaded";
    if (status !== "next-source") {
      break;
    }
  }
  return {
    validates: budget.validates,
    loads: state.loadAttempts,
    loaded,
    validateFailures: state.validateFailures,
  };
}

const repeat = (outcome: Outcome, n: number): Outcome[][] =>
  Array.from({ length: n }, () => [outcome]);

const OLD_EVERY_VALIDATE_CAP = 8;

test("a cache of refused models cannot POST /validate once per repo", () => {
  for (const size of [5, 40, 500, 5000]) {
    const cascade = runCascade(repeat("refuse", size));
    assert.equal(cascade.loads, 0, "a refused preflight never reaches /load");
    assert.ok(
      size <= MAX_AUTO_VALIDATE_FAILURES || cascade.validates < size,
      `the cascade issued one /validate per cached repo (${cascade.validates} of ` +
        `${size}), so nothing bounds it but the size of the inventory`,
    );
    assert.equal(
      cascade.validates,
      Math.min(size, MAX_AUTO_VALIDATE_FAILURES),
      `${size} refused repos should cost at most the budget, not ${cascade.validates}`,
    );
  }
});

test("a dead backend cannot POST /validate once per repo either", () => {
  for (const size of [5, 40, 500, 5000]) {
    const cascade = runCascade(repeat("reject", size));
    assert.equal(cascade.loads, 0);
    assert.ok(
      cascade.validates <= MAX_AUTO_VALIDATE_FAILURES,
      `${cascade.validates} /validate calls against a dead backend, over the ` +
        `${MAX_AUTO_VALIDATE_FAILURES} budget`,
    );
  }
});

test("refusals and rejections share one budget", () => {
  const mixed: Outcome[][] = Array.from({ length: 500 }, (_, i) => [
    i % 2 === 0 ? "refuse" : "reject",
  ]);
  const cascade = runCascade(mixed);
  assert.equal(cascade.validates, MAX_AUTO_VALIDATE_FAILURES);
});

test("a loadable model behind many refusals still loads", () => {
  assert.ok(
    MAX_AUTO_VALIDATE_FAILURES > OLD_EVERY_VALIDATE_CAP,
    `the budget (${MAX_AUTO_VALIDATE_FAILURES}) must clear the old every-validate cap ` +
      `(${OLD_EVERY_VALIDATE_CAP}) or the happy path is truncated exactly as before`,
  );
  for (let refusals = 0; refusals < MAX_AUTO_VALIDATE_FAILURES; refusals += 1) {
    const cascade = runCascade([...repeat("refuse", refusals), ["loads"]]);
    assert.ok(
      cascade.loaded,
      `a loadable model behind ${refusals} refusals must still load`,
    );
    assert.equal(cascade.loads, 1, "and it costs exactly one load attempt");
    assert.equal(
      cascade.validates,
      refusals + 1,
      "and one preflight per candidate reached",
    );
  }
});

test("the budget cuts in one candidate past its last refusal", () => {
  const cap = MAX_AUTO_VALIDATE_FAILURES;
  for (const refusals of [cap - 1, cap, cap + 1]) {
    const cascade = runCascade([...repeat("refuse", refusals), ["loads"]]);
    assert.equal(
      cascade.loaded,
      refusals < cap,
      `with ${refusals} refusals ahead of it the model should ` +
        `${refusals < cap ? "load" : "not be reached"}`,
    );
    assert.equal(
      cascade.validates,
      Math.min(refusals, cap) + (refusals < cap ? 1 : 0),
    );
  }
});

test("a passing preflight spends a load attempt, not the failure budget", () => {
  const cascade = runCascade(repeat("load-fails", 500));
  assert.equal(cascade.validateFailures, 0, "a pass costs no failure budget");
  assert.equal(
    cascade.loads,
    MAX_AUTO_LOAD_ATTEMPTS,
    "but it does spend a load attempt, so the load cap still bounds it",
  );
  assert.equal(cascade.validates, MAX_AUTO_LOAD_ATTEMPTS);
});

test("the two budgets compose rather than share", () => {
  const worst = [
    ...repeat("refuse", MAX_AUTO_VALIDATE_FAILURES - 1),
    ...repeat("load-fails", 500),
  ];
  const cascade = runCascade(worst);
  assert.equal(
    cascade.validates,
    MAX_AUTO_VALIDATE_FAILURES - 1 + MAX_AUTO_LOAD_ATTEMPTS,
  );
});

test("a repo's other quants each cost a preflight, and are bounded too", () => {
  const cascade = runCascade([repeat("refuse", 500).flat()]);
  assert.equal(cascade.validates, MAX_AUTO_VALIDATE_FAILURES);
});

test("cancellation stops the sweep before the next POST", () => {
  const cascade = runCascade(repeat("refuse", 500), {
    cancelAfterValidates: 4,
  });
  assert.equal(cascade.validates, 4, "the aborted candidate sends nothing");
  assert.ok(cascade.validates < MAX_AUTO_VALIDATE_FAILURES);
});

test("the budget is per cascade, not module-global", () => {
  // A module-level counter would leave a second auto-load with a spent budget.
  assert.doesNotMatch(source, new RegExp(`^let ${COUNTER_NAME} = 0;$`, "m"));
  assert.ok(
    cascadeBody.includes(`let ${COUNTER_NAME} = 0;`),
    `${COUNTER_NAME} is not declared inside the cascade`,
  );
  const first = runCascade(repeat("refuse", 500));
  assert.equal(first.validates, MAX_AUTO_VALIDATE_FAILURES);
  const second = runCascade([
    ...repeat("refuse", MAX_AUTO_VALIDATE_FAILURES - 1),
    ["loads"],
  ]);
  assert.ok(second.loaded, "the second cascade gets a full budget");
});

test("every dead-ended /validate is counted, and no passing one is", () => {
  const guard = source.indexOf("async function canAutoLoad(");
  assert.ok(guard >= 0, "canAutoLoad is no longer defined");
  const body = functionBody(guard, "canAutoLoad");
  const post = body.indexOf("await validateModel(");
  assert.ok(post >= 0, "canAutoLoad no longer POSTs /validate");
  const increment = `${COUNTER_NAME} += 1;`;

  // Counting before the POST would also charge passing preflights, truncating the happy path.
  assert.ok(
    !body.slice(0, post).includes(increment),
    "canAutoLoad must not count the preflight before it knows the outcome",
  );

  const refusalExits = body.split("return false;").slice(0, -1);
  assert.ok(
    refusalExits.length >= 2,
    "the refusal branches are no longer there",
  );
  for (const [index, before] of refusalExits.entries()) {
    assert.ok(
      before.slice(-160).includes(increment),
      `refusal branch ${index} returns without counting its /validate`,
    );
  }
  const rethrow = body.indexOf("throw error;", post);
  assert.ok(rethrow > post, "the rejected preflight is no longer rethrown");
  assert.ok(
    body.slice(post, rethrow).includes(increment),
    "a rejected /validate must spend the budget: the sweep continues after a transport " +
      "failure, so a dead backend would otherwise POST once per cached repo",
  );

  assert.equal(
    body.split(increment).length - 1,
    3,
    "a new branch out of canAutoLoad must decide, explicitly, whether it spends the budget",
  );
});

test("the default-download preflight stays outside the budget", () => {
  // The last preflight is deliberately ungated so an all-refused cache still falls back.
  const loopEnd = source.indexOf(
    "// The cap gates the default download too",
    sourceLoopStart,
  );
  assert.ok(loopEnd > sourceLoopStart, "the sweep's tail comment moved");
  const defaultPreflight = source.indexOf(
    "model_path: DEFAULT_CHAT_MODEL_REPO,",
    loopEnd,
  );
  assert.ok(defaultPreflight > loopEnd, "the default preflight moved");
  assert.ok(
    !source.slice(loopEnd, defaultPreflight).includes(VALIDATE_CAP_NAME),
    "the default-download preflight must not be gated on the validate budget",
  );
});
