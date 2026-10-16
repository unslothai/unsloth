// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Runs the real scripts/typecheck.mjs in a temp checkout with a fake tsc and a fake bun,
 * and asserts which checker ran with which arguments. The failure that matters is
 * `bun check` spawned on a Bun without it: that is `bun run check`, a false pass or a
 * "Script not found".
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const SCRIPT = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "scripts",
  "typecheck.mjs",
);
const POSIX_ONLY = { skip: process.platform === "win32" };

const HELP_HEAD =
  "Bun is a fast JavaScript runtime, package manager, bundler, and test runner.";
const OLD_HELP = `${HELP_HEAD} (1.3.13+bf2e2cecf)

Usage: bun <command> [...flags] [...args]

Commands:
  run       ./my-script.ts       Execute a file with Bun
            lint                 Run a package.json script
  test                           Run unit tests with Bun
`;
const NEW_HELP = `${HELP_HEAD} (1.4.3+c6da4a4d3)

Usage: bun <command> [...flags] [...args]

Commands:
  run       ./my-script.ts       Execute a file with Bun
            lint                 Run a package.json script
  check                          Type check a TypeScript project
  test                           Run unit tests with Bun
`;
const COLOR_HELP = NEW_HELP.replace(
  "  check ",
  "  \x1b[1m\x1b[35mcheck\x1b[0m ",
).replace(/\n/g, "\r\n");

const TSC_ARGS = ["-b", "--pretty", "false"];
const TSC_TEST_ARGS = ["-p", "tsconfig.test.json", "--pretty", "false"];
const BUN_ARGS = ["check", "--build", "--no-pretty"];
const BUN_TEST_ARGS = ["check", "--project=tsconfig.test.json", "--no-pretty"];

const FAKE_TSC = `const { appendFileSync } = require("node:fs");
const { join } = require("node:path");
appendFileSync(join(__dirname, "..", "..", "..", "tsc.log"), JSON.stringify(process.argv.slice(2)) + "\\n");
process.exit(Number(process.env.FAKE_TSC_EXIT || 0));
`;

type Fixture = {
  root: string;
  env: NodeJS.ProcessEnv;
  tscCalls: () => string[][];
  bunCalls: () => string[][];
};

const roots: string[] = [];
test.after(() => {
  for (const root of roots) {
    rmSync(root, { recursive: true, force: true });
  }
});

function readLog(path: string): string[][] {
  if (!existsSync(path)) {
    return [];
  }
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as string[]);
}

function fixture(
  options: { bunHelp?: string; checkScript?: boolean } = {},
): Fixture {
  const root = mkdtempSync(join(tmpdir(), "typecheck-script-"));
  roots.push(root);
  mkdirSync(join(root, "scripts"));
  copyFileSync(SCRIPT, join(root, "scripts", "typecheck.mjs"));
  const scripts: Record<string, string> = { build: "vite build" };
  if (options.checkScript) {
    scripts.check = "echo hijacked";
  }
  writeFileSync(join(root, "package.json"), JSON.stringify({ scripts }));
  const ts = join(root, "node_modules", "typescript");
  mkdirSync(join(ts, "bin"), { recursive: true });
  writeFileSync(join(ts, "bin", "tsc"), FAKE_TSC);
  writeFileSync(join(ts, "package.json"), '{"version":"7.0.2"}');
  mkdirSync(join(root, "bin"));
  if (options.bunHelp !== undefined) {
    writeFileSync(join(root, "help.txt"), options.bunHelp);
    const bun = join(root, "bin", "bun");
    // A node script, not sh: PATH holds only bin/ and node's own directory, which has no
    // coreutils under setup-node, nvm and similar installs.
    writeFileSync(
      bun,
      `#!/usr/bin/env node
const { appendFileSync, readFileSync } = require("node:fs");
const { join } = require("node:path");
const here = join(__dirname, "..");
const args = process.argv.slice(2);
appendFileSync(join(here, "bun.log"), JSON.stringify(args) + "\\n");
if (args[0] === "--help") {
  process.stdout.write(readFileSync(join(here, "help.txt")));
  process.exit(0);
}
process.exit(Number(process.env.FAKE_BUN_EXIT || 0));
`,
    );
    chmodSync(bun, 0o755);
  }
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const key of Object.keys(env)) {
    const upper = key.toUpperCase();
    if (
      upper === "PATH" ||
      upper === "UNSLOTH_TYPECHECK" ||
      upper === "FORCE_COLOR"
    ) {
      delete env[key];
    }
  }
  env.PATH = join(root, "bin") + delimiter + dirname(process.execPath);
  return {
    root,
    env,
    tscCalls: () => readLog(join(root, "tsc.log")),
    bunCalls: () => readLog(join(root, "bun.log")),
  };
}

function run(
  fx: Fixture,
  args: string[] = [],
  env: Record<string, string> = {},
) {
  const r = spawnSync(
    process.execPath,
    [join(fx.root, "scripts", "typecheck.mjs"), ...args],
    { encoding: "utf8", env: { ...fx.env, ...env }, timeout: 30_000 },
  );
  return { status: r.status, stderr: r.stderr };
}

test("without bun it runs tsc -b, then the test project with --tests", () => {
  const fx = fixture();
  const r = run(fx, ["--tests"]);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /typecheck: tsc \(TypeScript 7\.0\.2\)/);
  assert.deepEqual(fx.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);
});

test("build mode runs only tsc -b", () => {
  const fx = fixture();
  assert.equal(run(fx).status, 0);
  assert.deepEqual(fx.tscCalls(), [TSC_ARGS]);
});

test("a failing first step stops the second and passes its exit code through", () => {
  const fx = fixture();
  const r = run(fx, ["--tests"], { FAKE_TSC_EXIT: "2" });
  assert.equal(r.status, 2);
  assert.deepEqual(fx.tscCalls(), [TSC_ARGS]);
});

test(
  "Bun 1.3 (no check command) falls back to tsc and never spawns check",
  POSIX_ONLY,
  () => {
    const fx = fixture({ bunHelp: OLD_HELP });
    const r = run(fx, ["--tests"]);
    assert.equal(r.status, 0, r.stderr);
    assert.deepEqual(fx.bunCalls(), [["--help"]]);
    assert.deepEqual(fx.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);
  },
);

test(
  "Bun 1.4.3 runs bun check for both projects and not tsc",
  POSIX_ONLY,
  () => {
    const fx = fixture({ bunHelp: NEW_HELP });
    const r = run(fx, ["--tests"]);
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stderr, /typecheck: bun check \(Bun 1\.4\.3\)/);
    assert.deepEqual(fx.bunCalls(), [["--help"], BUN_ARGS, BUN_TEST_ARGS]);
    assert.deepEqual(fx.tscCalls(), []);
  },
);

test("colored CRLF help still counts as having check", POSIX_ONLY, () => {
  const fx = fixture({ bunHelp: COLOR_HELP });
  const r = run(fx);
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(fx.bunCalls(), [["--help"], BUN_ARGS]);
  assert.deepEqual(fx.tscCalls(), []);
});

test("a check package script keeps bun off the path", POSIX_ONLY, () => {
  const fx = fixture({ bunHelp: NEW_HELP, checkScript: true });
  const r = run(fx, ["--tests"]);
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(fx.bunCalls(), []);
  assert.deepEqual(fx.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);
});

test(
  "auto: a red bun check is confirmed by tsc, whose verdict wins",
  POSIX_ONLY,
  () => {
    const green = fixture({ bunHelp: NEW_HELP });
    const r = run(green, ["--tests"], { FAKE_BUN_EXIT: "1" });
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stderr, /bun check reported errors; confirming with tsc/);
    assert.deepEqual(green.bunCalls(), [["--help"], BUN_ARGS]);
    assert.deepEqual(green.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);

    const red = fixture({ bunHelp: NEW_HELP });
    const rr = run(red, ["--tests"], {
      FAKE_BUN_EXIT: "1",
      FAKE_TSC_EXIT: "1",
    });
    assert.equal(rr.status, 1);
    assert.deepEqual(red.tscCalls(), [TSC_ARGS]);
  },
);

test("UNSLOTH_TYPECHECK=tsc never probes bun", POSIX_ONLY, () => {
  const fx = fixture({ bunHelp: NEW_HELP });
  const r = run(fx, ["--tests"], { UNSLOTH_TYPECHECK: "tsc" });
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(fx.bunCalls(), []);
  assert.deepEqual(fx.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);
});

test("UNSLOTH_TYPECHECK=bun without bun fails loudly", () => {
  const fx = fixture();
  const r = run(fx, ["--tests"], { UNSLOTH_TYPECHECK: "bun" });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /no Bun with a `check` command/);
  assert.deepEqual(fx.tscCalls(), []);
});

test(
  "UNSLOTH_TYPECHECK=bun with Bun 1.3 fails without spawning check",
  POSIX_ONLY,
  () => {
    const fx = fixture({ bunHelp: OLD_HELP });
    const r = run(fx, [], { UNSLOTH_TYPECHECK: "bun" });
    assert.equal(r.status, 1);
    assert.deepEqual(fx.bunCalls(), [["--help"]]);
    assert.deepEqual(fx.tscCalls(), []);
  },
);

test(
  "UNSLOTH_TYPECHECK=bun with a check script fails and runs nothing",
  POSIX_ONLY,
  () => {
    const fx = fixture({ bunHelp: NEW_HELP, checkScript: true });
    const r = run(fx, [], { UNSLOTH_TYPECHECK: "bun" });
    assert.equal(r.status, 1);
    assert.match(r.stderr, /`check` script/);
    assert.deepEqual(fx.bunCalls(), []);
    assert.deepEqual(fx.tscCalls(), []);
  },
);

test("UNSLOTH_TYPECHECK=bun passes bun's exit code through", POSIX_ONLY, () => {
  const fx = fixture({ bunHelp: NEW_HELP });
  const r = run(fx, ["--tests"], {
    UNSLOTH_TYPECHECK: "bun",
    FAKE_BUN_EXIT: "1",
  });
  assert.equal(r.status, 1);
  assert.deepEqual(fx.bunCalls(), [["--help"], BUN_ARGS]);
  assert.deepEqual(fx.tscCalls(), []);
});

test("an unknown UNSLOTH_TYPECHECK value exits 2", () => {
  const fx = fixture();
  const r = run(fx, [], { UNSLOTH_TYPECHECK: "nonsense" });
  assert.equal(r.status, 2);
  assert.match(r.stderr, /unknown UNSLOTH_TYPECHECK=nonsense/);
  assert.deepEqual(fx.tscCalls(), []);
});

test(
  "Windows: an npm bun.cmd shim is never executed",
  { skip: process.platform !== "win32" },
  () => {
    const fx = fixture();
    const marker = join(fx.root, "shim-ran");
    writeFileSync(join(fx.root, "help.txt"), NEW_HELP);
    writeFileSync(
      join(fx.root, "bin", "bun.cmd"),
      `@echo off\r\necho ran> "${marker}"\r\ntype "${join(fx.root, "help.txt")}"\r\n`,
    );
    const r = run(fx, ["--tests"]);
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stderr, /typecheck: tsc/);
    assert.equal(existsSync(marker), false);
    assert.deepEqual(fx.tscCalls(), [TSC_ARGS, TSC_TEST_ARGS]);
  },
);
