// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Type-checks the frontend with `bun check` when a Bun that has it is on PATH, else with the
// pinned TypeScript compiler. Usage: node scripts/typecheck.mjs [--tests]
// UNSLOTH_TYPECHECK=auto (default) | tsc | bun. Bun is used only when the TypeScript version its
// `check` follows (process.versions.typescript) is the one the project pins, so both apply the
// same rules. In auto mode a red Bun result is also re-checked with tsc.

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const withTests = process.argv.slice(2).includes("--tests");
const tscPath = join(root, "node_modules", "typescript", "bin", "tsc");
// biome-ignore lint/suspicious/noControlCharactersInRegex: matches ANSI escapes
const ansi = /\x1b\[[0-?]*[ -/]*[@-~]/g;

const tscSteps = [["-b", "--pretty", "false"]];
// The `check` subcommand with long flags. `bun --check` is Bun's global flag: it ignores
// --build/--project/--no-pretty, and there -b means --bun and -p means --print. `--pretty false`
// would be read as a script name.
const bunSteps = [["check", "--build", "--no-pretty"]];
if (withTests) {
  tscSteps.push(["-p", "tsconfig.test.json", "--pretty", "false"]);
  bunSteps.push(["check", "--project=tsconfig.test.json", "--no-pretty"]);
}

function fail(message, code = 1) {
  console.error(`typecheck: ${message}`);
  process.exit(code);
}

// `bun check` runs a package script named `check` instead of checking, if one exists.
// An unreadable package.json counts as having one.
function hasCheckScript() {
  try {
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    return Object.hasOwn(pkg?.scripts ?? {}, "check");
  } catch {
    return true;
  }
}

// Output of a short bun command, or null when bun is missing or the command fails.
function bunOutput(args) {
  const probe = spawnSync("bun", args, {
    cwd: root,
    encoding: "utf8",
    // biome-ignore lint/style/useNamingConvention: environment variable names
    env: { ...process.env, FORCE_COLOR: "0", NO_COLOR: "1" },
    shell: false,
    stdio: ["ignore", "pipe", "ignore"],
    timeout: 5000,
    windowsHide: true,
  });
  if (probe.error || probe.status !== 0 || typeof probe.stdout !== "string") {
    return null;
  }
  return probe.stdout.replace(ansi, "").replace(/\r/g, "");
}

// Returns the Bun version string ("" if unknown) when `bun check` exists, else null.
// Bun before 1.4.3 has no `check` command and would treat it as `bun run check`.
function probeBun() {
  const help = bunOutput(["--help"]);
  if (help === null || !/^ {2}check\s/m.test(help)) {
    return null;
  }
  return help.match(/\((\d+\.\d+\.\d+)[^)]*\)/)?.[1] ?? "";
}

// The TypeScript version `bun check` follows, "" if Bun does not say.
function bunTypeScript() {
  return bunOutput(["-p", "process.versions.typescript"])?.trim() ?? "";
}

function tsVersion() {
  try {
    const pkg = JSON.parse(
      readFileSync(
        join(root, "node_modules", "typescript", "package.json"),
        "utf8",
      ),
    );
    return typeof pkg.version === "string" ? pkg.version : "";
  } catch {
    return "";
  }
}

function run(command, steps) {
  for (const args of steps) {
    const result = spawnSync(command, args, {
      cwd: root,
      shell: false,
      stdio: "inherit",
      windowsHide: true,
    });
    if (result.error) {
      console.error(
        `typecheck: could not run ${command}: ${result.error.message}`,
      );
      return 1;
    }
    if (result.status !== 0) {
      return result.status ?? 1;
    }
  }
  return 0;
}

function runTsc() {
  const version = tsVersion();
  console.error(`typecheck: tsc${version ? ` (TypeScript ${version})` : ""}`);
  return run(
    process.execPath,
    tscSteps.map((args) => [tscPath, ...args]),
  );
}

function runBun(version) {
  console.error(`typecheck: bun check${version ? ` (Bun ${version})` : ""}`);
  return run("bun", bunSteps);
}

const mode = process.env.UNSLOTH_TYPECHECK || "auto";

if (mode === "tsc") {
  process.exit(runTsc());
}
if (mode === "bun") {
  if (hasCheckScript()) {
    fail(
      "UNSLOTH_TYPECHECK=bun, but package.json has a `check` script that `bun check` would run",
    );
  }
  const version = probeBun();
  if (version === null) {
    fail(
      "UNSLOTH_TYPECHECK=bun, but no Bun with a `check` command (Bun >= 1.4.3) is on PATH",
    );
  }
  const bunTs = bunTypeScript();
  if (bunTs !== tsVersion()) {
    fail(
      `UNSLOTH_TYPECHECK=bun, but Bun ${version} follows TypeScript ${bunTs || "unknown"} and the project pins ${tsVersion() || "unknown"}`,
    );
  }
  process.exit(runBun(version));
}
if (mode !== "auto") {
  fail(`unknown UNSLOTH_TYPECHECK=${mode} (expected auto, tsc or bun)`, 2);
}

let bunVersion = hasCheckScript() ? null : probeBun();
if (bunVersion !== null) {
  const bunTs = bunTypeScript();
  const projectTs = tsVersion();
  if (!bunTs || bunTs !== projectTs) {
    console.error(
      `typecheck: Bun ${bunVersion} follows TypeScript ${bunTs || "unknown"}, not the pinned ${projectTs || "unknown"}; using tsc`,
    );
    bunVersion = null;
  }
}
if (bunVersion !== null && runBun(bunVersion) === 0) {
  process.exit(0);
}
if (bunVersion !== null) {
  console.error("typecheck: bun check reported errors; confirming with tsc");
}
process.exit(runTsc());
