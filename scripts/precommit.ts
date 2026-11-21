#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Staged-only commit gate: format, lint, comment policy, typecheck.

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import {
  hasFileSummary,
  joinWrappedComments,
  splitLines,
} from "./quality/comments.ts";

const CODE = /\.(?:ts|tsx|js|jsx|mjs|cjs|mts|cts)$/;

const ROOT = execFileSync("git", ["rev-parse", "--show-toplevel"], {
  encoding: "utf8",
}).trim();
process.chdir(ROOT);
const FRONTEND = path.join(ROOT, "studio", "frontend");

const useColor =
  process.env.FORCE_COLOR !== "0" &&
  (process.stdout.isTTY || Boolean(process.env.CLICOLOR_FORCE));
const paint = (code, text) =>
  useColor ? `\u001b[${code}m${text}\u001b[0m` : text;
const green = (text) => paint("32", text);
const red = (text) => paint("31", text);
const yellow = (text) => paint("33", text);
const dim = (text) => paint("2", text);
const bold = (text) => paint("1", text);

function git(args) {
  return execFileSync("git", args, {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  });
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
  return {
    status: result.status ?? 1,
    output: `${result.stdout ?? ""}${result.stderr ?? ""}`.trim(),
  };
}

function stage(files) {
  if (files.length > 0) git(["add", "--", ...files]);
}

const stagedFiles = git([
  "diff",
  "--cached",
  "--name-only",
  "--diff-filter=ACMR",
])
  .split(/\r?\n/)
  .filter(Boolean);
const codeFiles = stagedFiles.filter((file) => CODE.test(file));

if (codeFiles.length === 0) {
  console.log(dim("pre-commit: no staged JavaScript/TypeScript changes"));
  process.exit(0);
}

const added = new Set(
  git(["diff", "--cached", "--name-only", "--diff-filter=A"])
    .split(/\r?\n/)
    .filter(Boolean),
);

const checks = [];
const record = (name, ok, detail) => checks.push({ name, ok, detail });

// Format with Bun before linting so the fixers and the formatter agree.
const bunVersion = run("bun", ["--version"]);
if (bunVersion.status !== 0) {
  record(
    "Format (bun fmt)",
    false,
    "Bun is required for the commit gate. Install Bun from https://bun.com and retry.",
  );
} else {
  const generated = [/model-catalog-snapshot\.ts$/];
  const formatFiles = codeFiles.filter(
    (file) => !generated.some((pattern) => pattern.test(file)),
  );
  const config = path.join(FRONTEND, "bunfig.toml");
  const formatted = run("bun", [
    "fmt",
    ...(existsSync(config) ? [`--config=${config}`] : []),
    "--no-error-on-unmatched-pattern",
    ...formatFiles,
  ]);
  stage(codeFiles);
  record(
    "Format (bun fmt)",
    formatted.status === 0,
    formatted.status === 0
      ? `${codeFiles.length} file(s) · bun ${bunVersion.output}`
      : formatted.output,
  );
}

// Lint each staged file with the config that owns its directory.
const eslintBin = path.join(
  FRONTEND,
  "node_modules",
  "eslint",
  "bin",
  "eslint.js",
);
const frontendFiles = codeFiles
  .filter((file) => file.startsWith("studio/frontend/"))
  .map((file) => path.relative(FRONTEND, path.join(ROOT, file)));
const externalFiles = codeFiles.filter(
  (file) => !file.startsWith("studio/frontend/"),
);

function lint(files, cwd, extraArgs = []) {
  if (files.length === 0) return { ok: true, output: "no matching files" };
  const result = run(
    "bun",
    [eslintBin, ...extraArgs, "--fix", "--max-warnings=0", "--", ...files],
    { cwd },
  );
  return { ok: result.status === 0, output: result.output };
}

const frontendLint = lint(frontendFiles, FRONTEND);
stage(frontendFiles.map((file) => path.posix.join("studio/frontend", file)));
record(
  "ESLint (frontend, typed)",
  frontendLint.ok,
  frontendLint.ok ? `${frontendFiles.length} file(s)` : frontendLint.output,
);

const externalLint = lint(externalFiles, ROOT, [
  "--config",
  path.join(FRONTEND, "eslint.external.config.js"),
]);
stage(externalFiles);
record(
  "ESLint (repo-wide)",
  externalLint.ok,
  externalLint.ok ? `${externalFiles.length} file(s)` : externalLint.output,
);

// Comment and file-size policy.
// Only lines added by this commit are checked for trailing comments.
const addedLines = new Map();
const diffText = git(["diff", "--cached", "-U0", "--", ...codeFiles]);
let currentFile = null;
let newLine = 0;
for (const raw of diffText.split(/\r?\n/)) {
  if (raw.startsWith("+++ ")) {
    currentFile = raw.slice(4).replace(/^b\//, "");
    continue;
  }
  const hunk = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(raw);
  if (hunk) {
    newLine = Number(hunk[1]);
    continue;
  }
  if (currentFile === null || raw.length === 0) continue;
  if (raw.startsWith("+")) {
    addedLines.set(currentFile, [
      ...(addedLines.get(currentFile) ?? []),
      { line: newLine, text: raw.slice(1) },
    ]);
    newLine += 1;
  } else if (raw.startsWith(" ")) {
    newLine += 1;
  }
}

const policy = [];
let joinedComments = 0;
for (const file of codeFiles) {
  const absolute = path.join(ROOT, file);
  if (!existsSync(absolute)) continue;

  const original = readFileSync(absolute, "utf8");
  const joined = joinWrappedComments(original);
  if (joined.joined > 0) {
    writeFileSync(absolute, joined.text, "utf8");
    joinedComments += joined.joined;
  }

  const lines = splitLines(joined.text);

  const introducedItems = addedLines.get(file) ?? [];
  let commentRun = 0;
  let previousLine = -2;
  for (const item of introducedItems) {
    const contiguous = item.line === previousLine + 1;
    previousLine = item.line;
    if (/^\s*\/\//.test(item.text)) {
      commentRun = contiguous ? commentRun + 1 : 1;
      if (commentRun > 2) {
        policy.push(
          `${yellow(file)}  comment runs over two lines; keep one short line.`,
        );
        break;
      }
    } else {
      commentRun = 0;
    }
  }
  const introducedLines = introducedItems.map((item) => item.text);
  if (introducedLines.some((line) => /^\s*\/\/\s*[-=*#]{6,}/.test(line))) {
    policy.push(`${yellow(file)}  banner comment; remove the separator.`);
  }
  if (added.has(file) && !hasFileSummary(lines)) {
    policy.push(
      `${yellow(file)}  add a one-line // summary under the license header.`,
    );
  }
}
stage(codeFiles);
record(
  "Comments",
  policy.length === 0,
  policy.length === 0
    ? `${joinedComments} wrapped comment(s) joined`
    : policy.slice(0, 10).join("\n"),
);

// Typecheck the frontend project, including the test project.
const touchesFrontend = codeFiles.some((file) =>
  file.startsWith("studio/frontend/"),
);
const typecheck = touchesFrontend
  ? run("bun", [path.join(FRONTEND, "scripts", "typecheck.mjs"), "--tests"], {
      cwd: FRONTEND,
    })
  : { status: 0, output: "skipped: no frontend files staged" };
record(
  "Typecheck",
  typecheck.status === 0,
  typecheck.status === 0 ? typecheck.output || "src + tests" : typecheck.output,
);

// Report.
console.log(`\n${bold("Commit gate")}`);
for (const check of checks) {
  const mark = check.ok ? green("PASS") : red("FAIL");
  const label = check.name.padEnd(26);
  if (check.ok) {
    console.log(`  ${mark}  ${label}${dim(check.detail ?? "")}`);
  } else {
    console.log(`  ${mark}  ${red(label)}`);
    for (const line of String(check.detail ?? "")
      .split(/\r?\n/)
      .slice(0, 12)) {
      console.log(`        ${line}`);
    }
  }
}

const failed = checks.filter((check) => !check.ok);
if (failed.length > 0) {
  console.log(
    `${red(bold(`Commit blocked by ${failed.length} check(s).`))}  ${dim("Fix the items above and commit again.")}`,
  );
  process.exit(1);
}
console.log(
  green(bold("All checks passed.")) + dim("  Staged changes are clean."),
);
