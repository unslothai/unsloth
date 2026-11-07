// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Repo-wide comment CLI: --write joins wrapped comments, --check reports them.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { joinWrappedComments } from "./comments.ts";

const CODE = /\.(?:ts|tsx|js|jsx|mjs|cjs|mts|cts)$/;
const args = process.argv.slice(2);
const write = args.includes("--write");
const explicit = args.filter((arg) => !arg.startsWith("--"));

const root = execFileSync("git", ["rev-parse", "--show-toplevel"], {
  encoding: "utf8",
}).trim();

const files =
  explicit.length > 0
    ? explicit.map((file) => path.resolve(root, file))
    : execFileSync("git", ["ls-files", "-z"], { cwd: root, encoding: "utf8" })
        .split("\0")
        .filter((file) => CODE.test(file))
        .map((file) => path.join(root, file));

let joined = 0;
const offenders = [];

for (const file of files) {
  const relative = path.relative(root, file).replaceAll("\\", "/");
  if (relative.startsWith("studio/backend/assets/docs_ui/")) continue;
  const original = readFileSync(file, "utf8");
  const result = joinWrappedComments(original);

  if (write && result.joined > 0) {
    writeFileSync(file, result.text, "utf8");
    joined += result.joined;
  } else if (!write && result.joined > 0) {
    joined += result.joined;
    offenders.push(`${relative}: ${result.joined} wrapped comment(s)`);
  }
}

console.log(
  `${files.length} file(s) scanned, ${joined} wrapped comment(s) ${write ? "joined" : "found"}`,
);
if (offenders.length > 0) console.log(offenders.join("\n"));
if (!write && joined > 0) process.exit(1);
