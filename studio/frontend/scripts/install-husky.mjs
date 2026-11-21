// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run Husky from the git root, not this subdirectory.
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, "..");
const root = path.resolve(frontend, "..", "..");
const huskyEntry = path.join(frontend, "node_modules", "husky", "index.js");

// Skip when the install is not a Git working tree.
const probe = spawnSync("git", ["rev-parse", "--is-inside-work-tree"], {
  cwd: root,
  stdio: "ignore",
});
if (probe.status !== 0) process.exit(0);

const installer = `import(${JSON.stringify(
  pathToFileURL(huskyEntry).href,
)}).then(({ default: install }) => {
  const message = install();
  if (message) {
    console.error(message);
    process.exitCode = 1;
  }
})`;

const result = spawnSync(
  process.execPath,
  ["--input-type=module", "-e", installer],
  {
    cwd: root,
    stdio: "inherit",
  },
);
process.exit(result.status ?? 1);
