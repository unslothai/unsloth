// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function resolvePython(): string {
  if (process.env.E2E_PYTHON) {
    return process.env.E2E_PYTHON;
  }
  const candidates = [
    path.resolve(__dirname, "../../backend/.venv/bin/python3"),
    path.resolve(
      process.env.HOME ?? "",
      ".unsloth/studio/unsloth_studio/bin/python3",
    ),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return "python3";
}

export default function globalSetup(): void {
  const script = path.resolve(__dirname, "scripts/issue-tokens.py");
  const python = resolvePython();
  const output = execSync(`"${python}" "${script}"`, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  }).trim();
  const tokens = JSON.parse(output) as {
    access_token: string;
    refresh_token: string;
  };
  process.env.E2E_ACCESS_TOKEN = tokens.access_token;
  process.env.E2E_REFRESH_TOKEN = tokens.refresh_token;
}
