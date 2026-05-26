#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const frontendDir = path.dirname(fileURLToPath(import.meta.url));
const studioFrontendDir = path.resolve(frontendDir, "..");
const studioDir = path.resolve(studioFrontendDir, "..");
const backendRunPy = path.join(studioDir, "backend", "run.py");

const managedPython =
  process.platform === "win32"
    ? path.join(homedir(), ".unsloth", "studio", "unsloth_studio", "Scripts", "python.exe")
    : path.join(homedir(), ".unsloth", "studio", "unsloth_studio", "bin", "python");

const python =
  process.env.UNSLOTH_STUDIO_PYTHON ||
  (existsSync(managedPython) ? managedPython : "python3");
const backendPort = process.env.UNSLOTH_STUDIO_BACKEND_PORT || "8899";
const backendUrl =
  process.env.VITE_STUDIO_API_BASE || `http://127.0.0.1:${backendPort}`;

function spawnChecked(label, command, args, options = {}) {
  const child = spawn(command, args, {
    stdio: "inherit",
    shell: process.platform === "win32",
    ...options,
  });
  child.on("error", (error) => {
    console.error(`[${label}] failed to start: ${error.message}`);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (shuttingDown) return;
    if (signal) {
      console.error(`[${label}] exited from signal ${signal}`);
    } else if (code !== 0) {
      console.error(`[${label}] exited with code ${code}`);
    }
    shutdown(code ?? 1);
  });
  return child;
}

let shuttingDown = false;
let backend;
let vite;

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of [vite, backend]) {
    if (child && child.exitCode == null && child.signalCode == null) {
      child.kill(process.platform === "win32" ? undefined : "SIGTERM");
    }
  }
  setTimeout(() => process.exit(code), 150).unref();
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

console.log(`[studio-dev] backend: ${backendUrl}`);
console.log(`[studio-dev] python: ${python}`);

backend = spawnChecked("backend", python, [
  backendRunPy,
  "--host",
  "127.0.0.1",
  "--port",
  backendPort,
  "--api-only",
]);

vite = spawnChecked(
  "vite",
  process.platform === "win32" ? "npx.cmd" : "npx",
  ["vite"],
  {
    cwd: studioFrontendDir,
    env: {
      ...process.env,
      VITE_STUDIO_API_BASE: backendUrl,
    },
  },
);
