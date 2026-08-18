#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const manifest = JSON.parse(
  readFileSync(resolve(ROOT, "docs/rag-platform/phase-15-e2e-evidence.json"), "utf8"),
);
const failures = [];
const ids = manifest.scenarios.map((scenario) => scenario.id);
if (manifest.requiredScenarioCount !== 24 || ids.length !== 24) {
  failures.push("the mandatory E2E manifest must contain exactly 24 scenarios");
}
if (new Set(ids).size !== 24 || ids.some((id, index) => id !== index + 1)) {
  failures.push("scenario ids must be unique and contiguous from 1 to 24");
}
for (const scenario of manifest.scenarios) {
  if (!scenario.evidence?.length) failures.push(`scenario ${scenario.id} has no evidence`);
  for (const path of scenario.evidence ?? []) {
    if (!existsSync(resolve(ROOT, path))) failures.push(`scenario ${scenario.id} missing ${path}`);
  }
}
for (const path of manifest.runtimeSuites ?? []) {
  if (!existsSync(resolve(ROOT, path))) failures.push(`runtime suite missing ${path}`);
}
if (failures.length) {
  for (const failure of failures) console.error(`E2E evidence gate: ${failure}`);
  process.exit(1);
}
console.log("E2E evidence gate PASS (24 mandatory scenarios)");
