// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const sourceRoot = fileURLToPath(new URL("../src/", import.meta.url));
const authTestEntry = new URL("./auth-runtime-entry.mjs", import.meta.url).href;
const trackerFixture = new URL("./tracker-runtime-fixture.ts", import.meta.url)
  .href;
const executionWriteRaceFixture = new URL(
  "./execution-write-race-fixture.ts",
  import.meta.url,
).href;

function resolveSourcePath(candidate) {
  for (const file of [
    candidate,
    `${candidate}.ts`,
    `${candidate}.tsx`,
    path.join(candidate, "index.ts"),
    path.join(candidate, "index.tsx"),
  ]) {
    if (existsSync(file)) {
      return pathToFileURL(file).href;
    }
  }
  return null;
}

export async function resolve(specifier, context, nextResolve) {
  if (
    context.parentURL?.endsWith("/recipe-studio/data/executions-db.ts") &&
    (specifier === "@/features/auth" ||
      specifier === "@/features/user-assets" ||
      specifier === "@/features/user-assets/persistence-policy")
  ) {
    return { url: executionWriteRaceFixture, shortCircuit: true };
  }
  if (specifier === "@/shared/toast") {
    return { url: trackerFixture, shortCircuit: true };
  }
  if (
    specifier === "../api" &&
    context.parentURL?.endsWith("/recipe-studio/executions/tracker.ts")
  ) {
    return { url: trackerFixture, shortCircuit: true };
  }
  if (specifier === "@/features/auth") {
    return { url: authTestEntry, shortCircuit: true };
  }
  if (specifier.startsWith("@/")) {
    const url = resolveSourcePath(path.join(sourceRoot, specifier.slice(2)));
    if (url) return { url, shortCircuit: true };
  }
  if (
    specifier.startsWith(".") &&
    context.parentURL &&
    /\.(?:ts|tsx)$/.test(context.parentURL)
  ) {
    const url = resolveSourcePath(
      fileURLToPath(new URL(specifier, context.parentURL)),
    );
    if (url) return { url, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  const loaded = await nextLoad(url, context);
  if (!/\.(?:ts|tsx)$/.test(url) || loaded.source === undefined) return loaded;

  const source =
    typeof loaded.source === "string"
      ? loaded.source
      : Buffer.from(loaded.source).toString("utf8");
  const envValues = {
    BASE_URL: '"/"',
    DEV: "false",
    MODE: '"test"',
    PROD: "true",
  };
  return {
    ...loaded,
    source: source.replace(
      /import\.meta\.env\.([A-Z0-9_]+)/g,
      (_match, key) => envValues[key] ?? "undefined",
    ),
  };
}
