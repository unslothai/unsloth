// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// bundler-resolver's two rules, plus a redirect of the two backend modules to a stub.
// See delete-thread-message-stub.mjs for why the redirect is needed.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = fileURLToPath(new URL("../../src/", import.meta.url));
const STUB = new URL("./delete-thread-message-stub.mjs", import.meta.url).href;

// chat-api reaches the auth flow and the login page; chat-history-storage reaches chat-api.
const REDIRECTED =
  /(^|\/)(\.\.\/)*(api\/chat-api|utils\/chat-history-storage|chat-history-storage)(\.ts)?$/;

function firstExisting(base) {
  for (const candidate of [
    `${base}.ts`,
    `${base}.tsx`,
    `${base}/index.ts`,
    base,
  ]) {
    if (existsSync(candidate)) return pathToFileURL(candidate).href;
  }
  return null;
}

export function resolve(specifier, context, next) {
  if (REDIRECTED.test(specifier)) {
    return next(STUB, context);
  }
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(SRC + specifier.slice(2));
    return next(resolved ?? specifier, context);
  }
  if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
    const resolved = firstExisting(
      fileURLToPath(new URL(specifier, context.parentURL)),
    );
    if (resolved) return next(resolved, context);
  }
  return next(specifier, context);
}
