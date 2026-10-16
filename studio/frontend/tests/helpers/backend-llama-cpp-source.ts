// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/** The backend's `core/inference/llama_cpp` package as one string, modules in name order.
 *
 * It used to be a single `llama_cpp.py`; the parity tests here read Python literals out of
 * it. Joining every module keeps those reads working wherever a constant now lives. The
 * package `__init__` is left out: it only re-exports, so its name list would be the first
 * hit for any scan that looks a constant up by name. */
export function readLlamaCppPackage(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const dir = path.join(here, "../../../backend/core/inference/llama_cpp");
  return readdirSync(dir)
    .filter((name) => name.endsWith(".py") && name !== "__init__.py")
    .sort()
    .map((name) => readFileSync(path.join(dir, name), "utf8"))
    .join("\n");
}
