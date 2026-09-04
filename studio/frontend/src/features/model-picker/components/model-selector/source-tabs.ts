// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local models (LM Studio, Ollama, custom folders) are not fine-tuned; they live in the Hub
 *  picker's On Device section. */
export function isFineTunedSource(source?: string): boolean {
  return source !== "local";
}
