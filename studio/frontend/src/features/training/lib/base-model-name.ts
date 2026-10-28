// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A repo named as a base (pretrained, not chat-tuned) model: "Qwen3-4B-Base", "...-base-bnb-4bit".
 *  Unnamed base models are missed; the reasoning-format switch stays the user's either way. */
export function isBaseModelName(model: string | null | undefined): boolean {
  const name = (model ?? "").split("/").pop() ?? "";
  return /(?:^|[-_.])base(?:$|[-_.])/i.test(name);
}
