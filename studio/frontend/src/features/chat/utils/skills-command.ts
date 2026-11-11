// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `/skills` opens the Skills library without sending a prompt to the model. */
export function isSkillsCommand(text: string): boolean {
  return text.trim().toLowerCase() === "/skills";
}
