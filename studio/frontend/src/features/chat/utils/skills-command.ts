// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure helper for /skills: no imports, so the node tests load this file directly. */

const SKILLS_COMMAND = "/skills";

/** Exactly `/skills`, ignoring surrounding whitespace and case. */
export function isSkillsCommand(text: string): boolean {
  return text.trim().toLowerCase() === SKILLS_COMMAND;
}
