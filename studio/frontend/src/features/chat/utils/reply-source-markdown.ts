// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "./conversation-markdown";

/** Includes reasoning, tool calls and citations, which getCopyText() would drop. */
export function replySourceMarkdown(
  content: unknown,
  normalizeToolResult?: (result: unknown, toolName?: string) => unknown,
): string {
  return renderConversationBlocks(
    contentBlocksToMarkdownBlocks(content, normalizeToolResult),
  );
}
