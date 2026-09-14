// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "./conversation-markdown";

/** One assistant reply as the markdown its saved source holds. The reply's own reasoning, tool calls
 *  and citations are part of what a "save this reply" click asks to keep, and getCopyText()
 *  concatenates text parts alone, so they would be dropped and a reply that is only a tool call
 *  would read as empty. Same conversion the whole-chat save runs. */
export function replySourceMarkdown(
  content: unknown,
  normalizeToolResult?: (result: unknown, toolName?: string) => unknown,
): string {
  return renderConversationBlocks(
    contentBlocksToMarkdownBlocks(content, normalizeToolResult),
  );
}
