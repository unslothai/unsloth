// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CONVERSATION_MARKDOWN_EXTENSION,
  CONVERSATION_MARKDOWN_MIME_TYPE,
  type ConversationMarkdownMessage,
  buildConversationMarkdown,
} from "./conversation-markdown.ts";
import { stripSearchImageTokens } from "../search-images/search-images.ts";

type StoredConversationMessage = {
  readonly role: unknown;
};

type ConversationMarkdownExportDependencies<
  Message extends StoredConversationMessage,
> = {
  readonly loadMessages: (
    threadId: string,
  ) => Promise<readonly Message[] | null>;
  readonly renderMessage: (message: Message) => string;
  readonly download: (
    content: string,
    filename: string,
    mimeType: string,
  ) => Promise<void>;
  readonly exportTimestamp: () => string;
  readonly notifyNoContent: () => void;
};

export function createConversationMarkdownBuilder<
  Message extends StoredConversationMessage,
>({
  loadMessages,
  renderMessage,
}: Pick<
  ConversationMarkdownExportDependencies<Message>,
  "loadMessages" | "renderMessage"
>): (threadId: string) => Promise<string | null> {
  return async (threadId) => {
    const messages = await loadMessages(threadId);
    if (!messages) {
      return null;
    }
    const normalizedMessages: ConversationMarkdownMessage[] = messages.map(
      (message) => ({
        role: String(message.role ?? ""),
        // Strip renderer tokens here, not in the exporter, so the copy chord strips them too.
        content: stripSearchImageTokens(renderMessage(message)),
      }),
    );
    return buildConversationMarkdown(normalizedMessages, {
      includeImportMetadata: true,
    });
  };
}

function headingText(title: string): string {
  return title.replace(/\s+/g, " ").trim();
}

/** Named headings so the two halves of a compare row can be told apart. */
export async function buildNamedConversationsMarkdown(
  conversations: readonly { readonly id: string; readonly title: string }[],
  build: (threadId: string) => Promise<string | null>,
): Promise<string> {
  const parts: string[] = [];
  for (const conversation of conversations) {
    const markdown = await build(conversation.id);
    if (!markdown) continue;
    parts.push(
      conversations.length > 1
        ? `# ${headingText(conversation.title)}\n\n${markdown}`
        : markdown,
    );
  }
  return parts.join("\n---\n\n");
}

export function createConversationMarkdownExporter<
  Message extends StoredConversationMessage,
>({
  loadMessages,
  renderMessage,
  download,
  exportTimestamp,
  notifyNoContent,
}: ConversationMarkdownExportDependencies<Message>): (
  threadId: string,
) => Promise<void> {
  const build = createConversationMarkdownBuilder({
    loadMessages,
    renderMessage,
  });
  return async (threadId) => {
    const markdown = await build(threadId);
    if (markdown === null) return;
    if (!markdown) {
      notifyNoContent();
      return;
    }
    await download(
      markdown,
      `conversation-${exportTimestamp()}.${CONVERSATION_MARKDOWN_EXTENSION}`,
      CONVERSATION_MARKDOWN_MIME_TYPE,
    );
  };
}
