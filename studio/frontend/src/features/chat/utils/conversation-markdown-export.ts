// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CONVERSATION_MARKDOWN_EXTENSION,
  CONVERSATION_MARKDOWN_MIME_TYPE,
  type ConversationMarkdownMessage,
  buildConversationMarkdown,
} from "./conversation-markdown.ts";

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
  return async (threadId) => {
    const messages = await loadMessages(threadId);
    if (!messages) {
      return;
    }
    const normalizedMessages: ConversationMarkdownMessage[] = messages.map(
      (message) => ({
        role: String(message.role ?? ""),
        content: renderMessage(message),
      }),
    );
    const markdown = buildConversationMarkdown(normalizedMessages);
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
