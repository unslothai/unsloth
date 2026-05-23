// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  AssistantRuntimeProvider,
  type AttachmentAdapter,
  type ChatModelAdapter,
  type CompleteAttachment,
  CompositeAttachmentAdapter,
  ExportedMessageRepository,
  type ExportedMessageRepositoryItem,
  type LocalRuntimeOptions,
  type PendingAttachment,
  type ThreadHistoryAdapter,
  type ThreadMessage,
  WebSpeechDictationAdapter,
  type unstable_RemoteThreadListAdapter,
  useAui,
  useAuiEvent,
  useAuiState,
  useLocalRuntime,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import mammoth from "mammoth";
import {
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { extractText, getDocumentProxy } from "unpdf";
import { toast } from "sonner";
import { createOpenAIStreamAdapter } from "./api/chat-adapter";
import {
  loadConnectionsEnabled,
  loadExternalProviders,
  parseExternalModelId,
  providerTypeSupportsVision,
} from "./external-providers";
import {
  OPEN_DOCUMENT_SPREADSHEET_MIME,
  OPEN_DOCUMENT_TEXT_MIME,
  type OpenDocumentAttachmentContent,
  readActiveOpenDocumentAttachmentContent,
  readOpenDocumentAttachmentContent,
} from "./open-document";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import type { MessageRecord, ModelType, ThreadRecord } from "./types";
import {
  deleteStoredChatThreads,
  ensureStoredChatThread,
  getStoredChatThread,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  listStoredChatThreads,
  saveStoredChatMessage,
  saveStoredChatThread,
  updateStoredChatThread,
} from "./utils/chat-history-storage";
import { isChatThreadDeleted } from "./utils/chat-thread-tombstones";
import { syncExportedRepositoryToBackend } from "./utils/delete-thread-message";
import { getImageInputUnavailableReason } from "./utils/image-input-support";

const pendingHistoryAppendByMessageId = new Map<string, Promise<void>>();

type TitleResponse = {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
};

class VisionImageAdapter implements AttachmentAdapter {
  accept = "image/jpeg,image/png,image/webp,image/gif";

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const state = useChatRuntimeStore.getState();
    const checkpoint = state.params.checkpoint;
    const activeModel = state.models.find((m) => m.id === checkpoint);
    const externalSelection = parseExternalModelId(checkpoint);
    const isExternalModel = externalSelection !== null;
    const modelLoaded = !!checkpoint && !state.modelLoading;
    let externalSupportsVision: boolean | null = null;
    let externalModelLabel: string | null = null;
    if (externalSelection !== null) {
      const providers = loadConnectionsEnabled() ? loadExternalProviders() : [];
      const provider = providers.find(
        (p) => p.id === externalSelection.providerId,
      );
      externalSupportsVision = providerTypeSupportsVision(
        provider?.providerType,
      );
      externalModelLabel = externalSelection.modelId;
    }
    const unavailableReason = getImageInputUnavailableReason({
      activeModel,
      isExternalModel,
      externalSupportsVision,
      externalModelLabel,
      loadedIsMultimodal: state.loadedIsMultimodal,
      modelLoaded,
    });
    if (unavailableReason) {
      toast.error(unavailableReason);
      throw new Error(unavailableReason);
    }

    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
      throw new Error("Image size exceeds 20MB limit");
    }

    return {
      id: crypto.randomUUID(),
      type: "image",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    return {
      id: attachment.id,
      type: "image",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [
        {
          type: "image",
          image: await this.fileToBase64DataURL(attachment.file),
        },
      ],
      status: { type: "complete" },
    };
  }

  async remove(): Promise<void> {
    return Promise.resolve();
  }

  private async fileToBase64DataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(new Error("Failed to read image file"));
      reader.readAsDataURL(file);
    });
  }
}

class PDFAttachmentAdapter implements AttachmentAdapter {
  accept = "application/pdf";

  add({ file }: { file: File }): Promise<PendingAttachment> {
    return Promise.resolve({
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    });
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const buffer = new Uint8Array(await attachment.file.arrayBuffer());
    const pdf = await getDocumentProxy(buffer);
    const { text } = await extractText(pdf, { mergePages: true });
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[PDF: ${attachment.name}]\n${text}` }],
      status: { type: "complete" },
    };
  }

  remove(): Promise<void> {
    return Promise.resolve();
  }
}

class TextAttachmentAdapter implements AttachmentAdapter {
  accept = "text/plain,text/markdown,text/csv,text/xml,text/json,text/css";

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    return {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const text = await attachment.file.text();
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [
        {
          type: "text",
          text: `<attachment name=${attachment.name}>\n${text}\n</attachment>`,
        },
      ],
      status: { type: "complete" },
    };
  }

  remove(): Promise<void> {
    return Promise.resolve();
  }
}

class HtmlAttachmentAdapter implements AttachmentAdapter {
  accept = "text/html";

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    return {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const html = await attachment.file.text();
    // Strip HTML tags to extract readable text
    const doc = new DOMParser().parseFromString(html, "text/html");
    // Remove script and style elements
    for (const el of doc.querySelectorAll("script, style")) el.remove();
    const text = (doc.body.textContent ?? "").replace(/\s+/g, " ").trim();
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[HTML: ${attachment.name}]\n${text}` }],
      status: { type: "complete" },
    };
  }

  remove(): Promise<void> {
    return Promise.resolve();
  }
}

class DocxAttachmentAdapter implements AttachmentAdapter {
  accept =
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

  add({ file }: { file: File }): Promise<PendingAttachment> {
    return Promise.resolve({
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    });
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const arrayBuffer = await attachment.file.arrayBuffer();
    const { value } = await mammoth.extractRawText({ arrayBuffer });
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[DOCX: ${attachment.name}]\n${value}` }],
      status: { type: "complete" },
    };
  }

  remove(): Promise<void> {
    return Promise.resolve();
  }
}

class OpenDocumentAttachmentAdapter implements AttachmentAdapter {
  private readonly active = new Set<string>();
  private readonly sending = new Set<string>();
  private readonly content = new Map<
    string,
    Promise<OpenDocumentAttachmentContent | null>
  >();

  accept = [
    ".ods",
    ".odt",
    OPEN_DOCUMENT_SPREADSHEET_MIME,
    OPEN_DOCUMENT_TEXT_MIME,
  ].join(",");

  async *add({
    file,
  }: { file: File }): AsyncGenerator<PendingAttachment, void> {
    const id = crypto.randomUUID();
    this.active.add(id);
    const attachment = {
      id,
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "running", reason: "uploading", progress: 0 },
    } satisfies PendingAttachment;

    yield attachment;
    const content = readActiveOpenDocumentAttachmentContent(
      file,
      file.name,
      file.type,
      () => this.active.has(id),
    );
    this.content.set(id, content);

    try {
      if ((await content) && this.active.has(id) && !this.sending.has(id)) {
        yield {
          ...attachment,
          status: { type: "requires-action", reason: "composer-send" },
        };
      }
    } catch {
      this.active.delete(id);
      this.content.delete(id);
      if (!this.sending.has(id)) {
        yield {
          ...attachment,
          status: { type: "incomplete", reason: "error" },
        };
      }
    }
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    this.sending.add(attachment.id);
    try {
      const content =
        (await this.content.get(attachment.id)) ??
        (await readOpenDocumentAttachmentContent(
          attachment.file,
          attachment.name,
          attachment.contentType ?? "",
        ));
      const { label, text } = content;

      return {
        id: attachment.id,
        type: "document",
        name: attachment.name,
        contentType: attachment.contentType,
        content: [
          { type: "text", text: `[${label}: ${attachment.name}]\n${text}` },
        ],
        status: { type: "complete" },
      };
    } finally {
      this.active.delete(attachment.id);
      this.content.delete(attachment.id);
      this.sending.delete(attachment.id);
    }
  }

  remove(attachment: { id: string }): Promise<void> {
    this.active.delete(attachment.id);
    this.sending.delete(attachment.id);
    this.content.delete(attachment.id);
    return Promise.resolve();
  }
}

function clip(input: string, maxLen: number): string {
  const text = input.replace(/\s+/g, " ").trim();
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd();
}

function extractTextParts(m: ThreadMessage | undefined): string {
  if (!m) return "";
  const content = Array.isArray(m.content) ? m.content : [];
  return content
    .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("")
    .trim();
}

async function generateTitleWithModel(payload: {
  userText: string;
}): Promise<string | null> {
  const params = useChatRuntimeStore.getState().params;
  if (!params.checkpoint) return null;

  const user = clip(payload.userText, 256);
  const parts: string[] = [user];

  function normalizeTitle(raw: string): string | null {
    let title = raw.split(/\r?\n/, 1)[0] ?? "";
    title = title.replace(/^\s*title\s*:\s*/i, "");
    title = title.replace(/[^\x20-\x7E]+/g, " ");
    title = title.replace(/["'`]+/g, "");
    title = title.replace(/[.!?:;,]+/g, " ");
    title = title.replace(/\s+/g, " ").trim();

    // Model echo fail-safe.
    if (/\b(user|base|lora|assistant)\s*:/i.test(title)) {
      return null;
    }

    const words = title.split(" ").filter(Boolean).slice(0, 6);
    const joined = words.join(" ").trim();
    if (!joined) return null;
    return joined.length > 60 ? joined.slice(0, 60).trimEnd() : joined;
  }

  const response = await authFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: params.checkpoint,
      stream: false,
      temperature: 0.2,
      top_p: 0.9,
      max_tokens: 24,
      top_k: 20,
      repetition_penalty: 1.0,
      messages: [
        {
          role: "system",
          content:
            "Write 1 concise chat title for the user's message. Rules: 2-6 words, no quotes, no punctuation, ASCII only, do not echo input. Output title only.",
        },
        { role: "user", content: parts.join("\n") },
      ],
    }),
  });

  const body = (await response
    .json()
    .catch(() => null)) as TitleResponse | null;
  if (!response.ok) return null;
  const raw: string | undefined = body?.choices?.[0]?.message?.content;
  if (!raw) return null;
  return normalizeTitle(raw);
}

const inflightTitleByKey = new Set<string>();

function fallbackTitleFromUserText(userText: string): string {
  const firstLine = (userText || "").split(/\r?\n/, 1)[0] ?? "";
  const cleaned = firstLine.replace(/\s+/g, " ").trim();
  const max = 48;
  if (!cleaned) return "New Chat";
  return cleaned.slice(0, max) + (cleaned.length > max ? "..." : "");
}

function cloneContent(
  content: ThreadMessage["content"],
): ThreadMessage["content"] {
  if (typeof content === "string") {
    return content;
  }
  return Array.isArray(content) ? JSON.parse(JSON.stringify(content)) : [];
}

function cloneAttachments(
  attachments: readonly CompleteAttachment[] | undefined,
): readonly CompleteAttachment[] {
  if (!Array.isArray(attachments)) {
    return [];
  }
  return JSON.parse(JSON.stringify(attachments));
}

function toThreadMessage(m: MessageRecord): ThreadMessage {
  const content =
    Array.isArray(m.content) && m.content.length > 0
      ? cloneContent(m.content)
      : [{ type: "text" as const, text: "" }];

  if (m.role === "user") {
    return {
      id: m.id,
      createdAt: new Date(m.createdAt),
      role: "user" as const,
      content: content as Extract<ThreadMessage, { role: "user" }>["content"],
      attachments: cloneAttachments(m.attachments),
      metadata: { custom: {} },
    };
  }
  const custom = (m.metadata as Record<string, unknown>) ?? {};
  const savedTiming = custom.timing as
    | import("@assistant-ui/react").MessageTiming
    | undefined;
  return {
    id: m.id,
    createdAt: new Date(m.createdAt),
    role: "assistant" as const,
    content: content as Extract<
      ThreadMessage,
      { role: "assistant" }
    >["content"],
    status: { type: "complete" as const, reason: "unknown" as const },
    metadata: {
      custom,
      ...(savedTiming ? { timing: savedTiming } : {}),
      steps: [],
      unstable_annotations: [],
      unstable_data: [],
      unstable_state: null,
    },
  };
}

export async function ensureThreadRecord({
  threadId,
  modelType,
  pairId,
}: {
  threadId: string;
  modelType: ModelType;
  pairId?: string;
}): Promise<void> {
  if (isChatThreadDeleted(threadId)) {
    return;
  }
  const existing = (await listStoredChatThreads({ includeArchived: true })).find(
    (thread) => thread.id === threadId,
  );
  if (existing) {
    return;
  }

  const currentModelId = useChatRuntimeStore.getState().params.checkpoint ?? "";
  const record: ThreadRecord = {
    id: threadId,
    title: "New Chat",
    modelType,
    modelId: currentModelId,
    pairId,
    archived: false,
    createdAt: Date.now(),
  };

  try {
    await saveStoredChatThread(record);
  } catch (error) {
    // assistant-ui can issue overlapping first-message persistence calls.
    // If another call created the same thread while this one was waiting,
    // treat initialization as successful and let the message write continue.
    const existingAfterRace = await listStoredChatThreads({
      includeArchived: true,
    }).catch(() => []);
    if (existingAfterRace.some((thread) => thread.id === threadId)) {
      return;
    }
    throw error;
  }
}

function createStudioDbAdapter(
  modelType: ModelType,
  pairId?: string,
): unstable_RemoteThreadListAdapter {
  return {
    async fetch(remoteId: string) {
      const thread = await getStoredChatThread(remoteId);
      if (!thread) {
        throw new Error(`Thread ${remoteId} not found`);
      }
      return {
        remoteId: thread.id,
        status: thread.archived ? "archived" : "regular",
        title: thread.title,
      };
    },

    async list() {
      let threads: ThreadRecord[];
      try {
        threads = await listStoredChatThreads({ modelType, pairId });
      } catch (error) {
        if (!isExpectedBackgroundChatStorageError(error)) {
          throw error;
        }
        threads = [];
      }
      return {
        threads: threads.map((t) => ({
          status: (t.archived ? "archived" : "regular") as
            | "archived"
            | "regular",
          remoteId: t.id,
          title: t.title,
        })),
      };
    },

    async initialize(threadId: string) {
      await ensureThreadRecord({ threadId, modelType, pairId });
      return { remoteId: threadId, externalId: undefined };
    },

    async rename(remoteId: string, newTitle: string) {
      await ensureStoredChatThread(remoteId);
      await updateStoredChatThread(remoteId, { title: newTitle });
    },

    async archive(remoteId: string) {
      await ensureStoredChatThread(remoteId);
      await updateStoredChatThread(remoteId, { archived: true });
    },

    async unarchive(remoteId: string) {
      await ensureStoredChatThread(remoteId);
      await updateStoredChatThread(remoteId, { archived: false });
    },

    async delete(remoteId: string) {
      await deleteStoredChatThreads([remoteId]);
    },

    async generateTitle(remoteId: string, messages: readonly ThreadMessage[]) {
      const autoTitle = useChatRuntimeStore.getState().autoTitle;
      const thread = await getStoredChatThread(remoteId);
      const defaultTitle = "New Chat";

      function streamTitle(title: string) {
        return createAssistantStream((c) => {
          c.appendText(title);
          c.close();
        });
      }

      async function persistTitle(title: string): Promise<void> {
        await ensureStoredChatThread(remoteId, thread);
        await updateStoredChatThread(remoteId, { title });
        if (!pairId) return;
        const paired = (await listStoredChatThreads({ pairId })).find(
          (t) => t.id !== remoteId,
        );
        if (paired) {
          await ensureStoredChatThread(paired.id, paired);
          await updateStoredChatThread(paired.id, { title });
        }
      }

      if (!thread) {
        return streamTitle(defaultTitle);
      }

      // Only generate once per thread/pair.
      if (thread.title && thread.title !== "New Chat") {
        return streamTitle(thread.title);
      }

      const firstUser = messages.find((m) => m.role === "user");
      const userText = extractTextParts(firstUser) || defaultTitle;

      if (!autoTitle) {
        const title = fallbackTitleFromUserText(userText);
        await persistTitle(title);
        return streamTitle(title);
      }

      const key = pairId ? `pair:${pairId}` : `thread:${remoteId}`;
      if (inflightTitleByKey.has(key)) {
        return streamTitle(thread.title || defaultTitle);
      }

      // Compare: wait until both threads done.
      if (pairId) {
        const paired = (await listStoredChatThreads({ pairId })).find(
          (t) => t.id !== remoteId,
        );

        if (paired) {
          const running = useChatRuntimeStore.getState().runningByThreadId;
          if (running[paired.id]) {
            setTimeout(() => {
              void createStudioDbAdapter(modelType, pairId).generateTitle(
                remoteId,
                messages,
              );
            }, 600);
            return streamTitle(thread.title || defaultTitle);
          }
        }
      }

      inflightTitleByKey.add(key);
      try {
        const title =
          (await generateTitleWithModel({
            userText,
          })) || fallbackTitleFromUserText(userText);

        await persistTitle(title);
        return streamTitle(title);
      } finally {
        inflightTitleByKey.delete(key);
      }
    },
  };
}

type StudioRuntimeAdapters = NonNullable<LocalRuntimeOptions["adapters"]>;

function trackHistoryAppend(
  messageId: string,
  write: Promise<void>,
): Promise<void> {
  pendingHistoryAppendByMessageId.set(messageId, write);
  const cleanup = () => {
    setTimeout(() => {
      if (pendingHistoryAppendByMessageId.get(messageId) === write) {
        pendingHistoryAppendByMessageId.delete(messageId);
      }
    }, 30_000);
  };
  write.then(cleanup, cleanup);
  return write;
}

async function waitForRunStartHistoryAppend(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
): Promise<void> {
  const lastMessage = messages.at(-1);
  if (!lastMessage || lastMessage.role !== "user") {
    return;
  }
  const write = pendingHistoryAppendByMessageId.get(lastMessage.id);
  if (!write) {
    return;
  }
  let didPersist = false;
  try {
    await write;
    didPersist = true;
  } finally {
    if (
      didPersist &&
      pendingHistoryAppendByMessageId.get(lastMessage.id) === write
    ) {
      pendingHistoryAppendByMessageId.delete(lastMessage.id);
    }
  }
}

function createPersistedRunAdapter(adapter: ChatModelAdapter): ChatModelAdapter {
  return {
    ...adapter,
    async *run(options) {
      await waitForRunStartHistoryAppend(options.messages);
      const result = adapter.run(options);
      if (!result) {
        return;
      }
      if (typeof result === "object" && Symbol.asyncIterator in result) {
        yield* result;
        return;
      }
      yield await result;
    },
  };
}

function useStudioRuntimeAdapters(): StudioRuntimeAdapters {
  const aui = useAui();

  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        const { remoteId } = aui.threadListItem().getState();
        if (!remoteId) {
          return { messages: [] };
        }
        const roleOrder: Record<string, number> = {
          system: 0,
          user: 1,
          assistant: 2,
        };
        let msgs: MessageRecord[];
        try {
          msgs = await listStoredChatMessages(remoteId);
        } catch (error) {
          if (!isExpectedBackgroundChatStorageError(error)) {
            throw error;
          }
          msgs = [];
        }
        msgs.sort((a, b) => {
          if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
          const aOrder = roleOrder[a.role] ?? 99;
          const bOrder = roleOrder[b.role] ?? 99;
          if (aOrder !== bOrder) return aOrder - bOrder;
          return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
        });

        // Restore context usage from last assistant message if model matches
        const lastAssistant = [...msgs]
          .reverse()
          .find((m) => m.role === "assistant");
        const savedUsage = (lastAssistant?.metadata as Record<string, unknown>)
          ?.contextUsage as
          | {
              promptTokens: number;
              completionTokens: number;
              totalTokens: number;
              cachedTokens: number;
              cacheWriteTokens?: number;
              modelId?: string;
            }
          | undefined;
        const store = useChatRuntimeStore.getState();
        if (
          savedUsage &&
          store.ggufContextLength &&
          savedUsage.totalTokens <= store.ggufContextLength &&
          (!savedUsage.modelId ||
            savedUsage.modelId === store.params.checkpoint)
        ) {
          store.setContextUsage(savedUsage);
        }

        // If any message has a stored parentId, reconstruct the tree
        // so retries/regenerations load as branches instead of being
        // unrolled into a flat list.  For mixed legacy/new threads
        // (old messages without parentId + new messages with), infer
        // sequential parents for old messages to preserve the chain.
        // Fall back to fromArray for fully legacy threads.
        const hasParentIds = msgs.some((m) => m.parentId != null);
        if (hasParentIds) {
          let previousId: string | null = null;
          return {
            messages: msgs.map((m) => {
              const parentId = m.parentId != null ? m.parentId : previousId;
              previousId = m.id;
              return {
                parentId,
                message: toThreadMessage(m),
              };
            }),
          };
        }
        return ExportedMessageRepository.fromArray(msgs.map(toThreadMessage));
      },

      append({ parentId, message }: ExportedMessageRepositoryItem) {
        const write = (async () => {
          const { remoteId } = await aui.threadListItem().initialize();
          if (isChatThreadDeleted(remoteId)) {
            await deleteStoredChatThreads([remoteId]);
            return;
          }
          // Keep single-chat runtime state in sync once a new chat is first
          // persisted. Compare panes intentionally do not write global activeThreadId.
          const thread = await getStoredChatThread(remoteId);
          if (thread) {
            await ensureStoredChatThread(remoteId, thread);
          }
          if (thread?.modelType === "base" && !thread.pairId) {
            const store = useChatRuntimeStore.getState();
            if (store.activeThreadId !== remoteId) {
              store.setActiveThreadId(remoteId);
            }
          }
          const content = cloneContent(message.content);
          const attachments =
            message.role === "user" ? cloneAttachments(message.attachments) : [];
          const custom = message.metadata?.custom;
          const existingMessage = (await listStoredChatMessages(remoteId)).find(
            (storedMessage) => storedMessage.id === message.id,
          );
          const createdAt =
            existingMessage?.createdAt ??
            message.createdAt?.getTime?.() ??
            Date.now();
          await saveStoredChatMessage({
            id: message.id,
            threadId: remoteId,
            parentId: parentId ?? null,
            role: message.role,
            content,
            ...(attachments.length > 0 && { attachments }),
            ...(custom &&
              Object.keys(custom).length > 0 && { metadata: custom }),
            createdAt,
          });
        })();
        return trackHistoryAppend(message.id, write);
      },
    }),
    [aui],
  );

  const dictation = useMemo(
    () =>
      WebSpeechDictationAdapter.isSupported()
        ? new WebSpeechDictationAdapter()
        : undefined,
    [],
  );
  const attachments = useMemo(
    () =>
      new CompositeAttachmentAdapter([
        new VisionImageAdapter(),
        new TextAttachmentAdapter(),
        new HtmlAttachmentAdapter(),
        new PDFAttachmentAdapter(),
        new DocxAttachmentAdapter(),
        new OpenDocumentAttachmentAdapter(),
      ]),
    [],
  );
  const adapters = useMemo(
    () => ({ history, dictation, attachments }),
    [history, dictation, attachments],
  );

  return adapters;
}

const chatAdapter = createOpenAIStreamAdapter();

function useRuntimeHook(): ReturnType<typeof useLocalRuntime> {
  const adapters = useStudioRuntimeAdapters();
  const persistedChatAdapter = useMemo(
    () => createPersistedRunAdapter(chatAdapter),
    [],
  );
  return useLocalRuntime(persistedChatAdapter, { adapters });
}

function ThreadAutoSwitch({
  threadId,
  syncActiveThreadId = true,
}: {
  threadId: string;
  syncActiveThreadId?: boolean;
}): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);

  useEffect(() => {
    if (!isLoading && mainThreadId !== threadId) {
      const switchResult = aui.threads().switchToThread(threadId) as unknown;
      if (
        switchResult &&
        typeof (switchResult as Promise<void>).catch === "function"
      ) {
        void (switchResult as Promise<void>).catch(() => {
          if (syncActiveThreadId) {
            useChatRuntimeStore.getState().setActiveThreadId(null);
          }
        });
      }
    }
  }, [aui, isLoading, mainThreadId, syncActiveThreadId, threadId]);

  useEffect(() => {
    if (!syncActiveThreadId || isLoading || mainThreadId !== threadId) {
      return;
    }
    useChatRuntimeStore.getState().setActiveThreadId(threadId);
  }, [isLoading, mainThreadId, syncActiveThreadId, threadId]);

  return null;
}

function ThreadNewChatSwitch({
  nonce,
}: { nonce: string }): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);

  useEffect(() => {
    if (isLoading) {
      return;
    }
    // Switch to a fresh local thread without persisting it yet.
    // Persistence still happens on first message append.
    void aui.threads().switchToNewThread();
    useChatRuntimeStore.getState().setActiveThreadId(null);
  }, [aui, isLoading, nonce]);

  return null;
}

function ActiveThreadSync({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const setActiveThreadId = useChatRuntimeStore(
    (state) => state.setActiveThreadId,
  );

  useEffect(() => {
    if (!enabled) {
      return;
    }
    setActiveThreadId(mainThreadId ?? null);
  }, [enabled, mainThreadId, setActiveThreadId]);

  return null;
}

// Exposes the current thread's cancelRun() via the shared store so external
// surfaces (e.g. the sidebar trash button) can stop an in-flight stream
// before deleting the thread — mirroring the Stop → Trash sequence.
function CancelRegistrar(): ReactElement | null {
  const aui = useAui();
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const isRunning = useChatRuntimeStore((s) =>
    mainThreadId ? Boolean(s.runningByThreadId[mainThreadId]) : false,
  );

  useEffect(() => {
    if (!mainThreadId || !isRunning) return;
    const cancel = () => {
      try {
        aui.thread().cancelRun();
      } catch {
        // Run may have already ended between the caller's read and this call.
      }
    };
    useChatRuntimeStore.getState().registerThreadCancel(mainThreadId, cancel);
    return () => {
      useChatRuntimeStore.getState().clearThreadCancel(mainThreadId);
    };
  }, [aui, mainThreadId, isRunning]);

  return null;
}

function ThreadBackendAutosave({
  modelType,
  pairId,
}: {
  modelType: ModelType;
  pairId?: string;
}): ReactElement | null {
  const aui = useAui();
  const saveChainRef = useRef(Promise.resolve());

  const saveThread = useCallback(
    async (threadId: string): Promise<void> => {
      const runtime = aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) {
        return;
      }
      const exported = runtime.threads.getById(threadId).export();
      if (exported.messages.length === 0) {
        return;
      }

      const { remoteId } = await runtime.threads
        .getItemById(threadId)
        .initialize();
      if (isChatThreadDeleted(remoteId)) {
        await deleteStoredChatThreads([remoteId]);
        return;
      }
      await ensureStoredChatThread(remoteId);
      await syncExportedRepositoryToBackend(remoteId, exported);
      if (isChatThreadDeleted(remoteId)) {
        await deleteStoredChatThreads([remoteId]);
        return;
      }

      if (modelType === "base" && !pairId) {
        const store = useChatRuntimeStore.getState();
        const activeThreadId = runtime.threads.getState().mainThreadId;
        if (activeThreadId === threadId && store.activeThreadId !== remoteId) {
          store.setActiveThreadId(remoteId);
        }
      }
    },
    [aui, modelType, pairId],
  );

  const queueSave = useCallback(
    (threadId: string): void => {
      saveChainRef.current = saveChainRef.current
        .catch(() => {})
        .then(() => saveThread(threadId))
        .catch((error) => {
          if (!isExpectedBackgroundChatStorageError(error)) {
            console.error("Failed to autosave chat thread", error);
          }
        });
    },
    [saveThread],
  );

  useAuiEvent("thread.runEnd", ({ threadId }) => {
    queueSave(threadId);
  });

  useAuiEvent("thread.runStart", ({ threadId }) => {
    queueSave(threadId);
  });

  return null;
}

export function ChatRuntimeProvider({
  children,
  modelType = "base",
  pairId,
  initialThreadId,
  newThreadNonce,
  syncActiveThreadId = true,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  initialThreadId?: string;
  newThreadNonce?: string;
  syncActiveThreadId?: boolean;
}): ReactElement {
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useRuntimeHook,
    adapter: createStudioDbAdapter(modelType, pairId),
  });

  const aui = useAui({});

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      <ActiveThreadSync
        enabled={
          modelType === "base" && !pairId && !newThreadNonce && !initialThreadId
        }
      />
      <ThreadBackendAutosave modelType={modelType} pairId={pairId} />
      <CancelRegistrar />
      {initialThreadId && (
        <ThreadAutoSwitch
          threadId={initialThreadId}
          syncActiveThreadId={syncActiveThreadId}
        />
      )}
      {!initialThreadId && newThreadNonce && (
        <ThreadNewChatSwitch nonce={newThreadNonce} />
      )}
      {children}
    </AssistantRuntimeProvider>
  );
}
