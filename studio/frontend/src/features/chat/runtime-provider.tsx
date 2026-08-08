// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  AssistantRuntimeProvider,
  type Attachment,
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
  type unstable_RemoteThreadListAdapter,
  useAui,
  useAuiEvent,
  useAuiState,
  useLocalRuntime,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import {
  type ReactElement,
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { toast } from "sonner";
import { StudioDictationAdapter } from "./adapters/studio-dictation-adapter";
import { StudioSpeechSynthesisAdapter } from "./adapters/studio-speech-synthesis-adapter";
import {
  ThreadAutosaveHandle,
  createOpenAIStreamAdapter,
} from "./api/chat-adapter";
import { getResearchThreadState } from "./api/research-api";
import {
  ingestResearchUpdate,
  useResearchRunStore,
} from "./stores/research-run-store";
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
import { AudioAttachmentAdapter } from "./audio-attachment-adapter";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import { ToolPaneScopeContext, toolPaneScope } from "./tool-output-scope";
import {
  notifyPromptQueueRunFailed,
  requestPromptQueueStop,
  requestTemporaryPromptQueueStop,
} from "./utils/prompt-queue-boundary";
import {
  adoptPreStreamRunReservation,
  claimPreStreamRunReservation,
  findPreStreamRunReservation,
  isPreStreamRunReservationCancelled,
  preStreamRunThreadIdsForRuntime,
  releasePreStreamRunReservation,
} from "./utils/pre-stream-run-reservation";
import type { MessageRecord, ModelType, ThreadRecord } from "./types";
import {
  chatContentPartAttachmentIdFromSignature,
  chatContentPartAttachmentSignature,
  onChatAttachmentDeleted,
} from "./utils/chat-attachment-events";
import {
  refreshContextUsage,
  setActiveBranchReader,
} from "./utils/refresh-context-usage";
import {
  awaitStoredChatThreadRecord,
  deleteStoredChatThreads,
  ensureStoredChatThread,
  getStoredChatMessage,
  getStoredChatThread,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  listStoredChatThreads,
  markThreadIncognito,
  saveStoredChatMessage,
  saveStoredChatThread,
  trackStoredChatThreadRecord,
  updateStoredChatThread,
} from "./utils/chat-history-storage";
import {
  isChatThreadDeleted,
  markChatThreadDeleted,
} from "./utils/chat-thread-tombstones";
import { chatHistoryClearBoundary } from "./utils/chat-history-clear-boundary";
import { fallbackTitleFromUserText } from "./utils/chat-title";
import { syncExportedRepositoryToBackend } from "./utils/delete-thread-message";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { isAssistantLocalThreadId } from "./utils/thread-ids";

const pendingHistoryAppendByMessageId = new Map<string, Promise<void>>();
// Resolves to the thread id assigned when this message's chat was first persisted.
const pendingRunStartReadyByMessageId = new Map<
  string,
  Promise<string | undefined>
>();
const pendingRunStartThreadIdsByMessageId = new Map<string, string[]>();

type TitleResponse = {
  choices?: Array<{
    finish_reason?: string | null;
    message?: {
      content?: string;
    };
  }>;
};

class PreStreamAwareAttachmentAdapter implements AttachmentAdapter {
  private readonly delegate: AttachmentAdapter;
  private readonly getThreadIds: () => Array<string | null | undefined>;

  constructor(
    delegate: AttachmentAdapter,
    getThreadIds: () => Array<string | null | undefined>,
  ) {
    this.delegate = delegate;
    this.getThreadIds = getThreadIds;
  }

  get accept(): string {
    return this.delegate.accept;
  }

  add(state: { file: File }) {
    return this.delegate.add(state);
  }

  remove(attachment: Attachment): Promise<void> {
    return this.delegate.remove(attachment);
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const threadIds = this.getThreadIds();
    const reservationToken = findPreStreamRunReservation(threadIds);
    try {
      return await this.delegate.send(attachment);
    } catch (error) {
      if (
        reservationToken &&
        releasePreStreamRunReservation(reservationToken)
      ) {
        notifyPromptQueueRunFailed(threadIds.find(Boolean) ?? null);
      }
      throw error;
    }
  }
}

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
      loadError: state.lastModelLoadError,
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
    const [{ extractText, getDocumentProxy }, buffer] = await Promise.all([
      import("unpdf"),
      attachment.file.arrayBuffer().then((bytes) => new Uint8Array(bytes)),
    ]);
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
  // MIME is unreliable for source files, so also match by extension
  // (assistant-ui's fileMatchesAccept supports ".ext" entries). Covers svg, code,
  // config and other plain-text formats; html keeps its own adapter below.
  accept = [
    "text/plain,text/markdown,text/csv,text/xml,text/json,text/css",
    "application/json,application/xml,image/svg+xml",
    ".txt,.text,.log,.md,.markdown,.mdx,.rst,.csv,.tsv",
    ".json,.jsonl,.ndjson,.xml,.yaml,.yml,.toml,.ini,.cfg,.conf,.env,.properties",
    ".css,.scss,.sass,.less,.svg",
    ".js,.jsx,.mjs,.cjs,.ts,.tsx,.py,.pyi,.ipynb,.rb,.php,.go,.rs,.java,.kt,.kts,.scala,.swift",
    ".c,.h,.cc,.cpp,.hpp,.cxx,.cs,.m,.mm",
    ".sh,.bash,.zsh,.fish,.ps1,.bat,.lua,.pl,.pm,.r,.jl,.dart,.vue,.svelte,.astro",
    ".sql,.graphql,.gql,.proto,.tf,.tfvars,.gradle,.dockerfile,.makefile,.cmake,.diff,.patch",
  ].join(",");

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
    const doc = new DOMParser().parseFromString(html, "text/html");
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
    const [{ default: mammoth }, arrayBuffer] = await Promise.all([
      import("mammoth"),
      attachment.file.arrayBuffer(),
    ]);
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
  assistantText?: string;
}): Promise<string | null> {
  const params = useChatRuntimeStore.getState().params;
  if (!params.checkpoint) return null;

  const user = clip(payload.userText, 256);
  const assistant = clip(payload.assistantText ?? "", 384);
  const parts: string[] = [`User: ${user}`];
  if (assistant) {
    parts.push(`Assistant: ${assistant}`);
  }

  function normalizeTitle(raw: string): string | null {
    let title = raw.split(/\r?\n/, 1)[0] ?? "";
    title = title.replace(/^\s*title\s*:\s*/i, "");
    title = title.replace(/[^\x20-\x7E]+/g, " ");
    title = title.replace(/["'`]+/g, "");

    // Echo fail-safe: reject leading role labels before punctuation strips the ":".
    if (/^\s*(user|assistant|base|lora)\s*:/i.test(title)) {
      return null;
    }

    title = title.replace(/[.!?:;,]+/g, " ");
    title = title.replace(/\s+/g, " ").trim();

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
      enable_thinking: false,
      reasoning_effort: "none",
      messages: [
        {
          role: "system",
          content:
            "Write 1 concise chat title summarizing the conversation topic, not the user's exact wording. Use the assistant reply as context when provided. Rules: 2-6 words, no quotes, no punctuation, ASCII only, do not echo input. Output title only.",
        },
        { role: "user", content: parts.join("\n") },
      ],
    }),
  });

  const body = (await response
    .json()
    .catch(() => null)) as TitleResponse | null;
  if (!response.ok) return null;
  const choice = body?.choices?.[0];
  if (choice?.finish_reason === "length") return null;
  const raw: string | undefined = choice?.message?.content;
  if (!raw || /<\/?think>/i.test(raw)) return null;
  return normalizeTitle(raw);
}

const inflightTitleByKey = new Set<string>();

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
  projectId,
}: {
  threadId: string;
  modelType: ModelType;
  pairId?: string;
  projectId?: string | null;
}): Promise<void> {
  if (isChatThreadDeleted(threadId)) {
    return;
  }
  // Snapshot the toggle SYNCHRONOUSLY, before the await below. This runs in
  // the same tick as the user's send, so it reliably captures the toggle's
  // state at creation. Reading it after the await would let a toggle-off
  // that lands mid-await (the list call is a real network round-trip) flip
  // the decision and persist what should have been an incognito thread.
  const incognitoAtInit = useChatRuntimeStore.getState().incognito;
  // Fresh assistant-ui threads are local ids. Temporary chats can skip the
  // history list entirely so a storage outage cannot block the first send.
  if (incognitoAtInit && isAssistantLocalThreadId(threadId)) {
    markThreadIncognito(threadId);
    return;
  }
  // A point lookup, not a listing: this must not scale with how many chats exist.
  const existing = await getStoredChatThread(threadId);
  if (existing) {
    return;
  }
  // For non-local ids, keep the existing check first so an already-persisted
  // thread is never tagged -- that's what keeps a real thread saving normally
  // even if the toggle flips on while its run is still streaming.
  if (incognitoAtInit) {
    markThreadIncognito(threadId);
    return;
  }

  const currentModelId = useChatRuntimeStore.getState().params.checkpoint ?? "";
  const record: ThreadRecord = {
    id: threadId,
    title: "New Chat",
    modelType,
    modelId: currentModelId,
    pairId,
    projectId: projectId ?? null,
    archived: false,
    createdAt: Date.now(),
  };

  try {
    await saveStoredChatThread(record);
  } catch (error) {
    // assistant-ui can issue overlapping first-message persistence calls. If
    // another call created the same thread while this one waited, treat init as
    // successful and let the message write continue.
    const existingAfterRace = await getStoredChatThread(threadId).catch(
      () => undefined,
    );
    if (existingAfterRace) {
      return;
    }
    throw error;
  }
}

function createStudioDbAdapter(
  modelType: ModelType,
  pairId?: string,
  projectId?: string | null,
  listThreads = true,
): unstable_RemoteThreadListAdapter {
  return {
    async fetch(remoteId: string) {
      const thread = await getStoredChatThread(remoteId);
      if (!thread) {
        throw new Error(`Thread ${remoteId} not found`);
      }
      return {
        remoteId: thread.id,
        // Always regular: archive state is owned by the app's own controls.
        // Reporting archived here makes assistant-ui unarchive a chat the
        // moment it is opened.
        status: "regular",
        title: thread.title,
      };
    },

    async list() {
      if (!listThreads) {
        return { threads: [] };
      }
      let threads: ThreadRecord[];
      try {
        threads = await listStoredChatThreads({
          modelType,
          pairId,
          ...(projectId !== undefined ? { projectId } : {}),
        });
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

    initialize(threadId: string) {
      // assistant-ui withholds the first message until this resolves, so the row write is tracked, not awaited.
      trackStoredChatThreadRecord(threadId, () =>
        ensureThreadRecord({ threadId, modelType, pairId, projectId }),
      );
      // A run already streaming on this thread filed its handles under "__default" because
      // the id did not exist yet. Re-key them now, or the sidebar row and Stop look up an
      // id nothing is registered against.
      useChatRuntimeStore.getState().adoptDefaultThreadRun(threadId);
      return Promise.resolve({ remoteId: threadId, externalId: undefined });
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
      // No-op on archive state: the app owns it via the sidebar menu and the
      // archived chats settings dialog. assistant-ui calls this when an
      // archived chat is opened, which must not unarchive it.
      await ensureStoredChatThread(remoteId);
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

      const firstUserIndex = messages.findIndex((m) => m.role === "user");
      const firstUser =
        firstUserIndex === -1 ? undefined : messages[firstUserIndex];
      const firstAssistant =
        firstUserIndex === -1
          ? undefined
          : messages.find((m, i) => m.role === "assistant" && i > firstUserIndex);
      const userText = extractTextParts(firstUser) || defaultTitle;
      const assistantText = extractTextParts(firstAssistant);

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
              void createStudioDbAdapter(modelType, pairId, projectId).generateTitle(
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
            assistantText,
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

function trackRunStartReady(
  messageId: string,
  ready: Promise<string | undefined>,
  localThreadId: string,
): Promise<string | undefined> {
  pendingRunStartReadyByMessageId.set(messageId, ready);
  pendingRunStartThreadIdsByMessageId.set(messageId, [localThreadId]);
  ready.then(
    (remoteId) => {
      if (
        remoteId &&
        pendingRunStartReadyByMessageId.get(messageId) === ready
      ) {
        pendingRunStartThreadIdsByMessageId.set(messageId, [
          ...new Set([localThreadId, remoteId]),
        ]);
      }
    },
    () => undefined,
  );
  const cleanup = () => {
    setTimeout(() => {
      if (pendingRunStartReadyByMessageId.get(messageId) === ready) {
        pendingRunStartReadyByMessageId.delete(messageId);
        pendingRunStartThreadIdsByMessageId.delete(messageId);
      }
    }, 30_000);
  };
  ready.then(cleanup, cleanup);
  return ready;
}

function runStartThreadIdsForMessages(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
): string[] {
  const userMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  return userMessage
    ? (pendingRunStartThreadIdsByMessageId.get(userMessage.id) ?? [])
    : [];
}

async function waitForRunStartHistoryAppend(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
): Promise<string | undefined> {
  // Deep Research reserves an assistant placeholder before invoking the model
  // adapter, so the user message is not necessarily the final entry here.
  const userMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  if (!userMessage) {
    return;
  }
  const runStartReady = pendingRunStartReadyByMessageId.get(userMessage.id);
  const historyAppendReady = pendingHistoryAppendByMessageId.get(userMessage.id);
  if (runStartReady === undefined && historyAppendReady === undefined) {
    return undefined;
  }
  let didBecomeReady = false;
  let adoptedThreadId: string | undefined;
  try {
    [adoptedThreadId] = await Promise.all([
      runStartReady ?? Promise.resolve(undefined),
      historyAppendReady?.then(() => undefined),
    ]);
    didBecomeReady = true;
  } finally {
    if (
      didBecomeReady &&
      runStartReady &&
      pendingRunStartReadyByMessageId.get(userMessage.id) === runStartReady
    ) {
      pendingRunStartReadyByMessageId.delete(userMessage.id);
      pendingRunStartThreadIdsByMessageId.delete(userMessage.id);
    }
  }
  return adoptedThreadId;
}

function createPersistedRunAdapter(adapter: ChatModelAdapter): ChatModelAdapter {
  return {
    ...adapter,
    async *run(options) {
      const trackedRunStartThreadIds = runStartThreadIdsForMessages(
        options.messages,
      );
      const reservationThreadIds = preStreamRunThreadIdsForRuntime(
        [options.unstable_threadId, ...trackedRunStartThreadIds],
        useChatRuntimeStore.getState().activeThreadId,
      );
      const reservationToken =
        findPreStreamRunReservation(reservationThreadIds);
      if (reservationToken) {
        claimPreStreamRunReservation(reservationToken);
      }
      const throwIfReservationCancelled = () => {
        if (
          reservationToken &&
          isPreStreamRunReservationCancelled(reservationToken)
        ) {
          releasePreStreamRunReservation(reservationToken);
          throw new DOMException("The send was cancelled", "AbortError");
        }
      };
      throwIfReservationCancelled();
      const persistedRunThreadIds = preStreamRunThreadIdsForRuntime(
        [
          ...reservationThreadIds,
          ...trackedRunStartThreadIds,
        ],
        undefined,
      );
      let adoptedThreadId: string | undefined;
      try {
        adoptedThreadId = await waitForRunStartHistoryAppend(options.messages);
        throwIfReservationCancelled();
      } catch (error) {
        if (reservationToken) {
          releasePreStreamRunReservation(reservationToken);
        }
        // Queued runs do not carry a direct-send reservation. Their persisted
        // preflight can still fail before the model adapter consumes the queued
        // settings. Stop matching pending/waiting work as well as an already
        // dispatched item so a rapid follow-up cannot run after persistence failed.
        requestPromptQueueStop(persistedRunThreadIds);
        notifyPromptQueueRunFailed(
          options.unstable_threadId ?? persistedRunThreadIds[0] ?? null,
        );
        throw error;
      }
      if (reservationToken && adoptedThreadId) {
        adoptPreStreamRunReservation(reservationToken, [
          ...reservationThreadIds,
          adoptedThreadId,
        ]);
      }
      // The thread has an id by the time that resolves, but assistant-ui bound unstable_threadId
      // before the await. Hand the run its real id so a first turn never files its handles
      // under the unresolved key that concurrent runs share.
      const result = adapter.run(
        !options.unstable_threadId && adoptedThreadId
          ? { ...options, unstable_threadId: adoptedThreadId }
          : options,
      );
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

function useStudioRuntimeAdapters(
  modelType: ModelType,
  pairId?: string,
): StudioRuntimeAdapters {
  const aui = useAui();

  // Mirror Data-tab attachment deletions into the loaded thread. The in-memory
  // repository otherwise keeps the attachment, and a later repo-to-storage sync
  // (e.g. deleting a message in the thread) would write it back.
  useEffect(() => {
    let active = true;
    let pendingDeletion = Promise.resolve();
    const unsubscribe = onChatAttachmentDeleted((event) => {
      pendingDeletion = pendingDeletion.then(async () => {
        if (!active) return;
        const { messageId, attachmentId } = event;
        try {
          const thread = aui.thread();
          if (attachmentId.startsWith("content-part-sha256-")) {
            for (let attempt = 0; attempt < 3 && active; attempt += 1) {
              const exported = thread.export();
              const target = exported.messages.find(
                (item) => item.message.id === messageId,
              );
              if (!target || !Array.isArray(target.message.content)) return;
              const content = target.message.content;

              const signatures = content.map((part) =>
                chatContentPartAttachmentSignature(part),
              );
              const ids = await Promise.all(
                signatures.map((signature) =>
                  signature === null
                    ? null
                    : chatContentPartAttachmentIdFromSignature(signature),
                ),
              );
              const targetAttachments = (
                target.message as {
                  attachments?: readonly { id: string }[];
                }
              ).attachments;
              const hasTargetAttachment =
                Array.isArray(targetAttachments) &&
                targetAttachments.some(
                  (attachment) => attachment.id === attachmentId,
                );
              if (
                (!ids.includes(attachmentId) && !hasTargetAttachment) ||
                !active
              ) {
                return;
              }

              // Preserve any messages added or streamed while WebCrypto ran.
              // Retry if the target's managed content itself changed.
              const latest = thread.export();
              const latestTarget = latest.messages.find(
                (item) => item.message.id === messageId,
              );
              const latestContent = latestTarget?.message.content;
              if (!Array.isArray(latestContent)) return;
              const latestSignatures = latestContent.map((part) =>
                chatContentPartAttachmentSignature(part),
              );
              if (
                signatures.length !== latestSignatures.length ||
                signatures.some(
                  (signature, index) => signature !== latestSignatures[index],
                )
              ) {
                continue;
              }

              const messages = latest.messages.map((item) => {
                if (item.message.id !== messageId) return item;
                const attachments = (
                  item.message as {
                    attachments?: readonly { id: string }[];
                  }
                ).attachments;
                return {
                  ...item,
                  message: {
                    ...item.message,
                    content: latestContent.filter(
                      (_, index) => ids[index] !== attachmentId,
                    ),
                    ...(Array.isArray(attachments)
                      ? {
                          attachments: attachments.filter(
                            (attachment) =>
                              attachment.id !== attachmentId,
                          ),
                        }
                      : {}),
                  } as typeof item.message,
                };
              });
              if (active) thread.import({ ...latest, messages });
              return;
            }
            return;
          }

          const exported = thread.export();
          let changed = false;
          const messages = exported.messages.map((item) => {
            if (item.message.id !== messageId) return item;
            const message = item.message;
            const attachments = (
              message as { attachments?: readonly { id: string }[] }
            ).attachments;
            if (
              Array.isArray(attachments) &&
              attachments.some(
                (attachment) => attachment.id === attachmentId,
              )
            ) {
              changed = true;
              return {
                ...item,
                message: {
                  ...message,
                  attachments: attachments.filter(
                    (attachment) => attachment.id !== attachmentId,
                  ),
                } as typeof message,
              };
            }
            if (/^content-part-[0-9]+$/.test(attachmentId)) {
              // Legacy synthetic id for a blob stored as a message content part.
              const idx = Number(attachmentId.slice("content-part-".length));
              const content = message.content;
              if (
                !Array.isArray(content) ||
                !Number.isInteger(idx) ||
                idx < 0 ||
                idx >= content.length
              ) {
                return item;
              }
              const part = content[idx] as { type?: string };
              if (part?.type !== "image" && part?.type !== "audio") return item;
              changed = true;
              return {
                ...item,
                message: {
                  ...message,
                  content: content.filter((_, i) => i !== idx),
                } as typeof message,
              };
            }
            return item;
          });
          if (changed && active) thread.import({ ...exported, messages });
        } catch {
          // No active thread mounted: storage already holds the truth.
        }
      });
      return pendingDeletion;
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [aui]);

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
        // Durable research can outlive this runtime. Reattach its server-owned
        // assistant message to the inline card after navigation or refresh.
        const researchThreadState = await getResearchThreadState(remoteId).catch(
          () => null,
        );
        if (researchThreadState) {
          useResearchRunStore
            .getState()
            .setThreadClaimed(remoteId, researchThreadState.hasRun);
        }
        const activeResearchRun = researchThreadState?.activeRun ?? null;
        if (activeResearchRun) ingestResearchUpdate(activeResearchRun);
        if (activeResearchRun?.assistantMessageId) {
          const assistant = msgs.find(
            (message) => message.id === activeResearchRun.assistantMessageId,
          );
          if (assistant) {
            assistant.metadata = {
              ...(assistant.metadata ?? {}),
              researchRunId: activeResearchRun.id,
              researchRun: activeResearchRun,
              serverManaged: true,
              serverRevision: activeResearchRun.lastEventSeq,
            };
          }
        }
        msgs.sort((a, b) => {
          if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
          const aOrder = roleOrder[a.role] ?? 99;
          const bOrder = roleOrder[b.role] ?? 99;
          if (aOrder !== bOrder) return aOrder - bOrder;
          return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
        });

        // Restore context usage from last assistant message if model matches.
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
        // Window check applies only when a local GGUF window is known; external
        // providers have ggufContextLength === null.
        const withinLocalLimit =
          !store.ggufContextLength ||
          (savedUsage?.totalTokens ?? 0) <= store.ggufContextLength;
        // Legacy unscoped usage (no modelId) is trusted only when a known local
        // window bounds the totals, so an old local turn can't be misattributed
        // to a newly-selected external provider.
        const modelMatches = savedUsage?.modelId
          ? savedUsage.modelId === store.params.checkpoint
          : typeof store.ggufContextLength === "number" &&
            store.ggufContextLength > 0;
        // The value, not a boolean: the writes below need the narrowing.
        const restoredUsage =
          savedUsage && withinLocalLimit && modelMatches ? savedUsage : null;
        if (restoredUsage) {
          // Key by the thread this loader read, not whichever is active when the await resolves:
          // a switch inside it would file this thread's usage under the incoming one. Same rule
          // the adapter's end-of-run write follows.
          store.setThreadContextUsage(remoteId, restoredUsage);
          if (store.activeThreadId === remoteId) {
            store.setContextUsage(restoredUsage);
          }
        }
        // Only when nothing was restored: saved usage is the last completion's exact totals, and
        // refreshContextUsage does NOT stand down for usage already there, so it would overwrite
        // them with an estimate whose completionTokens is 0. A thread opened after a model switch
        // fails modelMatches and still gets priced (#7450).
        // Primary pane only: a compare pane never owns the global bar, so its count would be
        // rebuilt from storage, sent, then dropped at publish for not being activeThreadId.
        if (!restoredUsage && modelType === "base" && !pairId) {
          void refreshContextUsage({ threadId: remoteId });
        }

        // If any message has a stored parentId, reconstruct the tree so
        // retries/regenerations load as branches rather than a flat list. For
        // mixed legacy/new threads, infer sequential parents for old messages to
        // preserve the chain. Fall back to fromArray for fully legacy threads.
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
        const localThreadId = aui.threadListItem().getState().id;
        const historyClearGeneration = chatHistoryClearBoundary.capture();
        const throwIfHistoryWasCleared = async (remoteId: string) => {
          if (
            chatHistoryClearBoundary.capture() === historyClearGeneration
          ) {
            return;
          }
          markChatThreadDeleted(remoteId);
          await deleteStoredChatThreads([remoteId]);
          throw new DOMException("Chat history was cleared", "AbortError");
        };
        const initializeThread = aui
          .threadListItem()
          .initialize()
          .then(async (initialized) => {
            await throwIfHistoryWasCleared(initialized.remoteId);
            return initialized;
          });
        trackRunStartReady(
          message.id,
          initializeThread.then(({ remoteId }) => remoteId),
          localThreadId,
        );
        const write = (async () => {
          const { remoteId } = await initializeThread;
          await awaitStoredChatThreadRecord(remoteId);
          if (isChatThreadDeleted(remoteId)) {
            await deleteStoredChatThreads([remoteId]);
            return;
          }
          // Keep single-chat runtime state in sync once a new chat is first
          // persisted. Compare panes intentionally don't write global activeThreadId.
          if (modelType === "base" && !pairId) {
            const store = useChatRuntimeStore.getState();
            const visibleThreadId = aui.threads().getState().mainThreadId;
            if (
              (visibleThreadId === localThreadId ||
                visibleThreadId === remoteId) &&
              store.activeThreadId !== remoteId
            ) {
              store.setActiveThreadId(remoteId);
            }
          }
          // Point reads, issued together: the model run waits on this write.
          const [thread, existingMessage] = await Promise.all([
            getStoredChatThread(remoteId),
            getStoredChatMessage(remoteId, message.id),
          ]);
          await throwIfHistoryWasCleared(remoteId);
          if (thread?.modelType === "base" && !thread.pairId) {
            const store = useChatRuntimeStore.getState();
            const visibleThreadId = aui.threads().getState().mainThreadId;
            if (
              (visibleThreadId === localThreadId ||
                visibleThreadId === remoteId) &&
              store.activeThreadId !== remoteId
            ) {
              store.setActiveThreadId(remoteId);
            }
          }
          const content = cloneContent(message.content);
          const attachments =
            message.role === "user" ? cloneAttachments(message.attachments) : [];
          const custom = message.metadata?.custom;
          const createdAt =
            existingMessage?.createdAt ??
            message.createdAt?.getTime?.() ??
              Date.now();
          const existingMetadata = existingMessage?.metadata;
          const incomingRevision = Number(
            (custom as Record<string, unknown> | undefined)?.serverRevision ?? -1,
          );
          const existingRevision = Number(existingMetadata?.serverRevision ?? -1);
          const incomingMetadata = custom as
            | Record<string, unknown>
            | undefined;
          const sameResearchRun =
            typeof existingMetadata?.researchRunId === "string" &&
            existingMetadata.researchRunId === incomingMetadata?.researchRunId;
          const preserveServerManaged =
            existingMetadata?.serverManaged === true &&
            (sameResearchRun ||
              !incomingMetadata?.serverManaged ||
              existingRevision > incomingRevision);
          // Echo the backend's stored metadata verbatim on autosave: merging
          // incomingMetadata re-adds client-only fields (researchRun / serverRevision) the
          // server never persisted, so _research_message_would_change sees a diff and
          // rejects every streamed/snapshot update with 409.
          const metadata = preserveServerManaged
            ? existingMetadata
            : incomingMetadata;
          await saveStoredChatMessage({
            id: message.id,
            threadId: remoteId,
            parentId: parentId ?? null,
            role: message.role,
            content: preserveServerManaged ? existingMessage!.content : content,
            ...(attachments.length > 0 && { attachments }),
            ...(metadata && { metadata }),
            createdAt,
          });
          await throwIfHistoryWasCleared(remoteId);
        })();
        return trackHistoryAppend(
          message.id,
          chatHistoryClearBoundary.trackPending(write),
        );
      },
    }),
    [aui, modelType, pairId],
  );

  // Always register the adapter so the mic stays clickable for any engine. The
  // engine is resolved at listen() time and the composer shows guidance when it
  // cannot run, so engine switches also work on an already-mounted thread.
  const dictation = useMemo(() => new StudioDictationAdapter(), []);
  const speech = useMemo(
    () =>
      StudioSpeechSynthesisAdapter.isSupported()
        ? new StudioSpeechSynthesisAdapter()
        : undefined,
    [],
  );
  const attachments = useMemo(
    () =>
      new PreStreamAwareAttachmentAdapter(
        new CompositeAttachmentAdapter([
          new VisionImageAdapter(),
          new AudioAttachmentAdapter(),
          new TextAttachmentAdapter(),
          new HtmlAttachmentAdapter(),
          new PDFAttachmentAdapter(),
          new DocxAttachmentAdapter(),
          new OpenDocumentAttachmentAdapter(),
        ]),
        () => {
          const state = aui.threadListItem().getState();
          return preStreamRunThreadIdsForRuntime(
            [state.remoteId, state.id],
            useChatRuntimeStore.getState().activeThreadId,
          );
        },
      ),
    [aui],
  );
  const adapters = useMemo(
    () => ({ history, dictation, speech, attachments }),
    [history, dictation, speech, attachments],
  );

  return adapters;
}

function useRuntimeHook(
  modelType: ModelType,
  pairId?: string,
): ReturnType<typeof useLocalRuntime> {
  const adapters = useStudioRuntimeAdapters(modelType, pairId);
  const persistedChatAdapter = useMemo(
    () =>
      createPersistedRunAdapter(
        createOpenAIStreamAdapter({ modelType, pairId }),
      ),
    [modelType, pairId],
  );
  return useLocalRuntime(persistedChatAdapter, { adapters });
}

function createRuntimeHook(modelType: ModelType, pairId?: string) {
  return function useConfiguredRuntimeHook(): ReturnType<typeof useLocalRuntime> {
    return useRuntimeHook(modelType, pairId);
  };
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
      // Saved chats keep running in the background, but a temporary chat is
      // unreachable after this switch and must not retain an active queue.
      requestTemporaryPromptQueueStop();
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
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  // Read only by the recount below: New Chat itself must not care whether a run is still going.
  const runActive = useChatRuntimeStore((s) =>
    Object.values(s.runningByThreadId).some(Boolean),
  );
  // The outgoing thread is not read here: New Chat leaves it running.
  useEffect(() => {
    if (isLoading) {
      return;
    }
    // Saved chats keep running in the background. A temporary chat is never
    // persisted, so abandoning it must also discard its otherwise unreachable
    // queue. Queue provenance remains reliable even if incognito was cleared first.
    requestTemporaryPromptQueueStop();
    // Switch to a fresh local thread without persisting it yet; persistence
    // still happens on first message append.
    void aui.threads().switchToNewThread();
    useChatRuntimeStore.getState().setActiveThreadId(null);
  }, [aui, isLoading, nonce]);

  // The effect above blanks the bar, and this view reaches no other recount trigger: no persisted
  // thread for the history loader, and ActiveThreadSync is off while a nonce is present. Keyed on
  // the model too: on a RELOAD of /chat?new=<uuid> nothing is known until status answers.
  useEffect(() => {
    if (
      isLoading ||
      modelLoading ||
      runActive ||
      !checkpoint ||
      ggufContextLength == null
    ) {
      return;
    }
    const store = useChatRuntimeStore.getState();
    if (store.activeThreadId != null || store.contextUsage != null) return;
    void refreshContextUsage();
    // nonce: a fresh New Chat click re-runs the effect above, which blanks the bar again.
    // runActive is a DEPENDENCY, not just a guard: refreshContextUsage declines while anything
    // generates, and nothing else re-fires this when the run ends. ThreadContextUsageRecount
    // cannot cover for it -- an unpersisted New Chat has no activeThreadId.
  }, [checkpoint, ggufContextLength, isLoading, modelLoading, nonce, runActive]);

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

// Lets the recount read the on-screen branch, not the stored records: an incognito thread stores
// none, and a retried thread's newest stored leaf is not what the runtime would send.
function ActiveBranchRegistrar({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const aui = useAui();

  useEffect(() => {
    if (!enabled) {
      return;
    }
    setActiveBranchReader(() => {
      try {
        return aui.thread().getState().messages;
      } catch {
        // No thread mounted yet; the recount falls back to the stored records.
        return null;
      }
    });
    return () => setActiveBranchReader(null);
  }, [aui, enabled]);

  return null;
}

// Price whichever thread the bar points at whenever it has nothing to show. Only two paths reach
// it: (1) a model change empties contextUsageByThreadId and a mounted thread does not rerun its
// history loader; (2) on a deep link to /chat/:id the history loader and status can each land
// before the other, so neither independently timed callback counts.
function ThreadContextUsageRecount({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  // A DEPENDENCY, not just a guard: nothing else here changes when a run ends, so a count skipped
  // for being busy would never be retried. Every run, not just local ones, since that is what the
  // endpoint refuses on.
  const runActive = useChatRuntimeStore((s) =>
    Object.values(s.runningByThreadId).some(Boolean),
  );

  useEffect(() => {
    if (
      !enabled ||
      !activeThreadId ||
      modelLoading ||
      runActive ||
      !checkpoint ||
      ggufContextLength == null
    ) {
      return;
    }
    // Only into a blank bar: restored or completion-written usage is exact, this is an estimate.
    if (useChatRuntimeStore.getState().contextUsage != null) return;
    void refreshContextUsage({ threadId: activeThreadId });
  }, [
    activeThreadId,
    checkpoint,
    enabled,
    ggufContextLength,
    runActive,
    modelLoading,
  ]);

  return null;
}

// Exposes the current thread's cancelRun() via the shared store so external
// surfaces can stop an in-flight stream before deleting the thread.
function CancelRegistrar(): ReactElement | null {
  const aui = useAui();
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const remoteThreadId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );

  useEffect(() => {
    if (!mainThreadId) return;
    const runtime = aui.threads().__internal_getAssistantRuntime?.();
    const threadIds = Array.from(
      new Set([mainThreadId, remoteThreadId].filter((id): id is string => Boolean(id))),
    );
    const cancel = () => {
      for (const threadId of threadIds) {
        try {
          runtime?.threads.getById(threadId).cancelRun();
          return;
        } catch {
          // Try the other alias; the run may also have ended between reads.
        }
      }
    };
    for (const threadId of threadIds) {
      useChatRuntimeStore.getState().registerThreadCancel(threadId, cancel);
    }
    return () => {
      const store = useChatRuntimeStore.getState();
      let thread = null;
      for (const threadId of threadIds) {
        try {
          thread = runtime?.threads.getById(threadId) ?? null;
          if (thread) break;
        } catch {
          // Try the other alias.
        }
      }
      if (!thread?.getState().isRunning) {
        for (const threadId of threadIds) {
          store.clearThreadCancel(threadId, cancel);
        }
        return;
      }
      // assistant-ui enters its running state before adapter preflight turns
      // on runningByThreadId. Keep the only cancel handle after navigation,
      // then release it when assistant-ui reports that the run actually ended.
      let unsubscribe = () => {};
      unsubscribe = thread.subscribe(() => {
        if (thread.getState().isRunning) {
          return;
        }
        for (const threadId of threadIds) {
          useChatRuntimeStore
            .getState()
            .clearThreadCancel(threadId, cancel);
        }
        unsubscribe();
      });
    };
  }, [aui, mainThreadId, remoteThreadId]);

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
  const pendingFirstSavesRef = useRef(new Map<string, Promise<void>>());

  const reportAutosaveError = useCallback((error: unknown): void => {
    if (!isExpectedBackgroundChatStorageError(error)) {
      console.error("Failed to autosave chat thread", error);
    }
  }, []);

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
        .then(async () => {
          await pendingFirstSavesRef.current.get(threadId);
          await saveThread(threadId);
        })
        .catch(reportAutosaveError);
    },
    [reportAutosaveError, saveThread],
  );

  const saveFirstThreadSnapshot = useCallback(
    (threadId: string): void => {
      if (pendingFirstSavesRef.current.has(threadId)) {
        return;
      }

      const promise = saveThread(threadId)
        .catch(reportAutosaveError)
        .finally(() => {
          pendingFirstSavesRef.current.delete(threadId);
        });
      pendingFirstSavesRef.current.set(threadId, promise);
      ThreadAutosaveHandle.registerFirstSave(threadId, promise);
    },
    [reportAutosaveError, saveThread],
  );

  useAuiEvent("thread.runEnd", ({ threadId }) => {
    queueSave(threadId);
  });

  useAuiEvent("thread.runStart", ({ threadId }) => {
    const runtime = aui.threads().__internal_getAssistantRuntime?.();
    const { remoteId } =
      runtime?.threads.getItemById(threadId).getState() ?? {};
    if (!remoteId) {
      saveFirstThreadSnapshot(threadId);
      return;
    }
    queueSave(threadId);
  });

  return null;
}

// True when the chat tab is visible. While false, ChatPage stays mounted (runtime +
// autosave alive, so the stream survives) but its views/composers unmount, so no
// body-portaled surface bleeds over the active tab. Defaults true for use elsewhere.
export const ChatActiveContext = createContext(true);

export function useChatActive(): boolean {
  return useContext(ChatActiveContext);
}

export function ChatRuntimeProvider({
  children,
  modelType = "base",
  pairId,
  projectId,
  initialThreadId,
  newThreadNonce,
  syncActiveThreadId = true,
  listThreads = true,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  projectId?: string | null;
  initialThreadId?: string;
  newThreadNonce?: string;
  syncActiveThreadId?: boolean;
  listThreads?: boolean;
}): ReactElement {
  const runtimeHook = useMemo(
    () => createRuntimeHook(modelType, pairId),
    [modelType, pairId],
  );
  const runtime = useRemoteThreadListRuntime({
    runtimeHook,
    adapter: createStudioDbAdapter(modelType, pairId, projectId, listThreads),
  });

  const aui = useAui({});

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {/* Pane identity for the tool-output store maps: the adapter prefixes its
          keys with this scope so concurrent panes with colliding tool ids
          ("call_0") can't bleed live output into each other's cards. */}
      <ToolPaneScopeContext.Provider value={toolPaneScope(modelType, pairId)}>
        <ActiveThreadSync
          enabled={
            modelType === "base" &&
            !pairId &&
            !newThreadNonce &&
            !initialThreadId
          }
        />
        <ActiveBranchRegistrar enabled={modelType === "base" && !pairId} />
        <ThreadContextUsageRecount enabled={modelType === "base" && !pairId} />
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
        {/* The view stays mounted (only CSS-hidden) while off-route so the run
            stays attached and the stream alive; unmounting aborts generation. */}
        {children}
      </ToolPaneScopeContext.Provider>
    </AssistantRuntimeProvider>
  );
}
