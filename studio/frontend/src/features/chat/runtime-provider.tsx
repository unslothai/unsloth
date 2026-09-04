// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  classifiedAttachmentFile,
  needsAttachmentTrackInspection,
} from "@/lib/video-utils";
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
import { CHAT_HISTORY_UPDATED_EVENT } from "./api/chat-api";
import { getResearchThreadState } from "./api/research-api";
import {
  cancelChatGenerationRun,
  type ChatGenerationRun,
  getActiveChatGenerationRuns,
  ChatGenerationStalledError,
  followChatGenerationRun,
  isTerminalChatGenerationRun,
} from "./api/chat-generation-api";
import {
  TEXT_ATTACHMENT_ACCEPT,
  extractDocxAttachmentText,
  extractHtmlAttachmentText,
  extractPdfAttachmentText,
  getDocumentAttachmentSizeError,
  getDocxAttachmentError,
} from "./attachment-content";
import { AudioAttachmentAdapter } from "./audio-attachment-adapter";
import {
  isBinaryPropertyList,
  isBinaryTrackerModule,
  MAX_TEXT_ATTACHMENT_BYTES,
  isBinaryOfficeTemplate,
  isCompiledFortranModule,
  isBinaryVobSubSubtitle,
  readTextAttachmentOnce,
  UndecodableTextError,
} from "./text-attachment-accept";
import {
  loadConnectionsEnabled,
  loadExternalProviders,
  parseExternalModelId,
  providerModelSupportsVision,
} from "./external-providers";
import { chatModelLoaded } from "./lib/chat-model-loaded";
import {
  type OpenDocumentAttachmentContent,
  readActiveOpenDocumentAttachmentContent,
  readOpenDocumentAttachmentContent,
} from "./open-document";
import { OPEN_DOCUMENT_ATTACHMENT_ACCEPT } from "./open-document-accept";
import {
  awaitThreadScopedSettingsWrite,
  beginThreadScopedPairing,
  commitHeldThreadScopedEditsToTheirThread,
  releaseHeldThreadScopedEdits,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import {
  ingestResearchUpdate,
  useResearchRunStore,
} from "./stores/research-run-store";
import { ToolPaneScopeContext, toolPaneScope } from "./tool-output-scope";
import { ChatProjectScopeContext } from "./chat-project-scope";
import { readThreadCreationClaim } from "./utils/chat-thread-creation-claim";
import type { MessageRecord, ModelType, ThreadRecord } from "./types";
import type { OpenAIChatChunk } from "./types/api";
import {
  budgetImpliesTruncation,
  restoredAssistantStatus,
} from "./utils/continuation";
import {
  generationChunkCountsTowardTiming,
  generationChunkHasSubstantiveDelta,
  generationIsCorroboratedLive,
  threadHasDurableGenerationRun,
  generationNeedsRecovery,
  isLiveGenerationRun,
  generationRawContent,
  loadGenerationOverlaySnapshot,
  markServerActiveGenerationRunsUnknown,
  forgetServerActiveGenerationRun,
  syncServerActiveGenerationRuns,
  recoveredContentToImport,
  recoveredReasoningSummaryMetadata,
  recoveredGenerationFinalMetadata,
  generationRecoveryMetadata,
  shouldPreserveGenerationMetadata,
  subscribeGenerationRecoveryTriggers,
} from "./utils/chat-generation-recovery";
import { mergeContextTruncation } from "./utils/context-truncation";
import {
  extractDeltaText,
  parseAssistantContent,
} from "./utils/parse-assistant-content";
import {
  chatContentPartAttachmentIdFromSignature,
  chatContentPartAttachmentSignature,
  onChatAttachmentDeleted,
} from "./utils/chat-attachment-events";
import { chatHistoryClearBoundary } from "./utils/chat-history-clear-boundary";
import { createParentResolver } from "./utils/message-order";
import {
  awaitStoredChatThreadWrites,
  deleteStoredChatThreads,
  ensureStoredChatThread,
  getStoredChatMessage,
  getStoredChatThread,
  getStoredChatThreadReadResult,
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
import { fallbackTitleFromUserText } from "./utils/chat-title";
import { syncExportedRepositoryToBackend } from "./utils/delete-thread-message";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import {
  attachmentContentText,
  attachmentsSample,
  isPastedTextFile,
} from "./utils/pasted-text";
import {
  adoptPreStreamRunReservation,
  claimPreStreamRunReservation,
  findPreStreamRunReservation,
  isPreStreamRunReservationCancelled,
  preStreamRunThreadIdsForRuntime,
  releasePreStreamRunReservation,
} from "./utils/pre-stream-run-reservation";
import {
  notifyPromptQueueRunFailed,
  requestPromptQueueStop,
  requestTemporaryPromptQueueStop,
} from "./utils/prompt-queue-boundary";
import {
  refreshContextUsage,
  setActiveBranchReader,
} from "./utils/refresh-context-usage";
import {
  type RunCheckpointScheduler,
  createRunCheckpointScheduler,
} from "./utils/run-checkpoint-scheduler";
import { isAssistantLocalThreadId } from "./utils/thread-ids";
import { sanitizeThreadScopedSettings } from "./utils/thread-scoped-settings";
import { VideoAttachmentAdapter } from "./video-attachment-adapter";

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
    // A composite picks its adapter synchronously from the name and MIME type,
    // and both say "video" for an audio-only 3GP recording, so settle that from
    // the container's own tracks first, as the native readers do. Every other
    // file goes straight through, keeping the delegate's own return.
    if (!needsAttachmentTrackInspection(state.file)) {
      return this.delegate.add(state);
    }
    return this.addInspected(state);
  }

  private async addInspected(state: {
    file: File;
  }): Promise<PendingAttachment> {
    const file = await classifiedAttachmentFile(state.file);
    const added = await this.delegate.add({ ...state, file });
    if (Symbol.asyncIterator in added) {
      // Only the audio and video adapters claim a 3GP and both resolve to one
      // attachment, so this drains a generator to its last value rather than
      // forwarding the progress an adapter here does not report.
      let last: PendingAttachment | undefined;
      for await (const value of added) last = value;
      if (!last) throw new Error("The attachment adapter yielded nothing.");
      return last;
    }
    return added;
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
    const modelLoaded = chatModelLoaded({
      checkpoint,
      modelLoading: state.modelLoading,
      isExternalModel,
      residentCheckpoint: state.residentCheckpoint,
    });
    let externalSupportsVision: boolean | null = null;
    let externalModelLabel: string | null = null;
    if (externalSelection !== null) {
      const providers = loadConnectionsEnabled() ? loadExternalProviders() : [];
      const provider = providers.find(
        (p) => p.id === externalSelection.providerId,
      );
      externalSupportsVision = providerModelSupportsVision(
        provider?.providerType,
        externalSelection.modelId,
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
      visionDisabledByUser: state.loadedVisionDisabledByUser,
      mmprojFallbackReason: state.mmprojFallbackReason,
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

  // Refused here, not at send: the composer empties itself before it awaits send(), so a ceiling
  // that only fires there discards the typed message too. The throw is invisible: nothing
  // subscribes to attachmentAddError and the picker never awaits addAttachment, so the toast is
  // the only thing telling the user why no file appeared.
  add({ file }: { file: File }): Promise<PendingAttachment> {
    const sizeError = getDocumentAttachmentSizeError(file, "PDF");
    if (sizeError) {
      toast.error(sizeError);
      throw new Error(sizeError);
    }
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
    const text = await extractPdfAttachmentText(attachment.file);
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
  // MIME is unreliable for source files, so also match by extension (assistant-ui's
  // fileMatchesAccept supports ".ext"). Covers svg, code and plain text; html has its own.
  accept = TEXT_ATTACHMENT_ACCEPT;

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    // Before any read: decoding an .mbox of arbitrary size would hold the bytes
    // and the decoded string at once, and the native path already stops here.
    if (file.size > MAX_TEXT_ATTACHMENT_BYTES) {
      const reason = `Text attachments are limited to ${
        MAX_TEXT_ATTACHMENT_BYTES / (1024 * 1024)
      } MB.`;
      toast.error(reason);
      throw new Error(reason);
    }
    if (await isBinaryPropertyList(file)) {
      const reason =
        "Binary property-list files aren't supported. Convert the file to text before attaching it.";
      toast.error(reason);
      throw new Error(reason);
    }
    if (await isBinaryVobSubSubtitle(file)) {
      const reason =
        "VobSub bitmap subtitles aren't supported. Convert the .sub file to SRT or VTT before attaching it.";
      toast.error(reason);
      throw new Error(reason);
    }
    if (await isBinaryTrackerModule(file)) {
      const reason =
        "Tracker .mod audio files aren't supported as text attachments.";
      toast.error(reason);
      throw new Error(reason);
    }
    if (await isCompiledFortranModule(file)) {
      const reason =
        "Compiled Fortran .mod modules aren't supported as text attachments.";
      toast.error(reason);
      throw new Error(reason);
    }
    if (await isBinaryOfficeTemplate(file)) {
      const reason =
        "Legacy Word and PowerPoint templates aren't supported as text attachments.";
      toast.error(reason);
      throw new Error(reason);
    }
    // Decodes here so an unreadable encoding is reported while attaching rather
    // than at send, where the composer has no room to explain it.
    try {
      await readTextAttachmentOnce(file);
    } catch (error) {
      if (error instanceof UndecodableTextError) {
        toast.error(error.message);
      }
      throw error;
    }
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
    const text = await readTextAttachmentOnce(attachment.file);
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [
        {
          type: "text",
          // A pasted file gets its own tag and size, the markers that outlive the File once the message is stored.
          text: attachmentContentText(
            attachment.name,
            text,
            isPastedTextFile(attachment.file),
            attachment.file.size,
          ),
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
    const text = extractHtmlAttachmentText(await attachment.file.text());
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

  // The archive's own parts are checked too: a small .docx can declare a part that only mammoth's
  // inflate grows past the cap, and refusing that at send() would empty the composer.
  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const error = await getDocxAttachmentError(file);
    if (error) {
      toast.error(error);
      throw new Error(error);
    }
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
    const text = await extractDocxAttachmentText(attachment.file);
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[DOCX: ${attachment.name}]\n${text}` }],
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

  accept = OPEN_DOCUMENT_ATTACHMENT_ACCEPT;

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

// A paste leaves the message's text in an attachment, so a title built from inline text alone
// is "New Chat" for a paste-only turn. The sample is bounded.
function titleTextOf(m: ThreadMessage | undefined): string {
  const text = extractTextParts(m);
  if (m?.role !== "user") return text;
  const sample = attachmentsSample(m.attachments);
  if (sample.length === 0) return text;
  return text.length > 0 ? `${text}\n\n${sample}` : sample;
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
      // Titling is a one-shot summarisation: never let it enter the tool loop. Omitting the field
      // would inherit the server's tools-on default and put tool schemas in a 24-token prompt.
      enable_tools: false,
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
  const generationStatus = custom.generationStatus;
  const hasRunId = typeof custom.generationRunId === "string";
  const generationUnfinished =
    hasRunId &&
    (generationStatus === "queued" ||
      generationStatus === "running" ||
      generationStatus === "cancelling");
  const generationUnsettled =
    hasRunId &&
    generationStatus === "completed" &&
    custom.generationSettled !== true;
  // Persisted metadata alone must never restore a running message: a run that never terminalised
  // still says "running" on every later load, and `{type:"running"}` unmounts Send. Ask
  // whether the run is corroborated live, else show the partial as interrupted, which keeps
  // every character and offers Continue. Unfinished runs only: a completed-but-unsettled run
  // has a terminal status and is replayed by the follow instead.
  const needsGenerationRecovery =
    generationUnsettled ||
    (generationUnfinished && generationIsCorroboratedLive(custom, m.threadId));
  const restoredCustom =
    generationUnfinished &&
    !needsGenerationRecovery &&
    custom.incomplete === undefined
      ? { ...custom, incomplete: { reason: "interrupted" as const } }
      : custom;
  const restoredStatus = restoredAssistantStatus({ custom: restoredCustom });
  return {
    id: m.id,
    createdAt: new Date(m.createdAt),
    role: "assistant" as const,
    content: content as Extract<
      ThreadMessage,
      { role: "assistant" }
    >["content"],
    status: needsGenerationRecovery ? { type: "running" } : restoredStatus,
    metadata: {
      custom: restoredCustom,
      ...(savedTiming ? { timing: savedTiming } : {}),
      steps: [],
      unstable_annotations: [],
      unstable_data: [],
      unstable_state: null,
    },
  };
}

type GenerationRecovery = {
  promise: Promise<void>;
  views: Set<ReturnType<typeof useAui>>;
};

const generationRecoveries = new Map<string, GenerationRecovery>();

function scheduleGenerationRecovery(
  threadId: string,
  storedMessage: MessageRecord,
  aui: ReturnType<typeof useAui>,
): void {
  const metadata = (storedMessage.metadata ?? {}) as Record<string, unknown>;
  const runId = metadata.generationRunId;
  if (typeof runId !== "string" || !generationNeedsRecovery(metadata)) return;
  // This tab is streaming the run itself: following it from storage as well gives the reply two
  // writers, and the follower is always behind, so it imports a lagging prefix.
  if (isLiveGenerationRun(runId)) return;
  const existingRecovery = generationRecoveries.get(runId);
  if (existingRecovery) {
    existingRecovery.views.add(aui);
    return;
  }
  const views = new Set([aui]);

  const recovery = (async () => {
    let cursor = Number(metadata.generationSeq ?? 0);
    if (!Number.isSafeInteger(cursor) || cursor < 0) cursor = 0;
    let { raw, reasoningOpen } = generationRawContent(storedMessage.content);
    let completionTokens: number | undefined;
    let recoveryUsage:
      | {
          prompt_tokens?: unknown;
          completion_tokens?: unknown;
          total_tokens?: unknown;
          prompt_tokens_details?: { cached_tokens?: unknown };
          cache_creation_input_tokens?: unknown;
          cache_read_input_tokens?: unknown;
        }
      | undefined = (metadata.generationRecoveryUsage as typeof recoveryUsage) ?? undefined;
    let recoveryTimings: Record<string, unknown> | undefined =
      (metadata.generationRecoveryTimings as Record<string, unknown>) ?? undefined;
    let firstChunkAt =
      typeof metadata.generationFirstChunkAt === "number" &&
      Number.isFinite(metadata.generationFirstChunkAt)
        ? metadata.generationFirstChunkAt
        : undefined;
    let totalChunks = Number(metadata.generationChunkCount ?? 0);
    if (!Number.isSafeInteger(totalChunks) || totalChunks < 0) totalChunks = 0;
    let currentMetadata = { ...metadata };
    const serverCancel = () => {
      void cancelChatGenerationRun(runId).catch(() => {});
    };
    const runtime = useChatRuntimeStore.getState();
    runtime.registerThreadServerCancel(threadId, serverCancel);
    runtime.setThreadRunning(threadId, true, {
      local: true,
      owner: serverCancel,
    });

    /** Write one state of the reply to storage and to every view showing it. `running` is the
     *  caller's, not derived here: a follow that hit its no-progress deadline settles the message
     *  while its persisted run status is still non-terminal. */
    const commit = async (
      nextMetadata: Record<string, unknown>,
      running: boolean,
    ) => {
      currentMetadata = nextMetadata;
      const content = parseAssistantContent(
        reasoningOpen ? `${raw}</think>` : raw,
      ) as MessageRecord["content"];
      await saveStoredChatMessage({
        id: storedMessage.id,
        threadId,
        parentId: storedMessage.parentId ?? null,
        role: "assistant",
        content,
        metadata: nextMetadata,
        createdAt: storedMessage.createdAt,
      }).catch(() => {
        // The producer may have committed a newer status between the event and this write. Keep
        // following; the terminal publish carries all content.
      });

      for (const view of views) {
        if (view.threadListItem().getState().remoteId !== threadId) continue;
        try {
          const exported = view.thread().export();
          const messages = exported.messages.map((item) =>
            item.message.id === storedMessage.id
              ? {
                  ...item,
                  message: {
                    ...item.message,
                    // The status and the metadata are always this publish's, since they are what the recovery is
                    // following the run FOR. The body is not: a run this tab also streams replays far behind.
                    content: recoveredContentToImport(
                      item.message.content,
                      content,
                    ),
                    status: running
                      ? { type: "running" as const }
                      : restoredAssistantStatus({ custom: nextMetadata }),
                    metadata: {
                      ...item.message.metadata,
                      custom: nextMetadata,
                    },
                  } as ThreadMessage,
                }
              : item,
          );
          view.thread().import({ ...exported, messages });
        } catch {
          // Storage remains authoritative if the thread unmounted between awaits.
        }
      }
    };

    const publish = async (run: ChatGenerationRun) => {
      const status = run.status;
      const runModel = useChatRuntimeStore
        .getState()
        .models.find((model) => model.id === run.requestPayload.model);
      const lengthLimited =
        run.finishReason === "length" ||
        budgetImpliesTruncation({
          isMlx: runModel?.isMlx === true,
          maxTokens: run.requestPayload.max_tokens,
          completionTokens,
        });
      let nextMetadata = generationRecoveryMetadata({
        current: currentMetadata,
        runId,
        status,
        cursor,
        lastEventSeq: run.lastEventSeq,
        lengthLimited,
        firstChunkAt,
        totalChunks,
        usage: recoveryUsage,
        timings: recoveryTimings,
      });
      if (nextMetadata.generationSettled === true) {
        nextMetadata = recoveredGenerationFinalMetadata({
          current: nextMetadata,
          run,
          usage: recoveryUsage,
          timings: recoveryTimings,
          firstChunkAt,
          totalChunks,
        });
      }
      await commit(nextMetadata, generationNeedsRecovery(nextMetadata));
    };

    try {
      let lastPublishedStatus = "";
      let identityValidated = false;
      // The follower reports its no-progress deadline by throwing, and the settlement below is
      // exactly what must happen then; without this catch the message stays running forever.
      let followStalled = false;
      try {
        for await (const update of followChatGenerationRun(runId, {
          replayFrom: cursor,
        })) {
          if (!identityValidated) {
            if (
              update.run.threadId !== threadId ||
              update.run.assistantMessageId !== storedMessage.id
            ) {
              return;
            }
            if (cursor === 0 && raw.length === 0) {
              const requestMessages = update.run.requestPayload.messages;
              const lastRequestMessage = Array.isArray(requestMessages)
                ? requestMessages.at(-1)
                : undefined;
              if (
                lastRequestMessage?.role === "assistant" &&
                typeof lastRequestMessage.content === "string"
              ) {
                // Continue sends the old partial as an assistant prefill. The server-owned placeholder is
                // empty until the first client save, so a reload before that save seeds replay from the
                // request instead.
                raw = lastRequestMessage.content;
              }
            }
            identityValidated = true;
          }
          if (update.event && update.event.seq > cursor) {
            cursor = update.event.seq;
            if (update.event.type === "chunk") {
              const chunk = update.event.payload as {
                _reasoningDurationMs?: unknown;
                usage?: {
                  prompt_tokens?: unknown;
                  completion_tokens?: unknown;
                  total_tokens?: unknown;
                  prompt_tokens_details?: { cached_tokens?: unknown };
                  cache_creation_input_tokens?: unknown;
                  cache_read_input_tokens?: unknown;
                };
                timings?: Record<string, unknown>;
                choices?: Array<{
                  delta?: {
                    content?: unknown;
                    reasoning_content?: unknown;
                  };
                }>;
                context_truncated?: OpenAIChatChunk["context_truncated"];
              };
              if ("_reasoningDurationMs" in chunk) {
                currentMetadata = recoveredReasoningSummaryMetadata(
                  currentMetadata,
                  chunk._reasoningDurationMs,
                );
                lastPublishedStatus = update.run.status;
                await publish(update.run);
                continue;
              }
              if (generationChunkCountsTowardTiming(chunk)) {
                totalChunks += 1;
              }
              if (generationChunkHasSubstantiveDelta(chunk)) {
                firstChunkAt ??= update.event.createdAt;
              }
              if (chunk.context_truncated) {
                currentMetadata = {
                  ...currentMetadata,
                  contextTruncation: mergeContextTruncation(
                    currentMetadata.contextTruncation as OpenAIChatChunk["context_truncated"],
                    chunk.context_truncated,
                  ),
                };
              }
              if (chunk.usage) recoveryUsage = chunk.usage;
              if (chunk.timings) recoveryTimings = chunk.timings;
              if (typeof chunk.usage?.completion_tokens === "number") {
                completionTokens = chunk.usage.completion_tokens;
              }
              const deltaRecord = chunk.choices?.[0]?.delta;
              const reasoning =
                typeof deltaRecord?.reasoning_content === "string"
                  ? deltaRecord.reasoning_content
                  : "";
              const delta = extractDeltaText(deltaRecord?.content).text;
              if (reasoning) {
                if (!reasoningOpen) raw += "<think>";
                raw += reasoning;
                reasoningOpen = true;
              }
              if (delta) {
                if (reasoningOpen) raw += "</think>";
                raw += delta;
                reasoningOpen = false;
              }
            }
          }
          const shouldPublish =
            update.event?.type === "chunk" ||
            update.run.status !== lastPublishedStatus ||
            (["cancelled", "completed", "failed"].includes(update.run.status) &&
              cursor >= update.run.lastEventSeq);
          if (shouldPublish) {
            lastPublishedStatus = update.run.status;
            await publish(update.run);
          }
          if (isTerminalChatGenerationRun(update.run)) {
            // Only another successful active-list sync would otherwise drop it, so the thread would keep
            // reading as durable and a later subscriber-owned stream would be capped, losing the
            // checkpoints that are its only persistence.
            forgetServerActiveGenerationRun(runId);
          }
        }
      } catch (error) {
        if (!(error instanceof ChatGenerationStalledError)) throw error;
        followStalled = true;
      }
      if (followStalled || generationNeedsRecovery(currentMetadata)) {
        // Fence the run before presenting the reply as resumable: settling only in this tab leaves the
        // row active, and create_run refuses a thread that already has one, so Continue would 409.
        // Best effort: a producer wedged inside the engine stays in `cancelling` for the sweeper.
        serverCancel();
        // The follow returned with the run still non-terminal, so its no-progress deadline expired.
        // Settle the reply here rather than leaving a message that says "running" and keeps Send
        // unmounted. Everything replayed so far is kept.
        await commit(
          {
            ...currentMetadata,
            incomplete: { reason: "interrupted" as const },
            // The run row may still be non-terminal, so without this marker generationNeedsRecovery stays
            // true and the next trigger starts another follower. history.load clears it if
            // /chat-runs/active still names the run, so the server gets the last word.
            generationLocallyInterrupted: true,
          },
          false,
        );
      }
    } finally {
      const store = useChatRuntimeStore.getState();
      store.setThreadRunning(threadId, false, { owner: serverCancel });
      store.clearThreadServerCancel(threadId, serverCancel);
    }
  })()
    .catch(() => {})
    .finally(() => generationRecoveries.delete(runId));
  generationRecoveries.set(runId, { promise: recovery, views });
}

export async function ensureThreadRecord({
  threadId,
  modelType,
  pairId,
  projectId,
  incognito,
  neverSent,
  modelId,
  modelGgufVariant,
  createdAt,
}: {
  threadId: string;
  modelType: ModelType;
  pairId?: string;
  projectId?: string | null;
  /** Snapshot from the send this row belongs to, so a retry cannot read a since-flipped toggle. */
  incognito?: boolean;
  /** True only from initialize(), which runs once for a thread that has never been sent to. */
  neverSent?: boolean;
  /** Snapshot from the send this row belongs to, so retries cannot adopt a later checkpoint. */
  modelId?: string;
  modelGgufVariant?: string | null;
  /** Snapshot from the send this row belongs to, so retries retain its original creation time. */
  createdAt?: number;
}): Promise<void> {
  if (isChatThreadDeleted(threadId)) {
    return;
  }
  // Snapshot mutable creation inputs synchronously, before the await below: this runs in the same
  // tick as the user's send, so a toggle or checkpoint change during the point lookup cannot
  // change the identity of the thread a retry persists.
  const runtimeStateAtInit = useChatRuntimeStore.getState();
  const incognitoAtInit = incognito ?? runtimeStateAtInit.incognito;
  const modelIdAtInit = modelId ?? runtimeStateAtInit.params.checkpoint ?? "";
  const modelGgufVariantAtInit =
    modelGgufVariant !== undefined
      ? modelGgufVariant
      : runtimeStateAtInit.activeGgufVariant;
  const createdAtInit = createdAt ?? Date.now();
  // A temporary chat skips the history list so a storage outage cannot block its first send.
  // Gated on the caller knowing the thread is new, not on its id: a `__LOCALID_` id is the
  // permanent key of every chat the app creates, so keying on the prefix tagged SAVED chats.
  if (incognitoAtInit && neverSent) {
    markThreadIncognito(threadId);
    return;
  }
  // A point lookup, not a listing: this must not scale with how many chats exist.
  const existing = await getStoredChatThread(threadId);
  if (existing) {
    return;
  }
  // After the row check, so an already-persisted thread is never tagged: that is what keeps a
  // real chat saving normally when the toggle flips on mid-stream.
  if (incognitoAtInit) {
    markThreadIncognito(threadId);
    return;
  }

  const record: ThreadRecord = {
    id: threadId,
    title: "New Chat",
    modelType,
    modelId: modelIdAtInit,
    modelGgufVariant: modelGgufVariantAtInit,
    pairId,
    projectId: projectId ?? null,
    archived: false,
    createdAt: createdAtInit,
  };

  try {
    await saveStoredChatThread(record);
  } catch (error) {
    // assistant-ui can issue overlapping first-message persistence calls, so if another call
    // created the same thread while this one waited, treat init as successful.
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
        // Always regular: archive state is owned by the app's own controls, and reporting archived here
        // makes assistant-ui unarchive a chat the moment it is opened.
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
      // assistant-ui withholds the first message until this resolves, so the row write is tracked,
      // not awaited. Captured here, not inside the creator: these describe what the SEND was made
      // under, and all four move in between, since materialization is no longer the send's tick.
      const claim = readThreadCreationClaim(threadId);
      const runtimeStateAtInit = useChatRuntimeStore.getState();
      const incognitoAtInit = claim ? claim.incognito : runtimeStateAtInit.incognito;
      const modelIdAtInit = claim
        ? claim.modelId
        : (runtimeStateAtInit.params.checkpoint ?? "");
      const modelGgufVariantAtInit = claim
        ? claim.modelGgufVariant
        : runtimeStateAtInit.activeGgufVariant;
      const createdAtInit = claim ? claim.createdAt : Date.now();
      const projectIdAtInit = claim ? claim.projectId : projectId;
      trackStoredChatThreadRecord(threadId, () =>
        ensureThreadRecord({
          threadId,
          modelType,
          pairId,
          projectId: projectIdAtInit,
          incognito: incognitoAtInit,
          // The one caller that can promise this: assistant-ui runs initialize() once, for the id it
          // just minted. Others hand in whatever chat is open.
          neverSent: true,
          modelId: modelIdAtInit,
          modelGgufVariant: modelGgufVariantAtInit,
          createdAt: createdAtInit,
        }),
      );
      // A run already streaming on this thread filed its handles under "__default" because the id did
      // not exist yet. Re-key them, or the sidebar row and Stop look up an unregistered id.
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
      // No-op on archive state: the app owns it via the sidebar menu, and assistant-ui calls this
      // when an archived chat is opened, which must not unarchive it.
      await ensureStoredChatThread(remoteId);
    },

    async delete(remoteId: string) {
      await deleteStoredChatThreads([remoteId]);
    },

    async generateTitle(remoteId: string, messages: readonly ThreadMessage[]) {
      const autoTitle = useChatRuntimeStore.getState().autoTitle;
      // The run normally waits for its history append, but a bounded persistence wait can expire while
      // the creator is still queued, so use the same retry choke point as other mutations. A title
      // is cosmetic, so a row that never landed falls back to the default.
      const thread = await ensureStoredChatThread(remoteId).catch(
        () => undefined,
      );
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
          : messages.find(
              (m, i) => m.role === "assistant" && i > firstUserIndex,
            );
      const userText = titleTextOf(firstUser) || defaultTitle;
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
              void createStudioDbAdapter(
                modelType,
                pairId,
                projectId,
              ).generateTitle(remoteId, messages);
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
  // Deep Research reserves an assistant placeholder before invoking the model adapter, so the
  // user message is not necessarily the final entry here.
  const userMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");
  if (!userMessage) {
    return;
  }
  const runStartReady = pendingRunStartReadyByMessageId.get(userMessage.id);
  const historyAppendReady = pendingHistoryAppendByMessageId.get(
    userMessage.id,
  );
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

function createPersistedRunAdapter(
  adapter: ChatModelAdapter,
): ChatModelAdapter {
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
        [...reservationThreadIds, ...trackedRunStartThreadIds],
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
        // Queued runs carry no direct-send reservation and their persisted preflight can still fail
        // before the model adapter consumes the queued settings, so match pending/waiting work too.
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
      // before the await. Hand the run its real id so a first turn never uses the shared key.
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
  reloadReadyThreadId?: string,
  onInitialHistoryReady?: () => void,
  // A ref, so handing it down never changes the memoized runtime hook's identity: a new identity
  // would rebuild the runtime, which is the one thing this must not do.
  backgroundedRef?: { current: boolean },
  newThreadSwitchStateRef?: { current: NewThreadSwitchState },
): StudioRuntimeAdapters {
  const aui = useAui();

  useEffect(() => {
    const recoverCurrentThread = () => {
      const remoteId = aui.threadListItem().getState().remoteId;
      if (!remoteId) return;
      void listStoredChatMessages(remoteId)
        .then((messages) => {
          for (const message of messages) {
            if (
              message.role === "assistant" &&
              typeof (message.metadata as Record<string, unknown> | undefined)
                ?.generationRunId === "string"
            ) {
              scheduleGenerationRecovery(remoteId, message, aui);
            }
          }
        })
        .catch(() => {});
    };
    return subscribeGenerationRecoveryTriggers(
      globalThis,
      document,
      recoverCurrentThread,
    );
  }, [aui]);

  // Mirror Data-tab attachment deletions into the loaded thread: the in-memory repository
  // otherwise keeps the attachment, and a later repo-to-storage sync would write it back.
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

              // Preserve any messages added or streamed while WebCrypto ran; retry if the target's managed
              // content itself changed.
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
                            (attachment) => attachment.id !== attachmentId,
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
              attachments.some((attachment) => attachment.id === attachmentId)
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
        const completeLoad = <T,>(result: T, loadedThreadId?: string): T => {
          // A runtime bootstraps on an empty thread before switching to the requested one, so an
          // unrequested load is not readiness.
          const loadedTheRequestedThread =
            !reloadReadyThreadId || loadedThreadId === reloadReadyThreadId;
          if (onInitialHistoryReady) {
            if (loadedTheRequestedThread) onInitialHistoryReady();
          } else if (
            modelType === "base" &&
            !pairId &&
            loadedTheRequestedThread
          ) {
            window.dispatchEvent(new Event("unsloth:app-shell-ready"));
          }
          return result;
        };
        const { remoteId } = aui.threadListItem().getState();
        if (!remoteId) {
          return completeLoad({ messages: [] });
        }
        const roleOrder: Record<string, number> = {
          system: 0,
          user: 1,
          assistant: 2,
        };
        let msgs: MessageRecord[];
        let activeGenerationRuns: ChatGenerationRun[];
        let activeGenerationRunsLoaded: boolean;
        try {
          const snapshot = await loadGenerationOverlaySnapshot(
            remoteId,
            getActiveChatGenerationRuns,
            listStoredChatMessages,
          );
          msgs = snapshot.messages;
          activeGenerationRuns = snapshot.activeRuns;
          activeGenerationRunsLoaded = snapshot.activeRunsLoaded;
        } catch (error) {
          if (!isExpectedBackgroundChatStorageError(error)) {
            throw error;
          }
          msgs = [];
          activeGenerationRuns = [];
          activeGenerationRunsLoaded = false;
        }
        // The endpoint is named for active runs but returns rows, so a run that terminalised between
        // the write and this read comes back with it, and force-writing `generationSettled: false`
        // over a finished reply revives it. Take the still-live rows only; a failed read is silence.
        // Publish the still-live rows as the corroboration toThreadMessage restores a running status from.
        activeGenerationRuns = activeGenerationRuns.filter(
          (run) => !isTerminalChatGenerationRun(run),
        );
        // And drop any run the message snapshot already shows as finished: the runs read happens first,
        // so a run that completed in between is still listed. Filtering the LIST rather than
        // skipping at the overlay matters, since a stale mapping keeps the thread reading as durable
        // and a later subscriber-owned stream would be capped.
        const terminalMessageRuns = new Set(
          msgs
            .filter((message) => {
              const custom = (message.metadata ?? {}) as Record<string, unknown>;
              const status = custom.generationStatus;
              return (
                status === "completed" ||
                status === "failed" ||
                status === "cancelled"
              );
            })
            .map((message) => message.id),
        );
        activeGenerationRuns = activeGenerationRuns.filter(
          (run) => !terminalMessageRuns.has(run.assistantMessageId),
        );
        // The runs read happens BEFORE the messages read, so a run created between the two would read
        // as interrupted and re-enable the composer for a thread still generating. Close that gap
        // with one more lookup, only when a message names a run the first read missed.
        if (!activeGenerationRunsLoaded) {
          // The initial read failed, so this thread has no current answer. Retract any answer from an
          // earlier load: another tab may have started a run since, and the stale answer would restore
          // that live reply as interrupted.
          markServerActiveGenerationRunsUnknown(remoteId);
        }
        if (activeGenerationRunsLoaded) {
          let answered = true;
          const known = new Set(activeGenerationRuns.map((run) => run.id));
          const missed = msgs.some((message) => {
            const custom = (message.metadata ?? {}) as Record<string, unknown>;
            const runId = custom.generationRunId;
            const status = custom.generationStatus;
            return (
              typeof runId === "string" &&
              !known.has(runId) &&
              (status === "queued" ||
                status === "running" ||
                status === "cancelling")
            );
          });
          if (missed) {
            try {
              const second = await getActiveChatGenerationRuns(remoteId);
              activeGenerationRuns = second.filter(
                (run) => !isTerminalChatGenerationRun(run),
              );
            } catch {
              // `missed` already proved the first list predates this run, so it is not an answer either.
              // Leave the thread unanswered rather than promoting a list known to be stale.
              answered = false;
              markServerActiveGenerationRunsUnknown(remoteId);
            }
          }
          if (answered) {
            syncServerActiveGenerationRuns(
              remoteId,
              activeGenerationRuns.map((run) => run.id),
            );
          }
        }
        for (const run of activeGenerationRuns) {
          const assistant = msgs.find(
            (message) => message.id === run.assistantMessageId,
          );
          if (!assistant) continue;

          assistant.metadata = {
            ...(assistant.metadata ?? {}),
            generationRunId: run.id,
            generationStatus: run.status,
            generationSettled: false,
            serverManaged: true,
            // The server still has this run, which overrules a follower that gave up locally. Clearing the
            // marker here is what lets a genuinely slow run be picked back up.
            generationLocallyInterrupted: false,
          };
        }
        // Durable research can outlive this runtime, so reattach its server-owned assistant message to
        // the inline card after navigation or refresh.
        const researchThreadState = await getResearchThreadState(
          remoteId,
        ).catch(() => null);
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
        for (const message of msgs) {
          if (
            message.role === "assistant" &&
            typeof (message.metadata as Record<string, unknown> | undefined)
              ?.generationRunId === "string"
          ) {
            scheduleGenerationRecovery(
              remoteId,
              {
                ...message,
                content: cloneContent(message.content),
                metadata: { ...(message.metadata ?? {}) },
              },
              aui,
            );
          }
        }

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
        // providers have loadedContextLength === null. llama.cpp stops at the window, so
        // a saved count past it is stale; MLX runs past it by design, and a thread whose
        // recount is unsupported -- a VLM, or one carrying tool history -- would never
        // get another count to replace the one rejected here.
        const localLimit = store.loadedIsGguf ? store.loadedContextLength : null;
        const withinLocalLimit =
          !localLimit || (savedUsage?.totalTokens ?? 0) <= localLimit;
        // Legacy unscoped usage (no modelId) is trusted only when a known local
        // window bounds the totals, so an old local turn can't be misattributed
        // to a newly-selected external provider.
        const modelMatches = savedUsage?.modelId
          ? savedUsage.modelId === store.params.checkpoint
          : typeof store.loadedContextLength === "number" &&
            store.loadedContextLength > 0;
        // The value, not a boolean: the writes below need the narrowing.
        const restoredUsage =
          savedUsage && withinLocalLimit && modelMatches ? savedUsage : null;
        if (restoredUsage) {
          // Key by the thread this loader read, not whichever is active when the await resolves: a switch
          // inside it would file this thread's usage under the incoming one.
          store.setThreadContextUsage(remoteId, restoredUsage);
          if (store.activeThreadId === remoteId) {
            store.setContextUsage(restoredUsage);
          }
        }
        // Only when nothing was restored: saved usage is the last completion's exact totals, and
        // refreshContextUsage does NOT stand down for usage already there, so it would overwrite
        // them with an estimate whose completionTokens is 0. A thread opened after a model switch
        // fails modelMatches and still gets priced (#7450). Primary pane only: a compare pane
        // never owns the global bar.
        if (!restoredUsage && modelType === "base" && !pairId) {
          void refreshContextUsage({ threadId: remoteId });
        }

        // If any message has a stored parentId, reconstruct the tree so retries load as branches rather
        // than a flat list, inferring sequential parents for old messages in mixed threads. Fall
        // back to fromArray for fully legacy threads.
        const hasParentIds = msgs.some((m) => m.parentId != null);
        if (hasParentIds) {
          const resolveParent = createParentResolver();
          return completeLoad(
            {
              messages: msgs.map((m) => ({
                parentId: resolveParent(m),
                message: toThreadMessage(m),
              })),
            },
            remoteId,
          );
        }
        return completeLoad(
          ExportedMessageRepository.fromArray(msgs.map(toThreadMessage)),
          remoteId,
        );
      },

      append({ parentId, message }: ExportedMessageRepositoryItem) {
        const localThreadId = aui.threadListItem().getState().id;
        const historyClearGeneration = chatHistoryClearBoundary.capture();
        const throwIfHistoryWasCleared = async (remoteId: string) => {
          if (chatHistoryClearBoundary.capture() === historyClearGeneration) {
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
          // The model run waits for the authoritative row. Clear-all does not: it tombstones this known
          // id directly, so a stalled request cannot hold the clear hostage.
          await awaitStoredChatThreadWrites(remoteId);
          if (isChatThreadDeleted(remoteId)) {
            await deleteStoredChatThreads([remoteId]);
            return;
          }
          // Published before the reads below: a temporary chat has no row to confirm, and a read that
          // fails must not leave the runtime pointing at the previously open chat. Not while this pane
          // is only mounted to keep its run attached, since a hidden pane naming itself active makes
          // Export pull the unrelated base chat. Nor mid-switch: switchToNewThread() is async, so
          // mainThreadId is still the OUTGOING thread for that gap, which the attempt check catches.
          // Nor mid-switch: switchToNewThread() is async, so mainThreadId is still the OUTGOING thread for
          // the whole gap, and a write landing there republishes the chat the user just left.
          // attempt !== landedAttempt is that gap.
          const switchState = newThreadSwitchStateRef?.current;
          const switchInFlight = Boolean(
            switchState &&
              switchState.activeNonce !== null &&
              switchState.landedAttempt !== switchState.attempt,
          );
          if (
            modelType === "base" &&
            !pairId &&
            !backgroundedRef?.current &&
            !switchInFlight
          ) {
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
          // One point read: the model run waits on this write.
          const existingMessage = await getStoredChatMessage(
            remoteId,
            message.id,
          );
          await throwIfHistoryWasCleared(remoteId);
          const content = cloneContent(message.content);
          const attachments =
            message.role === "user"
              ? cloneAttachments(message.attachments)
              : [];
          const custom = message.metadata?.custom;
          const createdAt =
            existingMessage?.createdAt ??
            message.createdAt?.getTime?.() ??
            Date.now();
          const existingMetadata = existingMessage?.metadata;
          const incomingRevision = Number(
            (custom as Record<string, unknown> | undefined)?.serverRevision ??
              -1,
          );
          const existingRevision = Number(
            existingMetadata?.serverRevision ?? -1,
          );
          const incomingMetadata = custom as
            | Record<string, unknown>
            | undefined;
          const sameResearchRun =
            typeof existingMetadata?.researchRunId === "string" &&
            existingMetadata.researchRunId === incomingMetadata?.researchRunId;
          const sameGenerationRun =
            typeof existingMetadata?.generationRunId === "string" &&
            existingMetadata.generationRunId ===
              incomingMetadata?.generationRunId;
          const preserveGeneration = shouldPreserveGenerationMetadata(
            existingMetadata,
            incomingMetadata,
          );
          const preserveServerManaged =
            existingMetadata?.serverManaged === true &&
            (preserveGeneration ||
              sameResearchRun ||
              !incomingMetadata?.serverManaged ||
              existingRevision > incomingRevision);
          // The backend owns this message and refuses client edits, and every field the save would send
          // was just read back from it, so the request is answered 409 every time: one measured 43.6 s
          // generation cost 265 PUTs, 256 rejected. `parentId` is the one field not echoed back, and
          // it is dropped either way, since the server rejects the whole request.
          // One measured 43.6 s generation: 265 PUTs, 256 rejected, plus 353 whole-thread GETs from the
          // ensureStoredChatThread inside saveStoredChatMessage. Returning here drops both.
          if (preserveServerManaged) {
            await throwIfHistoryWasCleared(remoteId);
            return;
          }
          await saveStoredChatMessage({
            id: message.id,
            threadId: remoteId,
            parentId: parentId ?? null,
            role: message.role,
            content,
            ...(attachments.length > 0 && { attachments }),
            ...(incomingMetadata && { metadata: incomingMetadata }),
            createdAt,
          });
          await throwIfHistoryWasCleared(remoteId);
        })();
        return trackHistoryAppend(message.id, write);
      },
    }),
    [
      aui,
      backgroundedRef,
      modelType,
      newThreadSwitchStateRef,
      onInitialHistoryReady,
      pairId,
      reloadReadyThreadId,
    ],
  );

  // Always register the adapter so the mic stays clickable for any engine: the engine is resolved
  // at listen() time and the composer shows guidance when it cannot run.
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
          // Before the document adapters: a composite takes the first match, and .mkv/.mov must not fall
          // through to them.
          new VideoAttachmentAdapter(),
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
  reloadReadyThreadId?: string,
  onInitialHistoryReady?: () => void,
  backgroundedRef?: { current: boolean },
  newThreadSwitchStateRef?: { current: NewThreadSwitchState },
): ReturnType<typeof useLocalRuntime> {
  const adapters = useStudioRuntimeAdapters(
    modelType,
    pairId,
    reloadReadyThreadId,
    onInitialHistoryReady,
    backgroundedRef,
    newThreadSwitchStateRef,
  );
  const persistedChatAdapter = useMemo(
    () =>
      createPersistedRunAdapter(
        createOpenAIStreamAdapter({ modelType, pairId }),
      ),
    [modelType, pairId],
  );
  return useLocalRuntime(persistedChatAdapter, { adapters });
}

function createRuntimeHook(
  modelType: ModelType,
  pairId?: string,
  reloadReadyThreadId?: string,
  onInitialHistoryReady?: () => void,
  backgroundedRef?: { current: boolean },
  newThreadSwitchStateRef?: { current: NewThreadSwitchState },
) {
  return function useConfiguredRuntimeHook(): ReturnType<
    typeof useLocalRuntime
  > {
    return useRuntimeHook(
      modelType,
      pairId,
      reloadReadyThreadId,
      onInitialHistoryReady,
      backgroundedRef,
      newThreadSwitchStateRef,
    );
  };
}

// Only bounds the pathological case: a switch that never settles never spends its claim.
const MAX_PENDING_SAVED_THREAD_SWITCHES = 16;

type PendingSavedThreadSwitch = { id: string; settled: boolean };

type NewThreadSwitchState = {
  activeNonce: string | null;
  hasSwitched: boolean;
  // Bumped by every switch this provider starts. The nonce alone does not identify an attempt:
  // leaving for a saved chat and coming back releases it, so two switches for the SAME nonce
  // can be in flight.
  attempt: number;
  // One entry per switch STARTED, so a nonce view recognises any of these ids landing after the
  // route moved on. By id rather than id SHAPE, which cannot tell a stale arrival from a fresh
  // switch's thread. `settled` retires a claim with its own switch.
  pendingSavedThreadIds: PendingSavedThreadSwitch[];
  // The thread a nonce view is on, with the nonce it belongs to. Once the user has sent,
  // assistant-ui's newThreadId is gone and switchToNewThread() mints a SECOND blank thread; a
  // ?new= URL survives materialization, so the pair is what tells a returning nonce from a new.
  nonceThread: { nonce: string; threadId: string } | null;
  // The newest attempt whose own switch has LANDED. Ownership is only recorded from a thread this
  // nonce's switch actually opened: entering a nonce from a saved chat leaves that chat as
  // `mainThreadId` until switchToNewThread() resolves, so "unclaimed and current" is no proof.
  landedAttempt: number;
};

function ThreadAutoSwitch({
  threadId,
  syncActiveThreadId = true,
  paused,
  newThreadSwitchStateRef,
  onSwitchFailed,
}: {
  threadId: string;
  syncActiveThreadId?: boolean;
  paused: boolean;
  newThreadSwitchStateRef: { current: NewThreadSwitchState };
  onSwitchFailed?: () => void;
}): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);

  useEffect(() => {
    // Paused as well as loading: requestTemporaryPromptQueueStop() names every temporary queue on
    // the page, not this provider's, so a backgrounded pane would stop the on-screen queue.
    if (isLoading || paused) {
      return;
    }
    newThreadSwitchStateRef.current.activeNonce = null;
    if (mainThreadId !== threadId) {
      // Bumped, not read: this resolves asynchronously into a shared provider, so a newer switch of
      // EITHER kind must supersede it. One shared token let a late rejection detach the chat.
      const attemptAtStart = (newThreadSwitchStateRef.current.attempt += 1);
      // One entry per switch started, duplicates included: A, B, A starts two A switches and both can
      // land, so each arrival must spend exactly one entry.
      const claims = newThreadSwitchStateRef.current.pendingSavedThreadIds;
      const claim: PendingSavedThreadSwitch = { id: threadId, settled: false };
      claims.push(claim);
      // Oldest first: the longest-outstanding switch is least likely to still take a view.
      if (claims.length > MAX_PENDING_SAVED_THREAD_SWITCHES) {
        claims.splice(0, claims.length - MAX_PENDING_SAVED_THREAD_SWITCHES);
      }
      // Saved chats keep running in the background, but a temporary chat is unreachable after this
      // switch and must not retain an active queue.
      requestTemporaryPromptQueueStop();
      const switchResult = aui.threads().switchToThread(threadId) as unknown;
      if (
        switchResult &&
        typeof (switchResult as Promise<void>).then === "function"
      ) {
        // Both arms retire the claim, because both end the switch: a rejected switch never assigns a
        // main thread, so its claim could otherwise sit armed for ever.
        void (switchResult as Promise<void>).then(
          () => {
            claim.settled = true;
          },
          () => {
            claim.settled = true;
            // Ahead of the staleness guard, deliberately (#9251): this releases the retained reload shell,
            // and a superseded attempt still ended.
            onSwitchFailed?.();
            // Only if this switch is still the current one: unguarded, a rejection landing after the user
            // moved to a project landing cleared the active id that view had just set.
            if (newThreadSwitchStateRef.current.attempt !== attemptAtStart) return;
            if (syncActiveThreadId) {
              useChatRuntimeStore.getState().setActiveThreadId(null);
            }
          },
        );
      } else {
        // A synchronous switch is already over by the time this line runs.
        claim.settled = true;
      }
    }
  }, [
    aui,
    isLoading,
    mainThreadId,
    newThreadSwitchStateRef,
    onSwitchFailed,
    paused,
    syncActiveThreadId,
    threadId,
  ]);

  useEffect(() => {
    if (isLoading || mainThreadId !== threadId) {
      return;
    }
    // The switch landed while this view is still mounted, so it was not stale. Released here rather
    // than in the promise, since this effect only runs while the saved chat is on screen. Every
    // SETTLED claim, not just this thread's: in-flight ones can still land wrong.
    const state = newThreadSwitchStateRef.current;
    state.pendingSavedThreadIds = state.pendingSavedThreadIds.filter(
      (claim) => !claim.settled,
    );
    if (!syncActiveThreadId) {
      return;
    }
    useChatRuntimeStore.getState().setActiveThreadId(threadId);
  }, [
    isLoading,
    mainThreadId,
    newThreadSwitchStateRef,
    syncActiveThreadId,
    threadId,
  ]);

  return null;
}

function ThreadNewChatSwitch({
  nonce,
  paused,
  newThreadSwitchStateRef,
}: {
  nonce: string;
  paused: boolean;
  newThreadSwitchStateRef: { current: NewThreadSwitchState };
}): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const loadedContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  // Read only by the recount below: New Chat itself must not care whether a run is still going.
  const runActive = useChatRuntimeStore((s) =>
    Object.values(s.runningByThreadId).some(Boolean),
  );
  // The outgoing thread is not read here: New Chat leaves it running.
  useEffect(() => {
    if (isLoading || paused) {
      return;
    }
    const switchState = newThreadSwitchStateRef.current;
    if (switchState.activeNonce === nonce) {
      return;
    }
    // Only this nonce's own thread, and only once sent to: a blank placeholder is still replaced.
    const recorded =
      switchState.nonceThread?.nonce === nonce
        ? switchState.nonceThread.threadId
        : null;
    const runtimeThreads = aui.threads().__internal_getAssistantRuntime?.();
    // Guarded, unlike the reads elsewhere that pass an id the runtime just handed back: this one is
    // REMEMBERED in a ref that outlives every view switch, and getItemById() THROWS for a
    // dropped id rather than returning undefined, so the optional chain would not catch it.
    let recordedRemoteId: string | undefined;
    if (recorded) {
      try {
        recordedRemoteId = runtimeThreads?.threads
          .getItemById(recorded)
          .getState()?.remoteId;
      } catch {
        // The thread this nonce remembers is gone; treat it as a nonce that owns nothing.
        recordedRemoteId = undefined;
      }
    }
    // A tombstoned chat is not one to return to: deletion leaves the runtime item and its remoteId
    // intact, so "the store still knows it" is not "it still exists".
    // Falling through clears nonceThread below, which is what a nonce owning nothing looks like.
    const returningToOwnChat = Boolean(
      recorded && recordedRemoteId && !isChatThreadDeleted(recordedRemoteId),
    );
    if (!returningToOwnChat) {
      // A new nonce owns nothing yet; the old record must not survive into it.
      switchState.nonceThread = null;
    }
    const shouldClearAttachments = switchState.hasSwitched;
    const clearAfterSwitch =
      shouldClearAttachments && switchState.activeNonce === null;
    const attempt = switchState.attempt + 1;
    switchState.attempt = attempt;
    switchState.activeNonce = nonce;
    switchState.hasSwitched = true;
    const clearAttachments = () => {
      try {
        // Chained, not just called: clearAttachments() removes each staged file through the attachment
        // adapter, so a rejecting remove() would go unhandled.
        void Promise.resolve(aui.composer().clearAttachments()).catch(
          () => undefined,
        );
      } catch {
        // No thread mounted yet, so there is no composer to carry anything over.
      }
    };
    if (shouldClearAttachments && !clearAfterSwitch) {
      clearAttachments();
    }
    // Saved chats keep running in the background, but a temporary chat is never persisted, so
    // abandoning it must also discard its otherwise unreachable queue.
    requestTemporaryPromptQueueStop();
    // A reopen is as cancellable as a saved switch: a project landing remounts with the OLD nonce
    // and rotates it in an effect, so the reopen is in flight when the rotation starts its own.
    let reopenClaim: PendingSavedThreadSwitch | null = null;
    if (returningToOwnChat && recorded) {
      reopenClaim = { id: recorded, settled: false };
      switchState.pendingSavedThreadIds.push(reopenClaim);
      if (
        switchState.pendingSavedThreadIds.length >
        MAX_PENDING_SAVED_THREAD_SWITCHES
      ) {
        switchState.pendingSavedThreadIds.splice(
          0,
          switchState.pendingSavedThreadIds.length -
            MAX_PENDING_SAVED_THREAD_SWITCHES,
        );
      }
    }
    // Dropped when this nonce is still on screen: the reopen did its job and an armed claim would
    // have the correction undo it.
    const settleReopenClaim = () => {
      if (!reopenClaim) return;
      reopenClaim.settled = true;
      const switchStateNow = newThreadSwitchStateRef.current;
      if (switchStateNow.activeNonce !== nonce) return;
      const at = switchStateNow.pendingSavedThreadIds.indexOf(reopenClaim);
      if (at !== -1) switchStateNow.pendingSavedThreadIds.splice(at, 1);
    };
    // Switch to a fresh local thread without persisting it yet; persistence still happens on first message append.
    void Promise.resolve(
      returningToOwnChat && recorded
        ? aui.threads().switchToThread(recorded)
        : aui.threads().switchToNewThread(),
    ).then(
      () => {
        settleReopenClaim();
        // This attempt's own switch has landed, so the main thread from here on is one it opened. Only
        // for the CURRENT attempt: a superseded switch landing late says nothing.
        {
          const switchStateNow = newThreadSwitchStateRef.current;
          if (switchStateNow.attempt === attempt) {
            switchStateNow.landedAttempt = attempt;
          }
        }
        if (!clearAfterSwitch) return;
        const switchStateNow = newThreadSwitchStateRef.current;
        // By attempt as well as nonce, matching the rejection arm: a saved-thread detour releases the
        // nonce, so returning starts a newer attempt and this older completion would otherwise wipe
        // an attachment the newer one staged.
        if (switchStateNow.attempt !== attempt) return;
        if (switchStateNow.activeNonce !== nonce) return;
        clearAttachments();
      },
      () => {
        settleReopenClaim();
        // The fresh thread never opened, so the view is still on the outgoing one. Release the nonce, or
        // the guard at the top reads it as served and the same New Chat can never be retried.
        const switchStateNow = newThreadSwitchStateRef.current;
        // By attempt, not by nonce alone: a later switch for the same nonce owns the thread it opened,
        // and releasing it would switch away from a chat in use.
        if (
          switchStateNow.attempt === attempt &&
          switchStateNow.activeNonce === nonce
        ) {
          switchStateNow.activeNonce = null;
        }
      },
    );
    useChatRuntimeStore.getState().setActiveThreadId(null);
  }, [aui, isLoading, newThreadSwitchStateRef, nonce, paused]);

  // Reassert this view's thread when a switch started before it lands anyway. Leaving a landing
  // for a saved chat and coming back before switchToThread resolves used to end with
  // assistant-ui assigning that saved thread as the main one, since the promise does not know
  // the route moved. Recognised by the exact id that switch asked for.
  useEffect(() => {
    if (isLoading || paused) {
      return;
    }
    const switchState = newThreadSwitchStateRef.current;
    if (switchState.activeNonce !== nonce) {
      return;
    }
    // Already on the thread this nonce owns, and the claim here is the sibling effect's OWN reopen:
    // switchToThread() early-returns when its target is already current (assistant-ui#2577), so
    // the reopen resolves on a microtask and its claim is still outstanding in the same commit.
    if (
      mainThreadId &&
      switchState.nonceThread?.nonce === nonce &&
      switchState.nonceThread.threadId === mainThreadId
    ) {
      return;
    }
    const claimed = mainThreadId
      ? switchState.pendingSavedThreadIds.findIndex((claim) => claim.id === mainThreadId)
      : -1;
    if (claimed === -1) {
      // Not a stale arrival, so this is the thread this view owns. Recorded here, not at the switch:
      // the id changes on materialization and a reattach needs the persisted one.
      if (mainThreadId && switchState.landedAttempt === switchState.attempt) {
        switchState.nonceThread = { nonce, threadId: mainThreadId };
      }
      return;
    }
    switchState.pendingSavedThreadIds.splice(claimed, 1);
    const reattachTo =
      switchState.nonceThread?.nonce === nonce
        ? switchState.nonceThread.threadId
        : null;
    void Promise.resolve(
      reattachTo && reattachTo !== mainThreadId
        ? aui.threads().switchToThread(reattachTo)
        : aui.threads().switchToNewThread(),
    ).catch(() => undefined);
  }, [aui, isLoading, mainThreadId, newThreadSwitchStateRef, nonce, paused]);

  // The effect above blanks the bar, and this view reaches no other recount trigger. Keyed on the
  // model too: on a RELOAD of /chat?new=<uuid> nothing is known until status answers.
  useEffect(() => {
    if (
      isLoading ||
      paused ||
      modelLoading ||
      runActive ||
      !checkpoint ||
      loadedContextLength == null
    ) {
      return;
    }
    const store = useChatRuntimeStore.getState();
    if (store.activeThreadId != null || store.contextUsage != null) return;
    void refreshContextUsage();
    // nonce: a fresh New Chat click re-runs the effect above, which blanks the bar again. runActive
    // is a DEPENDENCY, not just a guard: refreshContextUsage declines while anything generates,
    // and nothing else re-fires this when the run ends.
  }, [
    checkpoint,
    loadedContextLength,
    isLoading,
    modelLoading,
    nonce,
    paused,
    runActive,
  ]);

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

function NonceThreadResumeRestore({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const aui = useAui();
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const wasEnabledRef = useRef(enabled);

  useEffect(() => {
    const resumed = enabled && !wasEnabledRef.current;
    wasEnabledRef.current = enabled;
    if (!resumed) {
      return;
    }
    // Only ever fills a hole. A restore that overwrote a live id would fight whoever set it.
    if (useChatRuntimeStore.getState().activeThreadId != null) {
      return;
    }
    // Published raw, as ActiveThreadSync does on the paths this stands in for. A `__LOCALID_` id is
    // NOT a reason to skip: initialize() writes the row under whatever id assistant-ui minted.
    if (!mainThreadId) {
      return;
    }
    // ...but an UNTOUCHED landing must stay untouched: a compare round trip leaves a blank
    // placeholder here, and publishing it makes ProjectLanding swap the overview for an empty
    // Thread. On remoteId, not id shape.
    const runtime = aui.threads().__internal_getAssistantRuntime?.();
    const { remoteId } =
      runtime?.threads.getItemById(mainThreadId).getState() ?? {};
    if (!remoteId) {
      return;
    }
    // ...and neither must a chat the user deleted while away. Unsloth deletes by tombstoning storage
    // rather than calling runtime.threads.delete(), so every check above still passes. On
    // remoteId, which is the id storage and the sidebar delete agree on.
    if (isChatThreadDeleted(remoteId)) {
      return;
    }
    useChatRuntimeStore.getState().setActiveThreadId(mainThreadId);
  }, [aui, enabled, mainThreadId]);

  return null;
}

// A thread read that fails leaves the chat unpaired, so it is worth a couple of goes.
const THREAD_READ_RETRY_MS = 1_500;
const THREAD_READ_RETRIES = 2;
// And one that never answers at all has to become a failure, or it holds sends forever.
const THREAD_READ_TIMEOUT_MS = 8_000;

// gated on hydration, or the initial /api/chat/settings response lands after a thread's own values.
function ThreadScopedSettingsSync({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const pendingNewThreadId = useAuiState(({ threads }) => threads.newThreadId);
  const settingsHydrated = useChatRuntimeStore(
    (state) => state.settingsHydrated,
  );

  useEffect(() => {
    const { applyThreadScopedSettings } = useChatRuntimeStore.getState();
    // A chat not yet sent to has no row, so pairing it holds every edit behind a read certain to
    // 404. The `__LOCALID_` prefix stays on the id for good, so only the runtime's
    // pending-new-thread id tells the two apart.
    if (activeThreadId !== null && activeThreadId === pendingNewThreadId) {
      applyThreadScopedSettings(null, null);
      return;
    }
    if (!enabled) {
      // Compare panes share one composer between two threads, so no single chat's snapshot could
      // apply: compare runs on the installation defaults and its edits move them. Said here rather
      // than leaving the module pointing at the last single chat.
      // Otherwise the module still points at the last single chat, whose stored pills a model load would
      // read back through threadScopedOverride.
      applyThreadScopedSettings(null, null);
      return;
    }
    if (activeThreadId === null) {
      if (settingsHydrated) applyThreadScopedSettings(null, null);
      return;
    }
    // The composer is interactive while /api/chat/settings is still out, so start holding this
    // chat's edits as soon as its id is known; waiting for hydration left that window writing
    // edits into the installation defaults.
    beginThreadScopedPairing(activeThreadId);
    if (!settingsHydrated) {
      return () => {
        // Hydration finishing re-runs this effect for the same chat, and the held edits are still
        // waiting for that chat's read, so keep holding them. Any other reason to leave means the
        // chat is going away.
        const now = useChatRuntimeStore.getState();
        if (!now.settingsHydrated || now.activeThreadId !== activeThreadId) {
          commitHeldThreadScopedEditsToTheirThread();
        }
      };
    }
    let cancelled = false;
    let paired = false;
    let unpaired = false;
    let defaulted = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let retriesLeft = THREAD_READ_RETRIES;
    // One per attempt, aborted when the attempt's deadline passes and when the chat is left. Without
    // it the losing side of the race stayed open on the server for the full write timeout.
    const reads = new Set<AbortController>();
    const abortReads = () => {
      for (const read of reads) read.abort();
      reads.clear();
    };

    const sync = () => {
      if (cancelled || paired) return;
      // The composer is live while this read is out, so hold any edit made meanwhile rather than
      // writing it to the installation defaults. Drop to those defaults for the duration: until
      // the read lands the store still holds the OUTGOING chat's values, and a send in that window
      // would carry the previous chat's permission level and pills.
      // A send in that window is captured by snapshotQueuedChatRunSettings and carries the previous
      // chat's permission level and pills, so a chat stored as "ask" could run tools without asking.
      if (!defaulted) {
        defaulted = true;
        applyThreadScopedSettings(null, null);
      }
      beginThreadScopedPairing(activeThreadId);
      // Settle this chat's own PATCH first: edit a chat, leave, come straight back and the read can
      // overtake the write and return the pre-edit snapshot.
      const read = new AbortController();
      reads.add(read);
      // The deadline covers the WAITS as well as the read, neither of which is bounded on its own; in
      // front of it their time went uncounted and the chain could outlast THREAD_PAIRING_WAIT_MS
      // with the gate shut. Inside it, a stall is one failed attempt.
      // The settings chain is a PATCH, and awaitStoredChatThreadWrites settles a row write opening with
      // an unbounded getStoredChatThread. Inside the deadline a stall is one failed attempt, which
      // retryThreadRead handles.
      void Promise.race([
        Promise.all([
          // This chat's own PATCH first: a read that overtakes it returns the pre-edit snapshot.
          awaitThreadScopedSettingsWrite(activeThreadId),
          // And its row, which may not exist yet: initialize() resolves as soon as the id is minted and
          // leaves the POST tracked, so on a first send a read can overtake it and release this chat's
          // held edits into the installation defaults.
          awaitStoredChatThreadWrites(activeThreadId),
        ]).then(() =>
          getStoredChatThreadReadResult(activeThreadId, {
            timeoutMs: THREAD_READ_TIMEOUT_MS,
            signal: read.signal,
          }),
        ),
        new Promise<never>((_, reject) =>
          setTimeout(() => {
            // The waits can expire before the fetch is issued; the controller is the only thing stopping one
            // issued a moment later from outliving them.
            read.abort();
            reject(new Error("thread settings read timed out"));
          }, THREAD_READ_TIMEOUT_MS),
        ),
      ])
        .finally(() => {
          reads.delete(read);
        })
        .then(({ thread, cacheable }) => {
          if (cancelled || paired) return;
          // A legacy fallback row means the backend GET FAILED and Dexie answered instead: that is the
          // failure case, not a confirmed missing row, so keep holding and retry.
          if (thread && !cacheable) {
            retryThreadRead();
            return;
          }
          // a legacy fallback row carries no snapshot, and pinning would overwrite the real one.
          if (!thread) {
            // A new chat's runtime-made id has no row yet, so it stays on the global settings. The answer is
            // in, so an edit held for it is a plain default change and goes out now; deferring that
            // wrote an unsaved chat's click to a row that does not exist.
            releaseHeldThreadScopedEdits();
            if (unpaired) return;
            unpaired = true;
            applyThreadScopedSettings(null, null);
            return;
          }
          paired = true;
          // the response spells every omitted field as null, which is not a value to apply.
          applyThreadScopedSettings(
            activeThreadId,
            thread.settings
              ? sanitizeThreadScopedSettings(thread.settings)
              : null,
          );
        })
        // A failed read leaves the installation defaults up, not the outgoing chat's settings, which
        // would otherwise stay live indefinitely. The held edit goes to the chat it was made in.
        .catch(() => retryThreadRead());
    };

    // The read did not answer for this chat: send what is held to the chat it was made in, then
    // keep the chat paired, or every later edit would fall through to the installation defaults.
    // Retry a bounded few times.
    const retryThreadRead = () => {
      if (cancelled) return;
      commitHeldThreadScopedEditsToTheirThread();
      if (retryTimer !== null) return;
      if (retriesLeft <= 0) {
        // Out of tries. Staying paired would hold every send behind "Loading this chat's settings" with
        // nothing left to resolve it, so give up openly and say so once.
        applyThreadScopedSettings(null, null);
        releaseHeldThreadScopedEdits();
        toast.error("Could not load this chat's settings", {
          description:
            "It is using the default settings. Reopen the chat to try again.",
        });
        return;
      }
      // Keep the chat paired between tries, or an edit in it would fall through to the installation defaults.
      beginThreadScopedPairing(activeThreadId);
      retriesLeft -= 1;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        sync();
      }, THREAD_READ_RETRY_MS);
    };

    sync();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, sync);
    return () => {
      cancelled = true;
      if (retryTimer !== null) clearTimeout(retryTimer);
      // Nothing is waiting on these once the chat is gone, and leaving them running is how an outage
      // turned every chat opened during it into three open requests.
      abortReads();
      // switched away mid-read: the edit belongs to the chat it was made in, not to the installation defaults.
      commitHeldThreadScopedEditsToTheirThread();
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, sync);
    };
  }, [activeThreadId, enabled, pendingNewThreadId, settingsHydrated]);

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
// it: a model change empties contextUsageByThreadId while a mounted thread does not rerun
// its history loader, and on a deep link the loader and status can each land before the other.
function ThreadContextUsageRecount({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const loadedContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  // A DEPENDENCY, not just a guard: nothing else here changes when a run ends, so a count skipped
  // for being busy would never be retried. Every run, since that is what the endpoint refuses on.
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
      loadedContextLength == null
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
    loadedContextLength,
    runActive,
    modelLoading,
  ]);

  return null;
}

// Exposes the current thread's cancelRun() via the shared store so external surfaces can stop an
// in-flight stream before deleting the thread.
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
      new Set(
        [mainThreadId, remoteThreadId].filter((id): id is string =>
          Boolean(id),
        ),
      ),
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
      // assistant-ui enters its running state before adapter preflight turns on runningByThreadId, so
      // keep the only cancel handle after navigation and release it when the run ends.
      let unsubscribe = () => {};
      unsubscribe = thread.subscribe(() => {
        if (thread.getState().isRunning) {
          return;
        }
        for (const threadId of threadIds) {
          useChatRuntimeStore.getState().clearThreadCancel(threadId, cancel);
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
  backgrounded,
  newThreadSwitchStateRef,
}: {
  modelType: ModelType;
  pairId?: string;
  backgrounded: boolean;
  newThreadSwitchStateRef: { current: NewThreadSwitchState };
}): ReactElement | null {
  const aui = useAui();
  const saveChainRef = useRef(Promise.resolve());
  const pendingFirstSavesRef = useRef(new Map<string, Promise<void>>());
  // A ref, not a saveThread dependency: the save may be queued while visible and resolve after
  // Compare hides the pane, so this must read at PUBLISH time.
  const backgroundedRef = useRef(backgrounded);
  backgroundedRef.current = backgrounded;

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

      // The save still runs while backgrounded; only the PUBLICATION is suppressed, or a hidden pane
      // reaches Compare's exportThreadIds. Same stand-down mid-switch: mainThreadId is still the
      // OUTGOING thread until switchToNewThread() resolves.
      const switchState = newThreadSwitchStateRef.current;
      const switchInFlight =
        switchState.activeNonce !== null &&
        switchState.landedAttempt !== switchState.attempt;
      if (
        modelType === "base" &&
        !pairId &&
        !backgroundedRef.current &&
        !switchInFlight
      ) {
        const store = useChatRuntimeStore.getState();
        const activeThreadId = runtime.threads.getState().mainThreadId;
        if (activeThreadId === threadId && store.activeThreadId !== remoteId) {
          store.setActiveThreadId(remoteId);
        }
      }
    },
    [aui, modelType, newThreadSwitchStateRef, pairId],
  );

  // Let checkpoints schedule their next timer after this write settles.
  const queueSave = useCallback(
    (threadId: string): Promise<void> => {
      const queued = saveChainRef.current
        .catch(() => {})
        .then(async () => {
          await pendingFirstSavesRef.current.get(threadId);
          await saveThread(threadId);
        })
        .catch(reportAutosaveError);
      saveChainRef.current = queued;
      return queued;
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

  // runEnd only reaches whichever thread is main, so a thread that stops being main mid-run never
  // gets its own and would checkpoint for the life of the page. Ask the runtime instead.
  const isRunActive = useCallback(
    (threadId: string): boolean => {
      const runtime = aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) {
        return false;
      }
      try {
        return runtime.threads.getById(threadId).getState().isRunning === true;
      } catch {
        // A deleted or detached thread throws out of getById rather than reporting idle.
        return false;
      }
    },
    [aui],
  );

  // Keep one scheduler for the component lifetime so its timers remain stoppable; the refs give it
  // the latest queueSave and liveness check.
  const queueSaveRef = useRef(queueSave);
  useEffect(() => {
    queueSaveRef.current = queueSave;
  }, [queueSave]);
  const isRunActiveRef = useRef(isRunActive);
  useEffect(() => {
    isRunActiveRef.current = isRunActive;
  }, [isRunActive]);
  const checkpointsRef = useRef<RunCheckpointScheduler | null>(null);
  const checkpoints = useCallback((): RunCheckpointScheduler => {
    checkpointsRef.current ??= createRunCheckpointScheduler(
      (threadId) => queueSaveRef.current(threadId),
      {
        isActive: (threadId) => isRunActiveRef.current(threadId),
        isBounded: (threadId) => threadHasDurableGenerationRun(threadId),
      },
    );
    return checkpointsRef.current;
  }, []);

  useEffect(() => {
    // A hidden renderer may never get its next interval: Chromium throttles chained timers to a wake
    // a minute and a WebView can be parked outright, so checkpoint on the way out. No
    // beforeunload, matching flushSettingsOnPageHidden.
    const flush = () => {
      checkpointsRef.current?.flushAll();
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        flush();
      }
    };
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      checkpointsRef.current?.stopAll();
    };
  }, []);

  useAuiEvent("thread.runEnd", ({ threadId }) => {
    checkpoints().stop(threadId);
    queueSave(threadId);
  });

  useAuiEvent("thread.runStart", ({ threadId }) => {
    checkpoints().start(threadId);
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

// True when the chat tab is visible. While false, ChatPage stays mounted (runtime and autosave
// alive, so the stream survives) but its views unmount, so no portaled surface bleeds through.
export const ChatActiveContext = createContext(true);

export function useChatActive(): boolean {
  return useContext(ChatActiveContext);
}

// True inside a Compare pane. Both panes mount the same message controls, so a window-level
// chord would otherwise be answered by whichever pane mounted first.
const ComparePaneContext = createContext(false);

export function useInComparePane(): boolean {
  return useContext(ComparePaneContext);
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
  backgrounded = false,
  onInitialHistoryReady,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  projectId?: string | null;
  initialThreadId?: string;
  newThreadNonce?: string;
  syncActiveThreadId?: boolean;
  listThreads?: boolean;
  // Mounted only to keep an in-flight run attached while another view is on screen: the runtime
  // stays alive, and everything driving the shared single-chat state stands down.
  backgrounded?: boolean;
  onInitialHistoryReady?: () => void;
}): ReactElement {
  // Read by the history adapter's own active-thread publication, the sibling of
  // ThreadBackendAutosave's, which needs the same stand-down. Kept in a ref so the memo below
  // never sees it change, since rebuilding the runtime hook would rebuild the runtime.
  const backgroundedRef = useRef(backgrounded);
  backgroundedRef.current = backgrounded;
  // Declared before the memo below because the memo reads it. Same ref the switch components
  // mutate, so the adapter can tell "this pane is on screen" from "this pane is being left".
  const newThreadSwitchStateRef = useRef<NewThreadSwitchState>({
    activeNonce: null,
    hasSwitched: false,
    attempt: 0,
    pendingSavedThreadIds: [],
    nonceThread: null,
    landedAttempt: 0,
  });
  const runtimeHook = useMemo(
    () =>
      createRuntimeHook(
        modelType,
        pairId,
        initialThreadId,
        onInitialHistoryReady,
        backgroundedRef,
        newThreadSwitchStateRef,
      ),
    [initialThreadId, modelType, onInitialHistoryReady, pairId],
  );
  const runtime = useRemoteThreadListRuntime({
    runtimeHook,
    adapter: createStudioDbAdapter(modelType, pairId, projectId, listThreads),
  });
  const signalFailedInitialSwitchReady = useCallback(() => {
    if (onInitialHistoryReady) {
      onInitialHistoryReady();
    } else if (modelType === "base" && !pairId) {
      window.dispatchEvent(new Event("unsloth:app-shell-ready"));
    }
  }, [modelType, onInitialHistoryReady, pairId]);

  const aui = useAui({});
  useEffect(() => {
    if (!initialThreadId && !newThreadNonce) {
      newThreadSwitchStateRef.current.hasSwitched = true;
    }
  }, [initialThreadId, newThreadNonce]);

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {/* Pane identity for the tool-output store maps: the adapter prefixes its keys with this scope so
          concurrent panes with colliding tool ids cannot bleed output into each other's cards. */}
      <ChatProjectScopeContext.Provider value={projectId ?? null}>
      <ToolPaneScopeContext.Provider value={toolPaneScope(modelType, pairId)}>
        <ComparePaneContext.Provider value={Boolean(pairId)}>
        <ActiveThreadSync
          enabled={
            modelType === "base" &&
            !pairId &&
            !newThreadNonce &&
            !initialThreadId &&
            !backgrounded
          }
        />
        {/* Compare clears activeThreadId on the way in and this view is hidden rather than unmounted, so
            nothing puts it back: the nonce is unchanged so ThreadNewChatSwitch returns, and
            ActiveThreadSync is off while a nonce is present. ThreadScopedSettingsSync is NOT
            nonce-gated, so the chat came back detached, with no title or context usage. */}
        <NonceThreadResumeRestore
          enabled={
            modelType === "base" &&
            !pairId &&
            !!newThreadNonce &&
            !initialThreadId &&
            !backgrounded
          }
        />
        <ThreadScopedSettingsSync
          enabled={modelType === "base" && !pairId && !backgrounded}
        />
        <ActiveBranchRegistrar
          enabled={modelType === "base" && !pairId && !backgrounded}
        />
        <ThreadContextUsageRecount
          enabled={modelType === "base" && !pairId && !backgrounded}
        />
        <ThreadBackendAutosave
          modelType={modelType}
          pairId={pairId}
          backgrounded={backgrounded}
          newThreadSwitchStateRef={newThreadSwitchStateRef}
        />
        <CancelRegistrar />
        {initialThreadId && (
          <ThreadAutoSwitch
            threadId={initialThreadId}
            syncActiveThreadId={syncActiveThreadId && !backgrounded}
            paused={backgrounded}
            newThreadSwitchStateRef={newThreadSwitchStateRef}
            onSwitchFailed={signalFailedInitialSwitchReady}
          />
        )}
        {!initialThreadId && newThreadNonce && (
          <ThreadNewChatSwitch
            nonce={newThreadNonce}
            paused={backgrounded}
            newThreadSwitchStateRef={newThreadSwitchStateRef}
          />
        )}
        {/* The view stays mounted (only CSS-hidden) while off-route so the run stays attached and the
            stream alive; unmounting aborts generation. */}
        {children}
        </ComparePaneContext.Provider>
      </ToolPaneScopeContext.Provider>
      </ChatProjectScopeContext.Provider>
    </AssistantRuntimeProvider>
  );
}
