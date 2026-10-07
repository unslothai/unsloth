// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { authFetch, getAuthSessionEpoch } from "@/features/auth";
import { isMcpToolOnly, mcpImageMappingsEnabled } from "./api/mcp-image";
import { listMcpServers } from "./api/mcp-servers-api";
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
  useLayoutEffect,
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
import {
  CHAT_HISTORY_UPDATED_EVENT,
  streamChatCompletions,
  uploadChatAttachmentOriginal,
} from "./api/chat-api";
import { selectCodeToolNames } from "./api/code-tool-placement";
import { getResearchThreadState } from "./api/research-api";
import {
  cancelChatGenerationRun,
  type ChatGenerationRun,
  getActiveChatGenerationRuns,
  ChatGenerationStalledError,
  followChatGenerationRun,
  isTerminalChatGenerationRun,
  toolApprovalIsPending,
} from "./api/chat-generation-api";
import {
  TEXT_ATTACHMENT_ACCEPT,
  decodeHtmlAttachmentBytes,
  extractDocxAttachmentText,
  extractHtmlAttachmentText,
  extractOfficeAttachmentText,
  extractPdfAttachmentText,
  getDocumentAttachmentSizeError,
  getDocxAttachmentError,
  getPdfAttachmentTextError,
} from "./attachment-content";
import {
  type ChatAttachmentOriginal,
  persistAttachmentOriginals,
  reuseStagedUpload,
  withAttachmentOriginal,
} from "./attachment-originals";
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
  externalModelSupportsStudioTools,
  providerModelSupportsStudioTools,
  providerModelSupportsVision,
} from "./external-providers";
import {
  CHAT_IMAGE_ACCEPT,
  convertedImageType,
  normalizeChatImage,
} from "./image-normalize";
import { chatModelLoaded } from "./lib/chat-model-loaded";
import {
  type OpenDocumentAttachmentContent,
  readActiveOpenDocumentAttachmentContent,
  readOpenDocumentAttachmentContent,
} from "./open-document";
import {
  OPEN_DOCUMENT_ATTACHMENT_ACCEPT,
  RTF_ATTACHMENT_ACCEPT,
  TOOL_ONLY_ATTACHMENT_EXTENSIONS,
} from "./open-document-accept";
import {
  providerHostsCodeExecution,
  providerSupportsBuiltinCodeExecution,
} from "./provider-capabilities";
import { readRtfAttachmentContent } from "./rtf";
import {
  awaitThreadScopedSettingsWrite,
  beginThreadScopedPairing,
  codeToolsOn,
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
  generationIsSettled,
  generationReplayMetadata,
  createRecoveryPublishSchedule,
  registerRecoveredRunStop,
  threadHasDurableGenerationRun,
  generationNeedsRecovery,
  requestParsesThinkTags,
  restoreCarriedPartsFromRaw,
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
import {
  beginSavedHistoryReconciliation,
  isSavedHistoryReconciliationSuperseded,
  reconcileOrdinarySavedMessagesInView,
} from "./utils/saved-history-reconciliation";
import { createGenerationToolRecovery } from "./utils/generation-tool-recovery";
import { providerCompactionConnectionKey } from "./utils/provider-compaction";
import { mergeContextTruncation } from "./utils/context-truncation";
import { registerLiveThreadView } from "./utils/live-thread-head";
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
import { estimateContextUsage } from "./utils/estimate-chat-tokens";
import {
  awaitStoredChatThreadWrites,
  deleteStoredChatThreads,
  ensureStoredChatThread,
  getStoredChatMessage,
  getStoredChatThread,
  getStoredChatThreadReadResult,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  readStoredChatMessages,
  listStoredChatThreads,
  markThreadIncognito,
  registerNewThreadIdSource,
  saveStoredChatMessage,
  saveStoredChatThread,
  syncStoredChatMessages,
  trackStoredChatThreadRecord,
  unmarkThreadIncognito,
  updateStoredChatThread,
} from "./utils/chat-history-storage";
import {
  isChatThreadDeleted,
  markChatThreadDeleted,
} from "./utils/chat-thread-tombstones";
import {
  answeringCheckpoint,
  buildTitleRequest,
  fallbackTitleFromUserText,
  titleCheckpoint,
  titleFromStream,
} from "./utils/chat-title";
import { syncExportedRepositoryToBackend } from "./utils/delete-thread-message";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import {
  attachmentContentText,
  attachmentsSample,
  isPastedTextFile,
} from "./utils/pasted-text";
import {
  annotationsContentText,
  annotationsOfFile,
} from "./utils/document-annotations";
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
  RUN_CHECKPOINT_INTERVAL_MS,
  type RunCheckpointScheduler,
  createRunCheckpointScheduler,
} from "./utils/run-checkpoint-scheduler";
import { isAssistantLocalThreadId } from "./utils/thread-ids";
import { sanitizeThreadScopedSettings } from "./utils/thread-scoped-settings";
import { VideoAttachmentAdapter } from "./video-attachment-adapter";

const pendingHistoryAppendByMessageId = new Map<string, Promise<void>>();
const pendingRunStartReadyByMessageId = new Map<
  string,
  Promise<string | undefined>
>();
const pendingRunStartThreadIdsByMessageId = new Map<string, string[]>();

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
    // Name and MIME say "video" for audio-only 3GP or .ts, so inspect bytes first.
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
    const { incognito } = useChatRuntimeStore.getState();
    const epoch = getAuthSessionEpoch();
    try {
      return await withAttachmentOriginal(
        attachment,
        await this.delegate.send(attachment),
        incognito,
        epoch,
        pythonToolRunsInStudio(),
      );
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

const MCP_TOOL_IMAGE_MIMES = ["image/png", "image/jpeg", "image/webp"];

const MCP_LOOKUP_FAILED =
  "Could not read your MCP servers, so the image was not attached. Try again.";

async function mcpToolOnlyEnabled(): Promise<boolean | null> {
  const state = useChatRuntimeStore.getState();
  const checkpoint = state.params.checkpoint;
  const toolsSupported = parseExternalModelId(checkpoint)
    ? externalModelSupportsStudioTools(checkpoint)
    : state.supportsTools ||
      !chatModelLoaded({
        checkpoint,
        modelLoading: state.modelLoading,
        isExternalModel: false,
        residentCheckpoint: state.residentCheckpoint,
      });
  if (!toolsSupported || !state.mcpEnabledForChat) return false;
  try {
    return mcpImageMappingsEnabled(await listMcpServers());
  } catch {
    return null;
  }
}

class VisionImageAdapter implements AttachmentAdapter {
  accept = CHAT_IMAGE_ACCEPT;
  private readonly converted = new Map<string, Promise<File | null>>();
  private readonly toolOnlyIds = new Set<string>();

  async *add({
    file: picked,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
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
    const unavailableReason = !modelLoaded
      ? null
      : getImageInputUnavailableReason({
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
    const mcpToolOnlyState = await mcpToolOnlyEnabled();
    // Fail closed: a configured mapping may be what this read missed.
    if (mcpToolOnlyState === null) {
      toast.error(MCP_LOOKUP_FAILED);
      throw new Error(MCP_LOOKUP_FAILED);
    }
    const mcpToolOnly = mcpToolOnlyState;
    if (unavailableReason && !mcpToolOnly) {
      toast.error(unavailableReason);
      throw new Error(unavailableReason);
    }
    if (
      mcpToolOnly &&
      (picked.size > 10 * 1024 * 1024 ||
        (!MCP_TOOL_IMAGE_MIMES.includes(picked.type) &&
          convertedImageType(picked) === null))
    ) {
      const reason =
        "Images for MCP tools must be PNG, JPEG or WebP and at most 10 MB.";
      toast.error(reason);
      throw new Error(reason);
    }

    if (mcpToolOnly && this.toolOnlyIds.size > 0) {
      const reason = "Only one image per message can go to MCP tools.";
      toast.error(reason);
      throw new Error(reason);
    }

    const maxSize = 20 * 1024 * 1024;
    if (picked.size > maxSize) {
      throw new Error("Image size exceeds 20MB limit");
    }
    const attachment = {
      id: crypto.randomUUID(),
      type: "image",
      name: picked.name,
      contentType: picked.type,
      file: picked,
      ...(mcpToolOnly ? { mcpToolOnly: true } : {}),
      status: { type: "requires-action", reason: "composer-send" },
    } satisfies PendingAttachment & { mcpToolOnly?: boolean };
    if (mcpToolOnly) this.toolOnlyIds.add(attachment.id);
    if (convertedImageType(picked) === null) {
      yield attachment;
      return;
    }
    yield {
      ...attachment,
      status: { type: "running", reason: "uploading", progress: 0 },
    };
    const conversion = normalizeChatImage(picked);
    this.converted.set(
      attachment.id,
      conversion.catch(() => null),
    );
    let file: File;
    try {
      file = await conversion;
    } catch (error) {
      if (!this.converted.has(attachment.id)) {
        return;
      }
      toast.error(error instanceof Error ? error.message : String(error));
      this.toolOnlyIds.delete(attachment.id);
      throw error;
    }
    if (mcpToolOnly && file.size > 10 * 1024 * 1024) {
      const reason =
        "The converted image is over the 10 MB limit for MCP tools.";
      this.toolOnlyIds.delete(attachment.id);
      toast.error(reason);
      throw new Error(reason);
    }
    // Removed while converting: yielding again would put it back.
    if (!this.converted.has(attachment.id)) {
      return;
    }
    yield { ...attachment, name: file.name, contentType: file.type, file };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const conversion = this.converted.get(attachment.id);
    this.converted.delete(attachment.id);
    this.toolOnlyIds.delete(attachment.id);
    const file = conversion ? await conversion : attachment.file;
    const current = await mcpToolOnlyEnabled();
    if (current !== isMcpToolOnly(attachment)) {
      const reason =
        current === null
          ? MCP_LOOKUP_FAILED
          : "MCP image settings changed since this image was attached. Remove it and attach it again.";
      toast.error(reason);
      throw new Error(reason);
    }
    // Also flagged on the part so modelVisibleMessage drops it.
    const toolOnly = isMcpToolOnly(attachment) ? { mcpToolOnly: true } : {};
    return {
      id: attachment.id,
      type: "image",
      ...toolOnly,
      name: file?.name ?? attachment.name,
      contentType: file?.type ?? attachment.contentType,
      content: file
        ? [
            {
              type: "image",
              image: await this.fileToBase64DataURL(file),
              ...toolOnly,
            },
          ]
        : [],
      status: { type: "complete" },
    };
  }

  async remove(attachment: { id: string }): Promise<void> {
    this.converted.delete(attachment.id);
    this.toolOnlyIds.delete(attachment.id);
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
  private readonly texts = new Map<string, Promise<string | null>>();

  // Refused here, not at send: the composer clears before send(), losing the message.
  async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    const sizeError = getDocumentAttachmentSizeError(file, "PDF");
    if (sizeError) {
      toast.error(sizeError);
      throw new Error(sizeError);
    }
    const attachment = {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "running", reason: "uploading", progress: 0 },
    } satisfies PendingAttachment;
    yield attachment;
    const text = extractPdfAttachmentText(file).catch(() => null);
    this.texts.set(attachment.id, text);
    const error = pdfAttachmentError(file.name, await text);
    // Removed or sent while reading: yielding again would put the chip back.
    if (this.texts.get(attachment.id) !== text) return;
    if (error) {
      toast.error(error);
      yield { ...attachment, status: { type: "incomplete", reason: "error" } };
      return;
    }
    yield {
      ...attachment,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const pending = this.texts.get(attachment.id);
    this.texts.delete(attachment.id);
    const text = await (pending ??
      extractPdfAttachmentText(attachment.file).catch(() => null));
    // Rechecked: settings can change after the attach check passed.
    const textError = pdfAttachmentError(attachment.name, text);
    if (textError && attachment.status.type !== "incomplete") {
      toast.error(textError);
    }
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [
        {
          type: "text",
          text: `[PDF: ${attachment.name}]\n${textError ?? text}`,
        },
      ],
      status: { type: "complete" },
    };
  }

  remove(attachment: Attachment): Promise<void> {
    this.texts.delete(attachment.id);
    return Promise.resolve();
  }
}

class TextAttachmentAdapter implements AttachmentAdapter {
  // MIME is unreliable for source files, so also match by extension.
  accept = TEXT_ATTACHMENT_ACCEPT;

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    // Check size before reading to avoid holding huge bytes and decoded text.
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
    const annotations = annotationsOfFile(attachment.file);
    const text = annotations ? "" : await readTextAttachmentOnce(attachment.file);
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [
        {
          type: "text",
          text: annotations
            ? annotationsContentText(annotations)
            : attachmentContentText(
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
    const bytes = new Uint8Array(await attachment.file.arrayBuffer());
    const text = extractHtmlAttachmentText(decodeHtmlAttachmentBytes(bytes));
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

  // Check archive parts too: a small .docx can inflate past the cap.
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

const OFFICE_LABELS: Record<string, "XLSX" | "PPTX"> = {
  xlsx: "XLSX",
  xlsm: "XLSX",
  pptx: "PPTX",
};

class OfficeAttachmentAdapter implements AttachmentAdapter {
  accept = [
    ".xlsx,.xlsm,.pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ].join(",");

  private label(name: string, type: string): "XLSX" | "PPTX" {
    const extension = name.split(".").pop()?.toLowerCase() ?? "";
    const byName = Object.hasOwn(OFFICE_LABELS, extension) ? OFFICE_LABELS[extension] : undefined;
    return byName ?? (type.includes("presentationml") ? "PPTX" : "XLSX");
  }

  // Read at add: the composer drops the typed message before send().
  private readonly texts = new Map<string, string>();

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const label = this.label(file.name, file.type);
    let text: string;
    try {
      text = await extractOfficeAttachmentText(file, label);
    } catch (cause) {
      const message = (cause as Error | undefined)?.message;
      const tooLarge = `${label} file is too large: ${file.name}`;
      const error =
        message === tooLarge || message === "File is too large to preview."
          ? tooLarge
          : `${label} file could not be read: ${file.name}`;
      toast.error(error);
      throw new Error(error);
    }
    const id = crypto.randomUUID();
    this.texts.set(id, text);
    return {
      id,
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const label = this.label(attachment.name, attachment.contentType ?? "");
    const text = this.texts.get(attachment.id) ?? (await extractOfficeAttachmentText(attachment.file, label));
    this.texts.delete(attachment.id);
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[${label}: ${attachment.name}]\n${text}` }],
      status: { type: "complete" },
    };
  }

  remove(attachment: Attachment): Promise<void> {
    this.texts.delete(attachment.id);
    return Promise.resolve();
  }
}

class RtfAttachmentAdapter implements AttachmentAdapter {
  accept = RTF_ATTACHMENT_ACCEPT;
  private readonly texts = new Map<string, string>();

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    let text: string;
    try {
      ({ text } = await readRtfAttachmentContent(file, file.name));
    } catch (cause) {
      const error = `RTF file could not be read: ${file.name}: ${(cause as Error).message}`;
      toast.error(error);
      throw new Error(error);
    }
    const id = crypto.randomUUID();
    this.texts.set(id, text);
    return {
      id,
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const text =
      this.texts.get(attachment.id) ??
      (await readRtfAttachmentContent(attachment.file, attachment.name)).text;
    this.texts.delete(attachment.id);
    return {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text: `[RTF: ${attachment.name}]\n${text}` }],
      status: { type: "complete" },
    };
  }

  remove(attachment: Attachment): Promise<void> {
    this.texts.delete(attachment.id);
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

const MAX_TOOL_ONLY_ATTACHMENT_BYTES = 200 * 1024 * 1024;

/** Must match how chat-adapter.ts decides it. */
function pythonToolRunsInStudio(): boolean {
  const state = useChatRuntimeStore.getState();
  const codeToolsEnabled = codeToolsOn(state);
  const external = parseExternalModelId(state.params.checkpoint);
  if (!external) return state.supportsTools && codeToolsEnabled;
  const provider = (
    loadConnectionsEnabled() ? loadExternalProviders() : []
  ).find((p) => p.id === external.providerId);
  if (
    !provider ||
    providerModelSupportsStudioTools(
      provider.providerType,
      external.modelId,
    ) !== true
  ) {
    return false;
  }
  return selectCodeToolNames({
    codeToolsEnabled,
    hostedCodeExecutionForThisTurn: providerSupportsBuiltinCodeExecution(
      provider.providerType,
      external.modelId,
      provider.baseUrl,
      provider.apiType,
    ),
    providerHostsCodeExecution: providerHostsCodeExecution(
      provider.providerType,
      provider.baseUrl,
      provider.apiType,
    ),
  }).local.includes("python");
}

function pythonToolOpensAttachments(): boolean {
  return pythonToolRunsInStudio() && !useChatRuntimeStore.getState().incognito;
}

function pdfAttachmentError(name: string, text: string | null): string | null {
  return text === null
    ? `PDF file could not be read: ${name}`
    : getPdfAttachmentTextError(name, text, pythonToolOpensAttachments());
}

class ToolOnlyAttachmentAdapter implements AttachmentAdapter {
  accept = TOOL_ONLY_ATTACHMENT_EXTENSIONS;
  private readonly uploads = new Map<
    string,
    Promise<ChatAttachmentOriginal | null>
  >();
  private readonly uploadedAt = new Map<string, number>();

  private upload(file: File): Promise<ChatAttachmentOriginal | null> {
    return uploadChatAttachmentOriginal(file).catch(() => null);
  }

  async *add({
    file,
  }: {
    file: File;
  }): AsyncGenerator<PendingAttachment, void> {
    const refusal = !pythonToolRunsInStudio()
      ? `Turn on Code with a model that runs the python tool to attach ${file.name}.`
      : useChatRuntimeStore.getState().incognito
        ? `Temporary chats save no files, so the python tool cannot open ${file.name}.`
        : file.size > MAX_TOOL_ONLY_ATTACHMENT_BYTES
          ? `File is too large: ${file.name}`
          : null;
    if (refusal) {
      toast.error(refusal);
      throw new Error(refusal);
    }
    const attachment = {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "running", reason: "uploading", progress: 0 },
    } satisfies PendingAttachment;
    yield attachment;
    const upload = this.upload(file);
    this.uploads.set(attachment.id, upload);
    this.uploadedAt.set(attachment.id, Date.now());
    const original = await upload;
    if (this.uploads.get(attachment.id) !== upload) return;
    if (!original) {
      this.uploads.delete(attachment.id);
      toast.error(`Could not upload ${file.name}`);
      yield { ...attachment, status: { type: "incomplete", reason: "error" } };
      return;
    }
    yield {
      ...attachment,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const original =
      (reuseStagedUpload(this.uploadedAt.get(attachment.id))
        ? await this.uploads.get(attachment.id)
        : null) ?? (await this.upload(attachment.file));
    this.uploads.delete(attachment.id);
    this.uploadedAt.delete(attachment.id);
    const text = original
      ? `[${attachment.name}: only the python tool can read this file]`
      : `[${attachment.name} could not be uploaded, so it cannot be read]`;
    const complete: CompleteAttachment = {
      id: attachment.id,
      type: "document",
      name: attachment.name,
      contentType: attachment.contentType,
      content: [{ type: "text", text }],
      status: { type: "complete" },
    };
    return original ? ({ ...complete, original } as CompleteAttachment) : complete;
  }

  async remove(attachment: { id: string }): Promise<void> {
    this.uploads.delete(attachment.id);
    this.uploadedAt.delete(attachment.id);
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

// A paste-only turn has its text in an attachment, so include it for the title.
function titleTextOf(m: ThreadMessage | undefined): string {
  const text = extractTextParts(m);
  if (m?.role !== "user") return text;
  const sample = attachmentsSample(m.attachments);
  if (sample.length === 0) return text;
  return text.length > 0 ? `${text}\n\n${sample}` : sample;
}

async function generateTitleWithModel(payload: {
  checkpoint: string;
  userText: string;
  assistantText?: string;
}): Promise<string | null> {
  if (!payload.checkpoint) return null;

  const user = clip(payload.userText, 256);
  const assistant = clip(payload.assistantText ?? "", 384);
  const parts: string[] = [`User: ${user}`];
  if (assistant) {
    parts.push(`Assistant: ${assistant}`);
  }

  try {
    // Inside the try: building this encrypts a browser key over the network.
    const request = await buildTitleRequest(
      payload.checkpoint,
      parts.join("\n"),
    );
    if (!request) return null;
    return await titleFromStream(
      streamChatCompletions(request, new AbortController().signal),
    );
  } catch {
    return null;
  }
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
  return JSON.parse(JSON.stringify(attachments.map((attachment) => ({ ...attachment, file: undefined }))));
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
      metadata: {
        custom: m.metadata?.createdAtEstimated === true ? { createdAtEstimated: true } : {},
      },
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
  // Never restore running from metadata alone; unverified runs show as interrupted.
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
  // This tab streams the run itself; a second, lagging writer would corrupt it.
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
    const stored = generationRawContent(storedMessage.content);
    const carried = stored.carried;
    // Keyed exactly as the live stream (toolConfirmationScopeId) so cards are the same.
    const toolRecovery = createGenerationToolRecovery(carried, runId, cursor, {
      register: (partId, approvalId, sessionId) =>
        useChatRuntimeStore
          .getState()
          .setToolConfirmation(
            partId,
            approvalId,
            sessionId,
            `${sessionId || "_default"}:${threadId}`,
          ),
      resolve: (partId) =>
        useChatRuntimeStore.getState().clearToolConfirmation(partId),
    });
    // Settled before disarmAll so an arm cannot land after the run is over.
    let seededApprovals: Promise<void> | null = null;
    let { raw, reasoningOpen } = stored;
    let parseThink = metadata.parseThinkTags !== false;
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
    let quoteCut =
      (metadata.incomplete as { reason?: unknown } | undefined)?.reason ===
      "quote_cut";
    let currentMetadata = { ...metadata };
    const serverCancel = () => {
      void cancelChatGenerationRun(runId).catch(() => {});
    };
    const runtime = useChatRuntimeStore.getState();
    runtime.registerThreadServerCancel(threadId, serverCancel);
    const unregisterStop = registerRecoveredRunStop(threadId, serverCancel);
    runtime.setThreadRunning(threadId, true, {
      local: true,
      owner: serverCancel,
    });

    // Save and finalisation share one rebuild, or names lag or are empty.
    const rebuild = () =>
      toolRecovery.withSources(
        restoreCarriedPartsFromRaw(
          reasoningOpen ? `${raw}</think>` : raw,
          carried,
          { parseThink },
        ),
      ) as MessageRecord["content"];
    const toolNames = (content: MessageRecord["content"]): string[] =>
      (Array.isArray(content) ? content : []).flatMap((part) => {
        const card = part as { type?: string; toolName?: unknown };
        return card.type === "tool-call" && typeof card.toolName === "string"
          ? [card.toolName]
          : [];
      });

    /** `running` is passed in: a stalled follow settles while the run is non-terminal. */
    const commit = async (
      nextMetadata: Record<string, unknown>,
      running: boolean,
      save = true,
    ) => {
      currentMetadata = nextMetadata;
      const content = rebuild();
      if (save) {
        await saveStoredChatMessage({
          id: storedMessage.id,
          threadId,
          parentId: storedMessage.parentId ?? null,
          role: "assistant",
          content,
          metadata: nextMetadata,
          createdAt: storedMessage.createdAt,
        }).catch(() => {
          // A newer server status can reject this save; settlement retries with full content.
        });
      }

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

    const schedule = createRecoveryPublishSchedule(RUN_CHECKPOINT_INTERVAL_MS);
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
        quoteCut,
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
          toolCalls: toolNames(rebuild()),
        });
      }
      const settled = nextMetadata.generationSettled === true;
      await commit(
        nextMetadata,
        generationNeedsRecovery(nextMetadata),
        schedule.takeSave(settled),
      );
    };
    const caughtUp = (run: ChatGenerationRun) =>
      schedule.shouldPublish(
        cursor,
        generationIsSettled(run.status, cursor, run.lastEventSeq),
      );

    try {
      let lastPublishedStatus = "";
      let identityValidated = false;
      // The follower signals its no-progress deadline by throwing.
      let followStalled = false;
      try {
        for await (const update of followChatGenerationRun(runId, {
          replayFrom: toolRecovery.replayFrom,
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
                // The placeholder is empty until the first save, so seed replay from the request.
                raw = lastRequestMessage.content;
              }
            }
            // Re-arm parked approvals only while the run can still be answered.
            if (!isTerminalChatGenerationRun(update.run)) {
              // Not awaited: awaiting holds /events closed, and /events marks the run attended.
              seededApprovals = toolRecovery.armSeededApprovals(
                update.run.requestPayload?.session_id,
                (approvalId) =>
                  toolApprovalIsPending(
                    approvalId,
                    typeof update.run.requestPayload?.session_id === "string"
                      ? update.run.requestPayload.session_id
                      : "",
                  ),
              );
              seededApprovals.catch(() => {});
            }
            if (typeof metadata.parseThinkTags !== "boolean") {
              parseThink = requestParsesThinkTags(update.run.requestPayload);
              currentMetadata = {
                ...currentMetadata,
                parseThinkTags: parseThink,
              };
            }
            schedule.attach(update.run.lastEventSeq);
            identityValidated = true;
          }
          let advanced = false;
          let recoveredProviderCompaction: ReturnType<
            typeof toolRecovery.apply
          >;
          if (update.event?.type === "chunk") {
            recoveredProviderCompaction = toolRecovery.apply(
              update.event.payload,
              raw.length,
              update.event.seq,
              update.run.requestPayload.session_id,
            );
          }
          if (update.event && update.event.seq > cursor) {
            advanced = true;
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
                quote_cut?: boolean;
              };
              if ("_reasoningDurationMs" in chunk) {
                currentMetadata = recoveredReasoningSummaryMetadata(
                  currentMetadata,
                  chunk._reasoningDurationMs,
                );
                if (caughtUp(update.run)) {
                  lastPublishedStatus = update.run.status;
                  await publish(update.run);
                }
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
              if (recoveredProviderCompaction) {
                const sourceProviderType = update.run.requestPayload.provider_type;
                const sourceModelId =
                  update.run.requestPayload.external_model ??
                  update.run.requestPayload.model;
                currentMetadata = {
                  ...currentMetadata,
                  ...recoveredProviderCompaction,
                  providerCompactionProviderType:
                    typeof sourceProviderType === "string"
                      ? sourceProviderType
                      : undefined,
                  providerCompactionModelId:
                    typeof sourceModelId === "string"
                      ? sourceModelId
                      : undefined,
                  providerCompactionConnectionKey:
                    providerCompactionConnectionKey(
                      update.run.requestPayload.provider_id,
                      update.run.requestPayload.provider_base_url,
                      update.run.requestPayload.provider_api_type,
                    ),
                };
              }
              if (chunk.quote_cut === true) quoteCut = true;
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
              const { text: delta, hasStructuredReasoning } = extractDeltaText(
                deltaRecord?.content,
              );
              if (!parseThink && (reasoning || hasStructuredReasoning)) {
                parseThink = true;
                currentMetadata = { ...currentMetadata, parseThinkTags: true };
              }
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
            caughtUp(update.run) &&
            ((update.event?.type === "chunk" && advanced) ||
              update.run.status !== lastPublishedStatus ||
              generationIsSettled(
                update.run.status,
                cursor,
                update.run.lastEventSeq,
              ));
          if (shouldPublish) {
            lastPublishedStatus = update.run.status;
            await publish(update.run);
          }
          if (isTerminalChatGenerationRun(update.run)) {
            // Disarm in the finally covers every exit; a stale mapping would cap later streams.
            forgetServerActiveGenerationRun(runId);
          }
        }
      } catch (error) {
        if (!(error instanceof ChatGenerationStalledError)) throw error;
        followStalled = true;
      }
      if (followStalled || generationNeedsRecovery(currentMetadata)) {
        // Fence the run server-side, or create_run 409s Continue and the next message.
        serverCancel();
        // The follow hit its no-progress deadline; settle the reply, keeping replayed content.
        await commit(
          {
            ...currentMetadata,
            ...generationReplayMetadata({
              cursor,
              firstChunkAt,
              totalChunks,
              usage: recoveryUsage,
              timings: recoveryTimings,
            }),
            incomplete: { reason: "interrupted" as const },
            // Stops another follower starting; history.load clears it if the server still has the run.
            generationLocallyInterrupted: true,
          },
          false,
        );
      }
    } finally {
      // Every exit, not just terminal: a leftover seeded card breaks later chords. Join the arm first.
      unregisterStop();
      if (seededApprovals) await seededApprovals.catch(() => {});
      toolRecovery.disarmAll();
      const store = useChatRuntimeStore.getState();
      store.setThreadRunning(threadId, false, { owner: serverCancel });
      store.clearThreadServerCancel(threadId, serverCancel);
    }
  })()
    .catch(() => {})
    .finally(() => generationRecoveries.delete(runId));
  generationRecoveries.set(runId, { promise: recovery, views });
}

const temporaryThreadCreation = new Map<
  string,
  { modelId: string; modelGgufVariant: string | null | undefined; createdAt: number }
>();

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
  incognito?: boolean;
  neverSent?: boolean;
  modelId?: string;
  modelGgufVariant?: string | null;
  createdAt?: number;
}): Promise<void> {
  if (isChatThreadDeleted(threadId)) {
    return;
  }
  // Snapshot synchronously before the await, so later toggles cannot change a retry's identity.
  const runtimeStateAtInit = useChatRuntimeStore.getState();
  const incognitoAtInit = incognito ?? runtimeStateAtInit.incognito;
  const modelIdAtInit = modelId ?? runtimeStateAtInit.params.checkpoint ?? "";
  const modelGgufVariantAtInit =
    modelGgufVariant !== undefined
      ? modelGgufVariant
      : runtimeStateAtInit.activeGgufVariant;
  const createdAtInit = createdAt ?? Date.now();
  // Gate on the caller knowing the thread is new: `__LOCALID_` ids are permanent for all chats.
  const creation = {
    modelId: modelIdAtInit,
    modelGgufVariant: modelGgufVariantAtInit,
    createdAt: createdAtInit,
  };
  if (incognitoAtInit && neverSent) {
    markThreadIncognito(threadId);
    temporaryThreadCreation.set(threadId, creation);
    return;
  }
  // A point lookup, not a listing: must not scale with chat count.
  const existing = await getStoredChatThread(threadId);
  if (existing) {
    return;
  }
  // After the row check, so an already-persisted thread is never tagged temporary.
  if (incognitoAtInit) {
    markThreadIncognito(threadId);
    temporaryThreadCreation.set(threadId, creation);
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
    // assistant-ui can overlap first-message persistence, so a concurrent create is success.
    const existingAfterRace = await getStoredChatThread(threadId).catch(
      () => undefined,
    );
    if (existingAfterRace) {
      return;
    }
    throw error;
  }
}

function parentsFirst(
  items: readonly ExportedMessageRepositoryItem[],
): ExportedMessageRepositoryItem[] {
  const byId = new Map(items.map((item) => [item.message.id, item]));
  const seen = new Set<string>();
  const ordered: ExportedMessageRepositoryItem[] = [];
  const visit = (item: ExportedMessageRepositoryItem) => {
    if (seen.has(item.message.id)) return;
    seen.add(item.message.id);
    const parent = item.parentId ? byId.get(item.parentId) : undefined;
    if (parent) visit(parent);
    ordered.push(item);
  };
  items.forEach(visit);
  return ordered;
}

/** Save a temporary chat to history in one transaction: a failure leaves at most an empty row,
 *  which a retry reuses. */
export async function persistTemporaryThread({
  threadId,
  modelType,
  messages,
}: {
  threadId: string;
  modelType: ModelType;
  messages: readonly ExportedMessageRepositoryItem[];
}): Promise<void> {
  unmarkThreadIncognito(threadId);
  try {
    const times = messages
      .map(({ message }) => message.createdAt?.getTime?.())
      .filter((time): time is number => typeof time === "number");
    const creation = temporaryThreadCreation.get(threadId);
    await ensureThreadRecord({
      threadId,
      modelType,
      projectId: null,
      incognito: false,
      ...(creation && {
        modelId: creation.modelId,
        modelGgufVariant: creation.modelGgufVariant,
      }),
      createdAt:
        creation?.createdAt ?? (times.length > 0 ? Math.min(...times) : Date.now()),
    });
    const epoch = getAuthSessionEpoch();
    const records: MessageRecord[] = await Promise.all(parentsFirst(messages).map(async ({ parentId, message }) => {
      // Upload in-memory documents now, before the File is lost to JSON.
      const attachments =
        message.role === "user"
          ? cloneAttachments(await persistAttachmentOriginals(message.attachments, epoch))
          : [];
      const metadata = message.metadata?.custom as
        | Record<string, unknown>
        | undefined;
      return {
        id: message.id,
        threadId,
        parentId: parentId ?? null,
        role: message.role,
        content: cloneContent(message.content),
        ...(attachments.length > 0 && { attachments }),
        ...(metadata && { metadata }),
        createdAt: message.createdAt?.getTime?.() ?? Date.now(),
      };
    }));
    await syncStoredChatMessages(threadId, records, { pruneMissing: false });
    temporaryThreadCreation.delete(threadId);
  } catch (error) {
    markThreadIncognito(threadId);
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
        // Always regular, or assistant-ui unarchives a chat when opened.
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
      // Tracked, not awaited: assistant-ui withholds the first message until this resolves.
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
          // Only initialize() can promise this: it runs once for the id it just minted.
          neverSent: true,
          modelId: modelIdAtInit,
          modelGgufVariant: modelGgufVariantAtInit,
          createdAt: createdAtInit,
        }),
      );
      // Re-key handles a running stream filed under "__default".
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
      // No-op on archive state, or opening an archived chat unarchives it.
      await ensureStoredChatThread(remoteId);
    },

    async delete(remoteId: string) {
      await deleteStoredChatThreads([remoteId]);
    },

    async generateTitle(remoteId: string, messages: readonly ThreadMessage[]) {
      const autoTitle = useChatRuntimeStore.getState().autoTitle;
      // A bounded persistence wait can expire; the title is cosmetic so fall back to default.
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
      const answeredWith = answeringCheckpoint(
        firstAssistant?.metadata?.custom,
      );

      if (!autoTitle) {
        const title = fallbackTitleFromUserText(userText);
        await persistTitle(title);
        return streamTitle(title);
      }

      const key = pairId ? `pair:${pairId}` : `thread:${remoteId}`;
      if (inflightTitleByKey.has(key)) {
        return streamTitle(thread.title || defaultTitle);
      }

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
            checkpoint: titleCheckpoint(
              answeredWith,
              useChatRuntimeStore.getState().params.checkpoint,
            ),
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
  // Deep Research reserves an assistant placeholder, so the user message may not be last.
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
        // Queued runs have no direct-send reservation, so match pending work too.
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
      // assistant-ui bound unstable_threadId before the await; pass the real id.
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
  // A ref so the memoized runtime hook identity never changes.
  backgroundedRef?: { current: boolean },
  newThreadSwitchStateRef?: { current: NewThreadSwitchState },
): StudioRuntimeAdapters {
  const signalReady = useAppShellReadySignal();
  const aui = useAui();

  useEffect(() => registerLiveThreadView(aui), [aui]);

  useEffect(() => {
    const recoverCurrentThread = () => {
      const remoteId = aui.threadListItem().getState().remoteId;
      if (!remoteId) return;
      const generation = beginSavedHistoryReconciliation(remoteId);
      void readStoredChatMessages(remoteId)
        .then(({ messages, fromBackend }) => {
          if (isSavedHistoryReconciliationSuperseded(remoteId, generation)) {
            return;
          }
          if (aui.threadListItem().getState().remoteId !== remoteId) {
            return;
          }
          // A legacy browser copy served during an outage is older, not an external update.
          if (fromBackend) {
            reconcileOrdinarySavedMessagesInView(aui, remoteId, messages, {
              editingMessageId:
                useChatRuntimeStore.getState().editingMessageId ?? null,
            });
          }
          if (isSavedHistoryReconciliationSuperseded(remoteId, generation)) {
            return;
          }
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

  // Mirror Data-tab deletions, or a later sync writes the attachment back.
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
          // A runtime bootstraps on an empty thread first, so an unrequested load is not readiness.
          const loadedTheRequestedThread =
            !reloadReadyThreadId || loadedThreadId === reloadReadyThreadId;
          if (onInitialHistoryReady) {
            if (loadedTheRequestedThread) onInitialHistoryReady();
          } else if (
            modelType === "base" &&
            !pairId &&
            loadedTheRequestedThread
          ) {
            signalReady();
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
        // The endpoint can return just-terminalised rows; publish only live ones.
        activeGenerationRuns = activeGenerationRuns.filter(
          (run) => !isTerminalChatGenerationRun(run),
        );
        // Also drop runs the message snapshot shows finished (runs are read first).
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
        // Runs are read before messages; recheck when a message names a run the first read missed.
        if (!activeGenerationRunsLoaded) {
          // Retract stale answers, or another tab's live run would show as interrupted.
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
            // The server still has the run, so clear the local interrupted marker.
            generationLocallyInterrupted: false,
          };
        }
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
        // llama.cpp stops at the window so a larger saved count is stale; MLX runs past it by design.
        const localLimit = store.loadedIsGguf ? store.loadedContextLength : null;
        const withinLocalLimit =
          !localLimit || (savedUsage?.totalTokens ?? 0) <= localLimit;
        // Legacy unscoped usage is trusted only when a local window bounds it.
        const modelMatches = savedUsage?.modelId
          ? savedUsage.modelId === store.params.checkpoint
          : typeof store.loadedContextLength === "number" &&
            store.loadedContextLength > 0;
        const restoredUsage =
          savedUsage && withinLocalLimit && modelMatches ? savedUsage : null;
        const shownUsage = restoredUsage ?? estimateContextUsage(msgs);
        if (shownUsage) {
          // Key by the thread this loader read, not whichever is active after the await.
          store.setThreadContextUsage(remoteId, shownUsage);
          if (store.activeThreadId === remoteId) {
            store.setContextUsage(shownUsage);
          }
        }
        // Only when nothing was restored: the recount would overwrite exact totals with an estimate.
        if (!restoredUsage && modelType === "base" && !pairId) {
          void refreshContextUsage({ threadId: remoteId });
        }

        // Rebuild the tree when any parentId exists; fully legacy threads use fromArray.
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
          // Clear-all tombstones directly so a stalled request cannot block it.
          await awaitStoredChatThreadWrites(remoteId);
          if (isChatThreadDeleted(remoteId)) {
            await deleteStoredChatThreads([remoteId]);
            return;
          }
          // Not while backgrounded or mid-switch (mainThreadId is still the outgoing thread then).
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
          // Server-managed messages reject client edits with 409, so skip the save.
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
      signalReady,
    ],
  );

  // Always registered: the engine is resolved at listen() time.
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
          // Before document adapters: first match wins and .mkv/.mov must not fall through.
          new VideoAttachmentAdapter(),
          new TextAttachmentAdapter(),
          new HtmlAttachmentAdapter(),
          new PDFAttachmentAdapter(),
          new DocxAttachmentAdapter(),
          new OfficeAttachmentAdapter(),
          new OpenDocumentAttachmentAdapter(),
          new RtfAttachmentAdapter(),
          new ToolOnlyAttachmentAdapter(),
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

const MAX_PENDING_SAVED_THREAD_SWITCHES = 16;

type PendingSavedThreadSwitch = { id: string; settled: boolean };

type NewThreadSwitchState = {
  activeNonce: string | null;
  hasSwitched: boolean;
  // Bumped per switch: two switches for the same nonce can be in flight.
  attempt: number;
  // One entry per switch started, matched by id, not id shape.
  pendingSavedThreadIds: PendingSavedThreadSwitch[];
  // ?new= survives materialization, so the nonce/thread pair tells return from new.
  nonceThread: { nonce: string; threadId: string } | null;
  // Only record ownership from a thread this nonce's switch actually opened.
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
    // Paused too: the stop names every temporary queue on the page.
    if (isLoading || paused) {
      return;
    }
    newThreadSwitchStateRef.current.activeNonce = null;
    if (mainThreadId !== threadId) {
      // Bumped, not read, so any newer switch supersedes this one.
      const attemptAtStart = (newThreadSwitchStateRef.current.attempt += 1);
      // Duplicates included: each arrival must spend exactly one entry.
      const claims = newThreadSwitchStateRef.current.pendingSavedThreadIds;
      const claim: PendingSavedThreadSwitch = { id: threadId, settled: false };
      claims.push(claim);
      if (claims.length > MAX_PENDING_SAVED_THREAD_SWITCHES) {
        claims.splice(0, claims.length - MAX_PENDING_SAVED_THREAD_SWITCHES);
      }
      // A temporary chat is unreachable after this switch, so stop its queue.
      requestTemporaryPromptQueueStop();
      const switchResult = aui.threads().switchToThread(threadId) as unknown;
      if (
        switchResult &&
        typeof (switchResult as Promise<void>).then === "function"
      ) {
        // Both arms retire the claim, or a rejected switch's claim stays armed.
        void (switchResult as Promise<void>).then(
          () => {
            claim.settled = true;
          },
          () => {
            claim.settled = true;
            // Ahead of the staleness guard on purpose: it releases the reload shell.
            onSwitchFailed?.();
            // Only if still current, or a late rejection clears another view's active id.
            if (newThreadSwitchStateRef.current.attempt !== attemptAtStart) return;
            if (syncActiveThreadId) {
              useChatRuntimeStore.getState().setActiveThreadId(null);
            }
          },
        );
      } else {
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
    // Release every settled claim here; in-flight ones can still land wrong.
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
  const runActive = useChatRuntimeStore((s) =>
    Object.values(s.runningByThreadId).some(Boolean),
  );
  useEffect(() => {
    if (isLoading || paused) {
      return;
    }
    const switchState = newThreadSwitchStateRef.current;
    if (switchState.activeNonce === nonce) {
      return;
    }
    const recorded =
      switchState.nonceThread?.nonce === nonce
        ? switchState.nonceThread.threadId
        : null;
    const runtimeThreads = aui.threads().__internal_getAssistantRuntime?.();
    // Guarded: getItemById() throws for a dropped id rather than returning undefined.
    let recordedRemoteId: string | undefined;
    if (recorded) {
      try {
        recordedRemoteId = runtimeThreads?.threads
          .getItemById(recorded)
          .getState()?.remoteId;
      } catch {
        recordedRemoteId = undefined;
      }
    }
    // Deletion leaves the runtime item intact, so check tombstones.
    const returningToOwnChat = Boolean(
      recorded && recordedRemoteId && !isChatThreadDeleted(recordedRemoteId),
    );
    if (!returningToOwnChat) {
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
        // Chained: clearAttachments() may reject via the adapter's remove().
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
    requestTemporaryPromptQueueStop();
    // A project landing remounts with the old nonce, so the reopen must be cancellable.
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
    const settleReopenClaim = () => {
      if (!reopenClaim) return;
      reopenClaim.settled = true;
      const switchStateNow = newThreadSwitchStateRef.current;
      if (switchStateNow.activeNonce !== nonce) return;
      const at = switchStateNow.pendingSavedThreadIds.indexOf(reopenClaim);
      if (at !== -1) switchStateNow.pendingSavedThreadIds.splice(at, 1);
    };
    void Promise.resolve(
      returningToOwnChat && recorded
        ? aui.threads().switchToThread(recorded)
        : aui.threads().switchToNewThread(),
    ).then(
      () => {
        settleReopenClaim();
        {
          const switchStateNow = newThreadSwitchStateRef.current;
          if (switchStateNow.attempt === attempt) {
            switchStateNow.landedAttempt = attempt;
          }
        }
        if (!clearAfterSwitch) return;
        const switchStateNow = newThreadSwitchStateRef.current;
        // By attempt as well as nonce, or an older completion wipes a newer attachment.
        if (switchStateNow.attempt !== attempt) return;
        if (switchStateNow.activeNonce !== nonce) return;
        clearAttachments();
      },
      () => {
        settleReopenClaim();
        // Release the nonce, or the same New Chat can never be retried.
        const switchStateNow = newThreadSwitchStateRef.current;
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

  // Reassert this view's thread if a stale switch lands anyway.
  useEffect(() => {
    if (isLoading || paused) {
      return;
    }
    const switchState = newThreadSwitchStateRef.current;
    if (switchState.activeNonce !== nonce) {
      return;
    }
    // switchToThread() early-returns on the current target (assistant-ui#2577).
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
      // Recorded here: the id changes on materialization.
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
    // runActive is a dependency: refreshContextUsage declines while generating.
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

function NewThreadIdRegistrar(): null {
  const aui = useAui();
  // Register before passive effects read storage.
  useLayoutEffect(
    () =>
      registerNewThreadIdSource(() => aui.threads().getState().newThreadId),
    [aui],
  );
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
    // Only fills a hole, never overwrites a live id.
    if (useChatRuntimeStore.getState().activeThreadId != null) {
      return;
    }
    // `__LOCALID_` ids are still real rows; do not skip them.
    if (!mainThreadId) {
      return;
    }
    // An untouched landing must stay untouched; check remoteId, not id shape.
    const runtime = aui.threads().__internal_getAssistantRuntime?.();
    const { remoteId } =
      runtime?.threads.getItemById(mainThreadId).getState() ?? {};
    if (!remoteId) {
      return;
    }
    // Deletes tombstone storage, so check that too.
    if (isChatThreadDeleted(remoteId)) {
      return;
    }
    useChatRuntimeStore.getState().setActiveThreadId(mainThreadId);
  }, [aui, enabled, mainThreadId]);

  return null;
}

const THREAD_READ_RETRY_MS = 1_500;
const THREAD_READ_RETRIES = 2;
const THREAD_READ_TIMEOUT_MS = 8_000;

// Gated on hydration, or the initial settings response overwrites thread values.
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
    // Unsent chats have no row; only the pending-new-thread id identifies them.
    if (activeThreadId !== null && activeThreadId === pendingNewThreadId) {
      applyThreadScopedSettings(null, null);
      return;
    }
    if (!enabled) {
      // Compare panes share one composer, so they use installation defaults.
      applyThreadScopedSettings(null, null);
      return;
    }
    if (activeThreadId === null) {
      if (settingsHydrated) applyThreadScopedSettings(null, null);
      return;
    }
    // Start holding edits as soon as the id is known; settings may still be loading.
    beginThreadScopedPairing(activeThreadId);
    if (!settingsHydrated) {
      return () => {
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
    // Abort reads per attempt so losers do not stay open server-side.
    const reads = new Set<AbortController>();
    const abortReads = () => {
      for (const read of reads) read.abort();
      reads.clear();
    };

    const sync = () => {
      if (cancelled || paired) return;
      // Use defaults while the read is out, or a send could run with the outgoing chat's permissions.
      if (!defaulted) {
        defaulted = true;
        applyThreadScopedSettings(null, null);
      }
      beginThreadScopedPairing(activeThreadId);
      const read = new AbortController();
      reads.add(read);
      // The deadline covers the waits too, which are unbounded on their own.
      void Promise.race([
        Promise.all([
          // This chat's PATCH first, or the read returns the pre-edit snapshot.
          awaitThreadScopedSettingsWrite(activeThreadId),
          // And its row: on a first send the read can overtake the tracked POST.
          awaitStoredChatThreadWrites(activeThreadId),
        ]).then(() =>
          getStoredChatThreadReadResult(activeThreadId, {
            timeoutMs: THREAD_READ_TIMEOUT_MS,
            signal: read.signal,
          }),
        ),
        new Promise<never>((_, reject) =>
          setTimeout(() => {
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
          // A legacy fallback row means the backend GET failed; keep holding and retry.
          if (thread && !cacheable) {
            retryThreadRead();
            return;
          }
          if (!thread) {
            releaseHeldThreadScopedEdits();
            if (unpaired) return;
            unpaired = true;
            applyThreadScopedSettings(null, null);
            return;
          }
          paired = true;
          // Omitted fields come back as null, which is not a value to apply.
          applyThreadScopedSettings(
            activeThreadId,
            thread.settings
              ? sanitizeThreadScopedSettings(thread.settings)
              : null,
          );
        })
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
        applyThreadScopedSettings(null, null);
        releaseHeldThreadScopedEdits();
        toast.error("Could not load this chat's settings", {
          description:
            "It is using the default settings. Reopen the chat to try again.",
        });
        return;
      }
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
      abortReads();
      commitHeldThreadScopedEditsToTheirThread();
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, sync);
    };
  }, [activeThreadId, enabled, pendingNewThreadId, settingsHydrated]);

  return null;
}

// Read the on-screen branch: incognito threads store nothing.
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
        return null;
      }
    });
    return () => setActiveBranchReader(null);
  }, [aui, enabled]);

  return null;
}

function ThreadContextUsageRecount({
  enabled,
}: { enabled: boolean }): ReactElement | null {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const loadedContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  // A dependency: nothing else retries a count skipped while busy.
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
    // Only into a blank or estimated bar: restored usage is exact.
    const shown = useChatRuntimeStore.getState().contextUsage;
    if (shown != null && !shown.estimated) return;
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
      // assistant-ui runs before preflight sets runningByThreadId, so keep the cancel handle.
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
  // A ref: the save may resolve after the pane is hidden, so read at publish time.
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

      // Only publication is suppressed while backgrounded or mid-switch.
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

  // runEnd only reaches the main thread, so ask the runtime instead.
  const isRunActive = useCallback(
    (threadId: string): boolean => {
      const runtime = aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) {
        return false;
      }
      try {
        return runtime.threads.getById(threadId).getState().isRunning === true;
      } catch {
        // getById throws for deleted or detached threads.
        return false;
      }
    },
    [aui],
  );

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
    // Hidden renderers may never fire the next timer, so checkpoint on the way out.
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

// False while the chat tab is hidden: runtime stays mounted, views unmount.
export const ChatActiveContext = createContext(true);

export function useChatActive(): boolean {
  return useContext(ChatActiveContext);
}

// Both panes mount the same controls, so a window chord needs pane scoping.
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
  backgrounded?: boolean;
  onInitialHistoryReady?: () => void;
}): ReactElement {
  const signalReady = useAppShellReadySignal();
  // A ref so the memo never sees it change and rebuild the runtime.
  const backgroundedRef = useRef(backgrounded);
  backgroundedRef.current = backgrounded;
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
      signalReady();
    }
  }, [modelType, onInitialHistoryReady, pairId, signalReady]);

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
        <NewThreadIdRegistrar />
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
