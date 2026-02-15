import {
  AssistantRuntimeProvider,
  type AttachmentAdapter,
  type CompleteAttachment,
  CompositeAttachmentAdapter,
  ExportedMessageRepository,
  type ExportedMessageRepositoryItem,
  type PendingAttachment,
  RuntimeAdapterProvider,
  Suggestions,
  SimpleTextAttachmentAdapter,
  type ThreadHistoryAdapter,
  type ThreadMessage,
  type ThreadUserMessagePart,
  WebSpeechDictationAdapter,
  type unstable_RemoteThreadListAdapter,
  useAui,
  useAuiState,
  useLocalRuntime,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import mammoth from "mammoth";
import { type ReactElement, type ReactNode, useEffect, useMemo } from "react";
import { extractText, getDocumentProxy } from "unpdf";
import { createOpenAIStreamAdapter } from "./api/chat-adapter";
import { db } from "./db";
import type { MessageRecord, ModelType } from "./types";

const DEFAULT_SUGGESTIONS = [
  "Draw a simple flowchart of a login system using Mermaid",
  "Solve the integral of x²·sin(x) step by step",
  "Write a Python function that finds the longest palindrome in a string",
  "Format a comparison of 3 databases as a markdown table with pros and cons",
];

class VisionImageAdapter implements AttachmentAdapter {
  accept = "image/jpeg,image/png,image/webp,image/gif";

  async add({ file }: { file: File }): Promise<PendingAttachment> {
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

function toThreadMessage(m: MessageRecord): ThreadMessage {
  const base = {
    id: m.id,
    createdAt: new Date(m.createdAt),
    content:
      Array.isArray(m.content) && m.content.length > 0
        ? m.content
        : [{ type: "text" as const, text: "" }],
  };

  if (m.role === "user") {
    return {
      ...base,
      role: "user" as const,
      attachments: [],
      metadata: { custom: {} },
    };
  }
  return {
    ...base,
    role: "assistant" as const,
    status: { type: "complete" as const, reason: "unknown" as const },
    metadata: {
      custom: (m.metadata as Record<string, unknown>) ?? {},
      steps: [],
      unstable_annotations: [],
      unstable_data: [],
      unstable_state: null,
    },
  };
}

function createDexieAdapter(
  modelType: ModelType,
  pairId?: string,
): unstable_RemoteThreadListAdapter {
  return {
    async fetch(remoteId: string) {
      const thread = await db.threads.get(remoteId);
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
      const threads = await db.threads
        .where("modelType")
        .equals(modelType)
        .reverse()
        .sortBy("createdAt");
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
      await db.threads.add({
        id: threadId,
        title: "New Chat",
        modelType,
        pairId,
        archived: false,
        createdAt: Date.now(),
      });
      return { remoteId: threadId, externalId: undefined };
    },

    async rename(remoteId: string, newTitle: string) {
      await db.threads.update(remoteId, { title: newTitle });
    },

    async archive(remoteId: string) {
      await db.threads.update(remoteId, { archived: true });
    },

    async unarchive(remoteId: string) {
      await db.threads.update(remoteId, { archived: false });
    },

    async delete(remoteId: string) {
      await db.messages.where("threadId").equals(remoteId).delete();
      await db.threads.delete(remoteId);
    },

    async generateTitle(remoteId: string, messages: readonly ThreadMessage[]) {
      const firstUser = messages.find((m) => m.role === "user");
      const textParts =
        firstUser?.content.filter(
          (part): part is Extract<ThreadUserMessagePart, { type: "text" }> =>
            part.type === "text",
        ) ?? [];
      const text = textParts.map((part) => part.text).join("") || "New Chat";
      const title = text.slice(0, 60) + (text.length > 60 ? "..." : "");

      await db.threads.update(remoteId, { title });

      if (pairId) {
        const paired = await db.threads
          .where("pairId")
          .equals(pairId)
          .filter((t) => t.id !== remoteId)
          .first();
        if (paired) {
          await db.threads.update(paired.id, { title });
        }
      }

      return createAssistantStream((controller) => {
        controller.appendText(title);
        controller.close();
      });
    },
  };
}

function ThreadHistoryProvider({
  children,
}: { children?: ReactNode }): ReactElement {
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
        const msgs = await db.messages.where("threadId").equals(remoteId).toArray();
        msgs.sort((a, b) => {
          if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
          const aOrder = roleOrder[a.role] ?? 99;
          const bOrder = roleOrder[b.role] ?? 99;
          if (aOrder !== bOrder) return aOrder - bOrder;
          return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
        });

        return ExportedMessageRepository.fromArray(msgs.map(toThreadMessage));
      },

      async append({ message }: ExportedMessageRepositoryItem) {
        const { remoteId } = await aui.threadListItem().initialize();
        const content = Array.isArray(message.content)
          ? JSON.parse(JSON.stringify(message.content))
          : [];
        const custom = message.metadata?.custom;
        const existing = await db.messages.get(message.id);
        const createdAt =
          existing?.createdAt ??
          message.createdAt?.getTime?.() ??
          Date.now();
        await db.messages.put({
          id: message.id,
          threadId: remoteId,
          role: message.role,
          content,
          ...(custom && Object.keys(custom).length > 0 && { metadata: custom }),
          createdAt,
        });
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
        new SimpleTextAttachmentAdapter(),
        new PDFAttachmentAdapter(),
        new DocxAttachmentAdapter(),
      ]),
    [],
  );
  const adapters = useMemo(
    () => ({ history, dictation, attachments }),
    [history, dictation, attachments],
  );

  return (
    <RuntimeAdapterProvider adapters={adapters}>
      {children}
    </RuntimeAdapterProvider>
  );
}

const chatAdapter = createOpenAIStreamAdapter();

function useRuntimeHook(): ReturnType<typeof useLocalRuntime> {
  return useLocalRuntime(chatAdapter);
}

function ThreadAutoSwitch({
  threadId,
}: { threadId: string }): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);

  useEffect(() => {
    if (!isLoading && mainThreadId !== threadId) {
      aui.threads().switchToThread(threadId);
    }
  }, [aui, isLoading, mainThreadId, threadId]);

  return null;
}

function ThreadNewChatSwitch({
  nonce,
}: { nonce: string }): ReactElement | null {
  const aui = useAui();
  const isLoading = useAuiState(({ threads }) => threads.isLoading);

  useEffect(() => {
    if (!isLoading) {
      aui.threads().switchToNewThread();
    }
  }, [aui, isLoading, nonce]);

  return null;
}

export function ChatRuntimeProvider({
  children,
  modelType = "base",
  pairId,
  initialThreadId,
  newThreadNonce,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  initialThreadId?: string;
  newThreadNonce?: string;
}): ReactElement {
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useRuntimeHook,
    adapter: {
      ...createDexieAdapter(modelType, pairId),
      unstable_Provider: ThreadHistoryProvider,
    },
  });

  const aui = useAui({
    suggestions: Suggestions(DEFAULT_SUGGESTIONS),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {initialThreadId && <ThreadAutoSwitch threadId={initialThreadId} />}
      {!initialThreadId && newThreadNonce && (
        <ThreadNewChatSwitch nonce={newThreadNonce} />
      )}
      {children}
    </AssistantRuntimeProvider>
  );
}
