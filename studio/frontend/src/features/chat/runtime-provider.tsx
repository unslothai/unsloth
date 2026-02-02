import {
  type AttachmentAdapter,
  AssistantRuntimeProvider,
  type CompleteAttachment,
  CompositeAttachmentAdapter,
  ExportedMessageRepository,
  type ExportedMessageRepositoryItem,
  type PendingAttachment,
  RuntimeAdapterProvider,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  Suggestions,
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
import { type ReactElement, type ReactNode, useEffect, useMemo } from "react";
import { extractText, getDocumentProxy } from "unpdf";
import { createStreamAdapter } from "./adapter";
import { db } from "./db";
import type { MessageRecord, ModelType } from "./types";

class PDFAttachmentAdapter implements AttachmentAdapter {
  accept = "application/pdf";

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

  async remove(): Promise<void> {}
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
        const msgs = await db.messages
          .where("threadId")
          .equals(remoteId)
          .sortBy("createdAt");

        return ExportedMessageRepository.fromArray(msgs.map(toThreadMessage));
      },

      async append({ message }: ExportedMessageRepositoryItem) {
        const { remoteId } = await aui.threadListItem().initialize();
        const content = Array.isArray(message.content)
          ? JSON.parse(JSON.stringify(message.content))
          : [];
        const custom = message.metadata?.custom;
        await db.messages.put({
          id: message.id,
          threadId: remoteId,
          role: message.role,
          content,
          ...(custom && Object.keys(custom).length > 0 && { metadata: custom }),
          createdAt: message.createdAt?.getTime() ?? Date.now(),
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
        new SimpleImageAttachmentAdapter(),
        new SimpleTextAttachmentAdapter(),
        new PDFAttachmentAdapter(),
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

const chatAdapter = createStreamAdapter();
const useRuntimeHook = (): ReturnType<typeof useLocalRuntime> =>
  useLocalRuntime(chatAdapter);

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

export function ChatRuntimeProvider({
  children,
  modelType = "base",
  pairId,
  initialThreadId,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  initialThreadId?: string;
}): ReactElement {
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useRuntimeHook,
    adapter: {
      ...createDexieAdapter(modelType, pairId),
      unstable_Provider: ThreadHistoryProvider,
    },
  });

  const aui = useAui({
    suggestions: Suggestions([
      "Draw a simple flowchart of a login system using Mermaid",
      "Solve the integral of x\u00B2\u00B7sin(x) step by step",
      "Write a Python function that finds the longest palindrome in a string",
      "Format a comparison of 3 databases as a markdown table with pros and cons",
    ]),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {initialThreadId && <ThreadAutoSwitch threadId={initialThreadId} />}
      {children}
    </AssistantRuntimeProvider>
  );
}
