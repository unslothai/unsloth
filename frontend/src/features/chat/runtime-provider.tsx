import {
  AssistantRuntimeProvider,
  CompositeAttachmentAdapter,
  ExportedMessageRepository,
  type ExportedMessageRepositoryItem,
  RuntimeAdapterProvider,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  Suggestions,
  type ThreadHistoryAdapter,
  type ThreadMessage,
  type ThreadUserMessagePart,
  WebSpeechDictationAdapter,
  type unstable_RemoteThreadListAdapter,
  useAssistantRuntime,
  useAui,
  useLocalRuntime,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { createAssistantStream } from "assistant-stream";
import {
  type MutableRefObject,
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
} from "react";
import { createStreamAdapter } from "./adapter";
import { db } from "./db";
import type { MessageRecord, ModelType, RuntimeBridge } from "./types";

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

function threadStatus(archived: boolean): "archived" | "regular" {
  return archived ? "archived" : "regular";
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
        status: threadStatus(thread.archived),
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
          status: threadStatus(t.archived),
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

        const converted = msgs.map((m) => toThreadMessage(m));
        return ExportedMessageRepository.fromArray(converted);
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

function RuntimeBridgeCapture({
  bridgeRef,
}: {
  bridgeRef: MutableRefObject<RuntimeBridge | null>;
}): ReactElement | null {
  const runtime = useAssistantRuntime();
  useEffect(() => {
    bridgeRef.current = {
      switchToThread: (id) => runtime.threadList.switchToThread(id),
      switchToNewThread: () => runtime.threadList.switchToNewThread(),
    };
    return () => {
      bridgeRef.current = null;
    };
  }, [runtime, bridgeRef]);
  return null;
}

function ThreadAutoSwitch({
  threadId,
}: { threadId: string }): ReactElement | null {
  const runtime = useAssistantRuntime();
  useEffect(() => {
    function trySwitch() {
      const s = runtime.threadList.getState();
      if (s.isLoading || s.mainThreadId === threadId) {
        return false;
      }
      runtime.threadList.switchToThread(threadId);
      return true;
    }

    if (trySwitch()) {
      return;
    }

    const unsub = runtime.threadList.subscribe(() => {
      if (trySwitch()) {
        unsub();
      }
    });
    return unsub;
  }, [runtime, threadId]);
  return null;
}

export function ChatRuntimeProvider({
  children,
  modelType = "base",
  pairId,
  bridgeRef,
  initialThreadId,
}: {
  children: ReactNode;
  modelType?: ModelType;
  pairId?: string;
  bridgeRef?: MutableRefObject<RuntimeBridge | null>;
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
      "Solve the integral of x²·sin(x) step by step",
      "Write a Python function that finds the longest palindrome in a string",
      "Format a comparison of 3 databases as a markdown table with pros and cons",
    ]),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {bridgeRef && <RuntimeBridgeCapture bridgeRef={bridgeRef} />}
      {initialThreadId && <ThreadAutoSwitch threadId={initialThreadId} />}
      {children}
    </AssistantRuntimeProvider>
  );
}
