// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "base" | "lora" | "model1" | "model2";

export type ChatView =
  | {
      mode: "project";
      projectId: string;
    }
  | {
      mode: "single";
      threadId?: string;
      newThreadNonce?: string;
      projectId?: string | null;
    }
  | { mode: "compare"; pairId: string; projectId?: string | null };

export interface ProjectRecord {
  id: string;
  name: string;
  instructions?: string;
  rootPath?: string | null;
  sandboxPath?: string | null;
  archived: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  modelId?: string;
  modelGgufVariant?: string | null;
  pairId?: string;
  projectId?: string | null;
  archived: boolean;
  createdAt: number;
  updatedAt?: number;
  /** OpenAI shell tool container id from a prior response. When set, the next turn reuses it via
   *  `environment.type="container_reference"` so the model can read files it wrote earlier; else
   *  auto-creates one. Containers expire after ~20 min idle; on a stale id the backend emits
   *  `container_invalidated` and the chat-adapter clears this field. */
  openaiCodeExecContainerId?: string | null;
  /** Anthropic code_execution container id from a prior response. When set, the next turn sends a
   *  top-level `container` on /v1/messages so filesystem state persists; else auto-creates one.
   *  Containers expire after ~1 hour; on a stale id the backend emits `container_invalidated` and
   *  the chat-adapter clears this field. */
  anthropicCodeExecContainerId?: string | null;
  /** If this thread was created via fork-from-message, points back at the source thread and
   *  branch-point message. Null or undefined for non-fork threads. Used by the sidebar "fork" badge
   *  and the parent thread's "N forks" indicator. */
  forkedFromThreadId?: string | null;
  forkedFromMessageId?: string | null;
  /** this chat's own settings, applied when it is opened; absent means the global ones. */
  settings?:
    | import("./utils/thread-scoped-settings").ThreadScopedSettings
    | null;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  parentId?: string | null;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}

/** One conversation parsed out of an import file, before it is written to storage. */
export interface ParsedConversation {
  title: string;
  threadId: string;
  messages: MessageRecord[];
  /** Open WebUI exports carry the flag; other formats leave it unset. */
  archived?: boolean;
  createdAt?: number;
}
