import type { PluginInput } from "@opencode-ai/plugin";

// Error patterns we can recover from
const RECOVERABLE_ERRORS = {
  TOOL_RESULT_MISSING: "tool_result block(s) missing",
  THINKING_BLOCK_ORDER: "thinking blocks must be at the start",
  THINKING_DISABLED: "thinking is not enabled",
  EMPTY_CONTENT: "content cannot be empty",
  INVALID_TOOL_RESULT: "tool_result must follow tool_use",
} as const;

type RecoverableErrorType = keyof typeof RECOVERABLE_ERRORS;

interface RecoveryState {
  processingErrors: Set<string>;
  recoveryAttempts: Map<string, number>;
}

const MAX_RECOVERY_ATTEMPTS = 3;

function extractErrorInfo(error: unknown): { message: string; messageIndex?: number } | null {
  if (!error) return null;

  let errorStr: string;
  if (typeof error === "string") {
    errorStr = error;
  } else if (error instanceof Error) {
    errorStr = error.message;
  } else {
    errorStr = JSON.stringify(error);
  }

  const errorLower = errorStr.toLowerCase();

  // Extract message index if present (e.g., "messages.5" or "message 5")
  const indexMatch = errorStr.match(/messages?[.\s](\d+)/i);
  const messageIndex = indexMatch ? parseInt(indexMatch[1], 10) : undefined;

  return { message: errorLower, messageIndex };
}

function identifyErrorType(errorMessage: string): RecoverableErrorType | null {
  for (const [type, pattern] of Object.entries(RECOVERABLE_ERRORS)) {
    if (errorMessage.includes(pattern.toLowerCase())) {
      return type as RecoverableErrorType;
    }
  }
  return null;
}

export function createSessionRecoveryHook(ctx: PluginInput) {
  const state: RecoveryState = {
    processingErrors: new Set(),
    recoveryAttempts: new Map(),
  };

  async function getSessionMessages(sessionID: string): Promise<unknown[]> {
    try {
      const resp = await ctx.client.session.messages({
        path: { id: sessionID },
        query: { directory: ctx.directory },
      });
      return (resp as { data?: unknown[] }).data || [];
    } catch {
      return [];
    }
  }

  async function abortSession(sessionID: string): Promise<void> {
    try {
      await ctx.client.session.abort({
        path: { id: sessionID },
        query: { directory: ctx.directory },
      });
    } catch {
      // Ignore abort errors
    }
  }

  async function resumeSession(
    sessionID: string,
    providerID?: string,
    modelID?: string,
    agent?: string,
  ): Promise<void> {
    try {
      // Find last user message to resume from
      const messages = await getSessionMessages(sessionID);
      const lastUserMsg = [...messages].reverse().find((m) => {
        const msg = m as Record<string, unknown>;
        const info = msg.info as Record<string, unknown> | undefined;
        return info?.role === "user";
      });

      if (!lastUserMsg) return;

      const parts = (lastUserMsg as Record<string, unknown>).parts as Array<{
        type: string;
        text?: string;
      }>;
      const text = parts?.find((p) => p.type === "text")?.text;

      if (!text) return;

      // Resume with continue prompt
      await ctx.client.session.prompt({
        path: { id: sessionID },
        body: {
          parts: [{ type: "text", text: "Continue from where you left off." }],
          ...(providerID && modelID ? { providerID, modelID } : {}),
          ...(agent ? { agent } : {}),
        },
        query: { directory: ctx.directory },
      });
    } catch {
      // Resume failed - user will need to manually continue
    }
  }

  async function attemptRecovery(
    sessionID: string,
    errorType: RecoverableErrorType,
    providerID?: string,
    modelID?: string,
    agent?: string,
  ): Promise<boolean> {
    const recoveryKey = `${sessionID}:${errorType}`;

    // Check recovery attempts
    const attempts = state.recoveryAttempts.get(recoveryKey) || 0;
    if (attempts >= MAX_RECOVERY_ATTEMPTS) {
      await ctx.client.tui
        .showToast({
          body: {
            title: "Recovery Failed",
            message: `Max attempts reached for ${errorType}. Manual intervention needed.`,
            variant: "error",
            duration: 5000,
          },
        })
        .catch(() => {});
      return false;
    }

    state.recoveryAttempts.set(recoveryKey, attempts + 1);

    await ctx.client.tui
      .showToast({
        body: {
          title: "Session Recovery",
          message: `Recovering from ${errorType.toLowerCase().replace(/_/g, " ")}...`,
          variant: "warning",
          duration: 3000,
        },
      })
      .catch(() => {});

    // Abort current session to stop the error state
    await abortSession(sessionID);

    // Wait a moment for abort to complete
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Attempt resume
    await resumeSession(sessionID, providerID, modelID, agent);

    await ctx.client.tui
      .showToast({
        body: {
          title: "Recovery Complete",
          message: "Session resumed. Continuing...",
          variant: "success",
          duration: 3000,
        },
      })
      .catch(() => {});

    return true;
  }

  return {
    event: async ({ event }: { event: { type: string; properties?: unknown } }) => {
      const props = event.properties as Record<string, unknown> | undefined;

      // Cleanup on session delete
      if (event.type === "session.deleted") {
        const sessionInfo = props?.info as { id?: string } | undefined;
        if (sessionInfo?.id) {
          // Clean up all recovery attempts for this session
          for (const key of state.recoveryAttempts.keys()) {
            if (key.startsWith(sessionInfo.id)) {
              state.recoveryAttempts.delete(key);
            }
          }
          for (const key of state.processingErrors) {
            if (key.startsWith(sessionInfo.id)) {
              state.processingErrors.delete(key);
            }
          }
        }
        return;
      }

      // Handle session errors
      if (event.type === "session.error") {
        const sessionID = props?.sessionID as string | undefined;
        const error = props?.error;

        if (!sessionID || !error) return;

        const errorInfo = extractErrorInfo(error);
        if (!errorInfo) return;

        const errorType = identifyErrorType(errorInfo.message);
        if (!errorType) return;

        // Prevent duplicate processing
        const errorKey = `${sessionID}:${errorType}:${Date.now()}`;
        if (state.processingErrors.has(errorKey)) return;
        state.processingErrors.add(errorKey);

        // Clear old error keys after 10 seconds
        setTimeout(() => state.processingErrors.delete(errorKey), 10000);

        // Attempt recovery
        await attemptRecovery(sessionID, errorType);
      }

      // Handle message errors
      if (event.type === "message.updated") {
        const info = props?.info as Record<string, unknown> | undefined;
        const sessionID = info?.sessionID as string | undefined;
        const error = info?.error;

        if (!sessionID || !error) return;

        const errorInfo = extractErrorInfo(error);
        if (!errorInfo) return;

        const errorType = identifyErrorType(errorInfo.message);
        if (!errorType) return;

        // Prevent duplicate processing
        const errorKey = `${sessionID}:${errorType}:${Date.now()}`;
        if (state.processingErrors.has(errorKey)) return;
        state.processingErrors.add(errorKey);

        setTimeout(() => state.processingErrors.delete(errorKey), 10000);

        const providerID = info.providerID as string | undefined;
        const modelID = info.modelID as string | undefined;
        const agent = info.agent as string | undefined;

        await attemptRecovery(sessionID, errorType, providerID, modelID, agent);
      }
    },
  };
}
