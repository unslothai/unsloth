import type { PluginInput } from "@opencode-ai/plugin";
import { config } from "../utils/config";

// Tools that benefit from truncation
const TRUNCATABLE_TOOLS = ["grep", "Grep", "glob", "Glob", "ast_grep_search"];

function estimateTokens(text: string): number {
  return Math.ceil(text.length / config.tokens.charsPerToken);
}

function truncateToTokenLimit(
  output: string,
  maxTokens: number,
  preserveLines: number = config.tokens.preserveHeaderLines,
): string {
  const currentTokens = estimateTokens(output);

  if (currentTokens <= maxTokens) {
    return output;
  }

  const lines = output.split("\n");

  // Preserve header lines
  const headerLines = lines.slice(0, preserveLines);
  const remainingLines = lines.slice(preserveLines);

  // Calculate available tokens for content
  const headerTokens = estimateTokens(headerLines.join("\n"));
  const truncationMsgTokens = 50; // Reserve for truncation message
  const availableTokens = maxTokens - headerTokens - truncationMsgTokens;

  if (availableTokens <= 0) {
    return `${headerLines.join("\n")}\n\n[Output truncated - context window limit reached]`;
  }

  // Accumulate lines until we hit the limit
  const resultLines: string[] = [];
  let usedTokens = 0;
  let truncatedCount = 0;

  for (const line of remainingLines) {
    const lineTokens = estimateTokens(line);
    if (usedTokens + lineTokens > availableTokens) {
      truncatedCount = remainingLines.length - resultLines.length;
      break;
    }
    resultLines.push(line);
    usedTokens += lineTokens;
  }

  if (truncatedCount === 0) {
    return output;
  }

  return [
    ...headerLines,
    ...resultLines,
    "",
    `[${truncatedCount} more lines truncated due to context window limit]`,
  ].join("\n");
}

interface TruncationState {
  sessionTokenUsage: Map<string, { used: number; limit: number }>;
}

export function createTokenAwareTruncationHook(ctx: PluginInput) {
  const state: TruncationState = {
    sessionTokenUsage: new Map(),
  };

  async function updateTokenUsage(sessionID: string): Promise<{ used: number; limit: number }> {
    try {
      const resp = await ctx.client.session.messages({
        path: { id: sessionID },
        query: { directory: ctx.directory },
      });

      const messages = (resp as { data?: unknown[] }).data;
      if (!Array.isArray(messages) || messages.length === 0) {
        return { used: 0, limit: config.tokens.defaultContextLimit };
      }

      // Find last assistant message with usage info
      const lastAssistant = [...messages].reverse().find((m) => {
        const msg = m as Record<string, unknown>;
        const info = msg.info as Record<string, unknown> | undefined;
        return info?.role === "assistant";
      }) as Record<string, unknown> | undefined;

      if (!lastAssistant) {
        return { used: 0, limit: config.tokens.defaultContextLimit };
      }

      const info = lastAssistant.info as Record<string, unknown> | undefined;
      const usage = info?.usage as Record<string, unknown> | undefined;

      const inputTokens = (usage?.inputTokens as number) || 0;
      const cacheRead = (usage?.cacheReadInputTokens as number) || 0;
      const used = inputTokens + cacheRead;

      // Get model limit (simplified - use default for now)
      const limit = config.tokens.defaultContextLimit;

      const result = { used, limit };
      state.sessionTokenUsage.set(sessionID, result);
      return result;
    } catch {
      return state.sessionTokenUsage.get(sessionID) || { used: 0, limit: config.tokens.defaultContextLimit };
    }
  }

  function calculateMaxOutputTokens(used: number, limit: number): number {
    const remaining = limit - used;
    const available = Math.floor(remaining * config.tokens.safetyMargin);

    if (available <= 0) {
      return 0;
    }

    return Math.min(available, config.tokens.defaultMaxOutputTokens);
  }

  return {
    // Update token usage when assistant messages are received
    event: async ({ event }: { event: { type: string; properties?: unknown } }) => {
      const props = event.properties as Record<string, unknown> | undefined;

      if (event.type === "session.deleted") {
        const sessionInfo = props?.info as { id?: string } | undefined;
        if (sessionInfo?.id) {
          state.sessionTokenUsage.delete(sessionInfo.id);
        }
        return;
      }

      // Update usage on message updates
      if (event.type === "message.updated") {
        const info = props?.info as Record<string, unknown> | undefined;
        const sessionID = info?.sessionID as string | undefined;
        if (sessionID && info?.role === "assistant") {
          await updateTokenUsage(sessionID);
        }
      }
    },

    // Truncate tool output
    "tool.execute.after": async (input: { name: string; sessionID: string }, output: { output?: string }) => {
      // Only truncate specific tools
      if (!TRUNCATABLE_TOOLS.includes(input.name)) {
        return;
      }

      if (!output.output || typeof output.output !== "string") {
        return;
      }

      try {
        // Get current token usage
        const { used, limit } = await updateTokenUsage(input.sessionID);
        const maxTokens = calculateMaxOutputTokens(used, limit);

        if (maxTokens <= 0) {
          output.output = "[Output suppressed - context window exhausted. Consider compacting.]";
          return;
        }

        // Truncate if needed
        const currentTokens = estimateTokens(output.output);
        if (currentTokens > maxTokens) {
          output.output = truncateToTokenLimit(output.output, maxTokens);
        }
      } catch {
        // On error, apply static truncation as fallback
        const currentTokens = estimateTokens(output.output);
        if (currentTokens > config.tokens.defaultMaxOutputTokens) {
          output.output = truncateToTokenLimit(output.output, config.tokens.defaultMaxOutputTokens);
        }
      }
    },
  };
}
