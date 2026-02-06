import type { PluginInput } from "@opencode-ai/plugin";

import { config } from "../utils/config";
import { getContextLimit } from "../utils/model-limits";

export interface ContextWindowMonitorConfig {
  /** Model context limits loaded from opencode.json */
  modelContextLimits?: Map<string, number>;
}

interface MonitorState {
  lastWarningTime: Map<string, number>;
  lastUsageRatio: Map<string, number>;
}

export function createContextWindowMonitorHook(ctx: PluginInput, hookConfig?: ContextWindowMonitorConfig) {
  const modelLimits = hookConfig?.modelContextLimits;
  const state: MonitorState = {
    lastWarningTime: new Map(),
    lastUsageRatio: new Map(),
  };

  function getEncouragementMessage(usageRatio: number): string {
    const remaining = Math.round((1 - usageRatio) * 100);

    if (usageRatio < config.contextWindow.warningThreshold) {
      return ""; // No message needed
    }

    if (usageRatio < config.contextWindow.criticalThreshold) {
      return `Context: ${remaining}% remaining. Plenty of room - don't rush.`;
    }

    return `Context: ${remaining}% remaining. Consider wrapping up or compacting soon.`;
  }

  return {
    // Inject context awareness into chat params
    "chat.params": async (
      input: { sessionID: string },
      output: { system?: string; options?: Record<string, unknown> },
    ) => {
      const usageRatio = state.lastUsageRatio.get(input.sessionID);

      if (usageRatio && usageRatio >= config.contextWindow.warningThreshold) {
        const message = getEncouragementMessage(usageRatio);
        if (message && output.system) {
          output.system = `${output.system}\n\n<context-status>${message}</context-status>`;
        }
      }
    },

    event: async ({ event }: { event: { type: string; properties?: unknown } }) => {
      const props = event.properties as Record<string, unknown> | undefined;

      // Cleanup on session delete
      if (event.type === "session.deleted") {
        const sessionInfo = props?.info as { id?: string } | undefined;
        if (sessionInfo?.id) {
          state.lastWarningTime.delete(sessionInfo.id);
          state.lastUsageRatio.delete(sessionInfo.id);
        }
        return;
      }

      // Track usage on message updates
      if (event.type === "message.updated") {
        const info = props?.info as Record<string, unknown> | undefined;
        const sessionID = info?.sessionID as string | undefined;

        if (!sessionID || info?.role !== "assistant") return;

        const tokens = info.tokens as { input?: number; cache?: { read?: number } } | undefined;
        const inputTokens = tokens?.input || 0;
        const cacheRead = tokens?.cache?.read || 0;
        const totalUsed = inputTokens + cacheRead;

        const modelID = (info.modelID as string) || "";
        const providerID = (info.providerID as string) || "";
        const contextLimit = getContextLimit(modelID, providerID, modelLimits);
        const usageRatio = totalUsed / contextLimit;

        state.lastUsageRatio.set(sessionID, usageRatio);

        // Show toast warning if threshold crossed
        if (usageRatio >= config.contextWindow.warningThreshold) {
          const lastWarning = state.lastWarningTime.get(sessionID) || 0;
          if (Date.now() - lastWarning > config.contextWindow.warningCooldownMs) {
            state.lastWarningTime.set(sessionID, Date.now());

            const remaining = Math.round((1 - usageRatio) * 100);
            const variant = usageRatio >= config.contextWindow.criticalThreshold ? "warning" : "info";

            await ctx.client.tui
              .showToast({
                body: {
                  title: "Context Window",
                  message: `${remaining}% remaining (${Math.round(totalUsed / 1000)}K / ${Math.round(contextLimit / 1000)}K tokens)`,
                  variant,
                  duration: config.timeouts.toastWarningMs,
                },
              })
              .catch(() => {});
          }
        }
      }
    },
  };
}
