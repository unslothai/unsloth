import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { PluginInput } from "@opencode-ai/plugin";

import { config } from "../utils/config";
import { extractErrorMessage } from "../utils/errors";
import { getContextLimit } from "../utils/model-limits";

export interface AutoCompactConfig {
  /** Compaction threshold (0-1), defaults to config.compaction.threshold */
  compactionThreshold?: number;
  /** Model context limits loaded from opencode.json */
  modelContextLimits?: Map<string, number>;
}

interface PendingCompaction {
  resolve: () => void;
  reject: (error: Error) => void;
  timeoutId: ReturnType<typeof setTimeout>;
}

interface AutoCompactState {
  inProgress: Set<string>;
  lastCompactTime: Map<string, number>;
  pendingCompactions: Map<string, PendingCompaction>;
}

export function createAutoCompactHook(ctx: PluginInput, hookConfig?: AutoCompactConfig) {
  const threshold = hookConfig?.compactionThreshold ?? config.compaction.threshold;
  const modelLimits = hookConfig?.modelContextLimits;

  const state: AutoCompactState = {
    inProgress: new Set(),
    lastCompactTime: new Map(),
    pendingCompactions: new Map(),
  };

  async function writeSummaryToLedger(sessionID: string): Promise<void> {
    try {
      // Fetch session messages to find the summary
      const resp = await ctx.client.session.messages({
        path: { id: sessionID },
        query: { directory: ctx.directory },
      });

      const messages = (resp as { data?: unknown[] }).data;
      if (!Array.isArray(messages)) return;

      // Find the summary message (has summary: true)
      const summaryMsg = [...messages].reverse().find((m) => {
        const msg = m as Record<string, unknown>;
        const info = msg.info as Record<string, unknown> | undefined;
        return info?.role === "assistant" && info?.summary === true;
      }) as Record<string, unknown> | undefined;

      if (!summaryMsg) return;

      // Extract text parts from the summary
      const parts = summaryMsg.parts as Array<{ type: string; text?: string }> | undefined;
      if (!parts) return;

      const summaryText = parts
        .filter((p) => p.type === "text" && p.text)
        .map((p) => p.text)
        .join("\n\n");

      if (!summaryText.trim()) return;

      // Create ledger directory if needed
      const ledgerDir = join(ctx.directory, config.paths.ledgerDir);
      await mkdir(ledgerDir, { recursive: true });

      // Write ledger file - summary is already structured (Factory.ai/pi-mono format)
      const timestamp = new Date().toISOString();
      const sessionName = sessionID.slice(0, 8); // Use first 8 chars of session ID
      const ledgerPath = join(ledgerDir, `${config.paths.ledgerPrefix}${sessionName}.md`);

      // Add metadata header, then the structured summary as-is
      const ledgerContent = `---
session: ${sessionName}
updated: ${timestamp}
---

${summaryText}
`;

      await writeFile(ledgerPath, ledgerContent, "utf-8");
    } catch (e) {
      // Don't fail the compaction flow if ledger write fails
      console.error("[auto-compact] Failed to write ledger:", e);
    }
  }

  function waitForCompaction(sessionID: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        state.pendingCompactions.delete(sessionID);
        reject(new Error("Compaction timed out"));
      }, config.compaction.timeoutMs);

      state.pendingCompactions.set(sessionID, { resolve, reject, timeoutId });
    });
  }

  async function triggerCompaction(
    sessionID: string,
    providerID: string,
    modelID: string,
    usageRatio: number,
  ): Promise<void> {
    if (state.inProgress.has(sessionID)) {
      return;
    }

    // Check cooldown
    const lastCompact = state.lastCompactTime.get(sessionID) || 0;
    if (Date.now() - lastCompact < config.compaction.cooldownMs) {
      return;
    }

    state.inProgress.add(sessionID);

    try {
      const usedPercent = Math.round(usageRatio * 100);
      const thresholdPercent = Math.round(threshold * 100);

      await ctx.client.tui
        .showToast({
          body: {
            title: "Auto Compacting",
            message: `Context at ${usedPercent}% (threshold: ${thresholdPercent}%). Summarizing...`,
            variant: "warning",
            duration: config.timeouts.toastWarningMs,
          },
        })
        .catch(() => {});

      // Set up listener BEFORE calling summarize to avoid race condition
      // (summary message event could fire before we start listening)
      const compactionPromise = waitForCompaction(sessionID);

      // Start the compaction - this returns immediately while compaction runs async
      await ctx.client.session.summarize({
        path: { id: sessionID },
        body: { providerID, modelID },
        query: { directory: ctx.directory },
      });

      // Wait for the summary message to be created (message.updated with summary: true)
      await compactionPromise;

      state.lastCompactTime.set(sessionID, Date.now());

      // Write summary to ledger file (only after compaction is confirmed complete)
      await writeSummaryToLedger(sessionID);

      await ctx.client.tui
        .showToast({
          body: {
            title: "Compaction Complete",
            message: "Session summarized. Continuing...",
            variant: "success",
            duration: config.timeouts.toastSuccessMs,
          },
        })
        .catch(() => {});

      // Auto-continue after compaction - prompt the agent to resume work
      await ctx.client.session
        .prompt({
          path: { id: sessionID },
          body: {
            parts: [
              {
                type: "text",
                text: "Context was compacted. Continue from where you left off - check the 'In Progress' and 'Next Steps' sections in the summary above.",
              },
            ],
            model: { providerID, modelID },
          },
          query: { directory: ctx.directory },
        })
        .catch(() => {
          // If auto-continue fails, user can manually prompt
        });
    } catch (e) {
      const errorMsg = extractErrorMessage(e);
      await ctx.client.tui
        .showToast({
          body: {
            title: "Compaction Failed",
            message: errorMsg.slice(0, 100),
            variant: "error",
            duration: config.timeouts.toastErrorMs,
          },
        })
        .catch(() => {});
    } finally {
      state.inProgress.delete(sessionID);
    }
  }

  return {
    event: async ({ event }: { event: { type: string; properties?: unknown } }) => {
      const props = event.properties as Record<string, unknown> | undefined;

      // Cleanup on session delete
      if (event.type === "session.deleted") {
        const sessionInfo = props?.info as { id?: string } | undefined;
        if (sessionInfo?.id) {
          state.inProgress.delete(sessionInfo.id);
          state.lastCompactTime.delete(sessionInfo.id);
          const pending = state.pendingCompactions.get(sessionInfo.id);
          if (pending) {
            clearTimeout(pending.timeoutId);
            state.pendingCompactions.delete(sessionInfo.id);
            pending.reject(new Error("Session deleted"));
          }
        }
        return;
      }

      // Monitor message events
      if (event.type === "message.updated") {
        const info = props?.info as Record<string, unknown> | undefined;
        const sessionID = info?.sessionID as string | undefined;

        if (!sessionID || info?.role !== "assistant") return;

        // Check if this is a summary message - signals compaction complete
        if (info?.summary === true) {
          const pending = state.pendingCompactions.get(sessionID);
          if (pending) {
            clearTimeout(pending.timeoutId);
            state.pendingCompactions.delete(sessionID);
            pending.resolve();
          }
          return;
        }

        // Skip triggering compaction if we're already waiting for one
        if (state.pendingCompactions.has(sessionID)) return;

        const tokens = info?.tokens as { input?: number; cache?: { read?: number } } | undefined;
        const inputTokens = tokens?.input || 0;
        const cacheRead = tokens?.cache?.read || 0;
        const totalUsed = inputTokens + cacheRead;

        if (totalUsed === 0) return;

        const modelID = (info?.modelID as string) || "";
        const providerID = (info?.providerID as string) || "";
        const contextLimit = getContextLimit(modelID, providerID, modelLimits);
        const usageRatio = totalUsed / contextLimit;

        // Trigger compaction if over threshold
        if (usageRatio >= threshold) {
          triggerCompaction(sessionID, providerID, modelID, usageRatio);
        }
      }
    },
  };
}
