// src/hooks/artifact-auto-index.ts
// Auto-indexes artifacts when written to thoughts/ directories

import type { PluginInput } from "@opencode-ai/plugin";
import { readFileSync } from "node:fs";
import { getArtifactIndex } from "../tools/artifact-index";
import { log } from "../utils/logger";

const LEDGER_PATH_PATTERN = /thoughts\/ledgers\/CONTINUITY_(.+)\.md$/;
const PLAN_PATH_PATTERN = /thoughts\/shared\/plans\/(.+)\.md$/;

export function parseLedger(content: string, filePath: string, sessionName: string) {
  const goalMatch = content.match(/## Goal\n([^\n]+)/);
  const stateMatch = content.match(/### In Progress\n- \[ \] ([^\n]+)/);
  const decisionsMatch = content.match(/## Key Decisions\n([\s\S]*?)(?=\n## |$)/);

  // Parse file operations from new ledger format
  const fileOpsSection = content.match(/## File Operations\n([\s\S]*?)(?=\n## |$)/);
  let filesRead = "";
  let filesModified = "";

  if (fileOpsSection) {
    const readMatch = fileOpsSection[1].match(/### Read\n([\s\S]*?)(?=\n### |$)/);
    const modifiedMatch = fileOpsSection[1].match(/### Modified\n([\s\S]*?)(?=\n### |$)/);

    if (readMatch) {
      // Extract paths from markdown list items like "- `path`"
      const paths = readMatch[1].match(/`([^`]+)`/g);
      filesRead = paths ? paths.map((p) => p.replace(/`/g, "")).join(",") : "";
    }

    if (modifiedMatch) {
      const paths = modifiedMatch[1].match(/`([^`]+)`/g);
      filesModified = paths ? paths.map((p) => p.replace(/`/g, "")).join(",") : "";
    }
  }

  return {
    id: `ledger-${sessionName}`,
    sessionName,
    filePath,
    goal: goalMatch?.[1] || "",
    stateNow: stateMatch?.[1] || "",
    keyDecisions: decisionsMatch?.[1]?.trim() || "",
    filesRead,
    filesModified,
  };
}

function parsePlan(content: string, filePath: string, fileName: string) {
  // Extract title (first heading)
  const titleMatch = content.match(/^# (.+)$/m);
  const title = titleMatch?.[1] || fileName;

  // Extract overview
  const overviewMatch = content.match(/## Overview\n\n([\s\S]*?)(?=\n## |$)/);
  const overview = overviewMatch?.[1]?.trim() || "";

  // Extract approach
  const approachMatch = content.match(/## Approach\n\n([\s\S]*?)(?=\n## |$)/);
  const approach = approachMatch?.[1]?.trim() || "";

  return {
    id: `plan-${fileName}`,
    title,
    filePath,
    overview,
    approach,
  };
}

export function createArtifactAutoIndexHook(_ctx: PluginInput) {
  return {
    "tool.execute.after": async (
      input: { tool: string; args?: Record<string, unknown> },
      _output: { output?: string },
    ) => {
      // Only process Write tool
      if (input.tool !== "write") return;

      const filePath = input.args?.filePath as string | undefined;
      if (!filePath) return;

      try {
        // Check if it's a ledger
        const ledgerMatch = filePath.match(LEDGER_PATH_PATTERN);
        if (ledgerMatch) {
          const content = readFileSync(filePath, "utf-8");
          const index = await getArtifactIndex();
          const record = parseLedger(content, filePath, ledgerMatch[1]);
          await index.indexLedger(record);
          return;
        }

        // Check if it's a plan
        const planMatch = filePath.match(PLAN_PATH_PATTERN);
        if (planMatch) {
          const content = readFileSync(filePath, "utf-8");
          const index = await getArtifactIndex();
          const record = parsePlan(content, filePath, planMatch[1]);
          await index.indexPlan(record);
          return;
        }
      } catch (e) {
        // Silent failure - don't interrupt user flow
        log.error("artifact-auto-index", `Error indexing ${filePath}`, e);
      }
    },
  };
}
