// src/hooks/file-ops-tracker.ts
import type { PluginInput } from "@opencode-ai/plugin";

interface FileOps {
  read: Set<string>;
  modified: Set<string>;
}

// Per-session file operation tracking
const sessionFileOps = new Map<string, FileOps>();

function getOrCreateOps(sessionID: string): FileOps {
  let ops = sessionFileOps.get(sessionID);
  if (!ops) {
    ops = { read: new Set(), modified: new Set() };
    sessionFileOps.set(sessionID, ops);
  }
  return ops;
}

export function trackFileOp(sessionID: string, operation: "read" | "write" | "edit", filePath: string): void {
  const ops = getOrCreateOps(sessionID);
  if (operation === "read") {
    ops.read.add(filePath);
  } else {
    // write and edit both modify files
    ops.modified.add(filePath);
  }
}

export function getFileOps(sessionID: string): FileOps {
  const ops = sessionFileOps.get(sessionID);
  if (!ops) {
    return { read: new Set(), modified: new Set() };
  }
  return ops;
}

export function clearFileOps(sessionID: string): void {
  sessionFileOps.delete(sessionID);
}

export function getAndClearFileOps(sessionID: string): FileOps {
  const ops = getFileOps(sessionID);
  // Return copies of the sets before clearing
  const result = {
    read: new Set(ops.read),
    modified: new Set(ops.modified),
  };
  clearFileOps(sessionID);
  return result;
}

export function formatFileOpsForPrompt(ops: FileOps): string {
  const readPaths = Array.from(ops.read).sort();
  const modifiedPaths = Array.from(ops.modified).sort();

  let result = "<file-operations>\n";
  result += `Read: ${readPaths.length > 0 ? readPaths.join(", ") : "(none)"}\n`;
  result += `Modified: ${modifiedPaths.length > 0 ? modifiedPaths.join(", ") : "(none)"}\n`;
  result += "</file-operations>";

  return result;
}

export function createFileOpsTrackerHook(_ctx: PluginInput) {
  return {
    "tool.execute.after": async (
      input: { tool: string; sessionID: string; args?: Record<string, unknown> },
      _output: { output?: string },
    ) => {
      const toolName = input.tool.toLowerCase();

      // Only track read, write, edit tools
      if (!["read", "write", "edit"].includes(toolName)) {
        return;
      }

      // Extract file path from args
      const filePath = input.args?.filePath as string | undefined;
      if (!filePath) return;

      trackFileOp(input.sessionID, toolName as "read" | "write" | "edit", filePath);
    },

    event: async ({ event }: { event: { type: string; properties?: unknown } }) => {
      // Clean up on session delete
      if (event.type === "session.deleted") {
        const props = event.properties as { info?: { id?: string } } | undefined;
        if (props?.info?.id) {
          clearFileOps(props.info.id);
        }
      }
    },
  };
}
