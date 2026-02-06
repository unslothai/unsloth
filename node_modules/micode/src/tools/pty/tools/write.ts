// src/tools/pty/tools/write.ts
import { tool } from "@opencode-ai/plugin/tool";
import type { PTYManager } from "../manager";

const DESCRIPTION = `Sends input data to an active PTY session.

Use this tool to:
- Type commands or text into an interactive terminal
- Send special key sequences (Ctrl+C, Enter, arrow keys, etc.)
- Respond to prompts in interactive programs

Usage:
- \`id\`: The PTY session ID (from pty_spawn or pty_list)
- \`data\`: The input to send (text, commands, or escape sequences)

Common escape sequences:
- Enter/newline: "\\n" or "\\r"
- Ctrl+C (interrupt): "\\x03"
- Ctrl+D (EOF): "\\x04"
- Ctrl+Z (suspend): "\\x1a"
- Tab: "\\t"
- Arrow Up: "\\x1b[A"
- Arrow Down: "\\x1b[B"
- Arrow Right: "\\x1b[C"
- Arrow Left: "\\x1b[D"

Returns success or error message.

Examples:
- Send a command: data="ls -la\\n"
- Interrupt a process: data="\\x03"
- Answer a prompt: data="yes\\n"`;

/**
 * Parse escape sequences in a string to their actual byte values.
 * Handles: \n, \r, \t, \xNN (hex), \uNNNN (unicode), \\
 */
function parseEscapeSequences(input: string): string {
  return input.replace(/\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|[nrt0\\])/g, (match, seq: string) => {
    if (seq.startsWith("x")) {
      return String.fromCharCode(parseInt(seq.slice(1), 16));
    }
    if (seq.startsWith("u")) {
      return String.fromCharCode(parseInt(seq.slice(1), 16));
    }
    switch (seq) {
      case "n":
        return "\n";
      case "r":
        return "\r";
      case "t":
        return "\t";
      case "0":
        return "\0";
      case "\\":
        return "\\";
      default:
        return match;
    }
  });
}

export function createPtyWriteTool(manager: PTYManager) {
  return tool({
    description: DESCRIPTION,
    args: {
      id: tool.schema.string().describe("The PTY session ID (e.g., pty_a1b2c3d4)"),
      data: tool.schema.string().describe("The input data to send to the PTY"),
    },
    execute: async (args) => {
      const session = manager.get(args.id);
      if (!session) {
        throw new Error(`PTY session '${args.id}' not found. Use pty_list to see active sessions.`);
      }

      if (session.status !== "running") {
        throw new Error(`Cannot write to PTY '${args.id}' - session status is '${session.status}'.`);
      }

      // Parse escape sequences to actual bytes
      const parsedData = parseEscapeSequences(args.data);

      const success = manager.write(args.id, parsedData);
      if (!success) {
        throw new Error(`Failed to write to PTY '${args.id}'.`);
      }

      const preview = args.data.length > 50 ? `${args.data.slice(0, 50)}...` : args.data;
      const displayPreview = preview
        .replace(/\x03/g, "^C")
        .replace(/\x04/g, "^D")
        .replace(/\n/g, "\\n")
        .replace(/\r/g, "\\r");
      return `Sent ${parsedData.length} bytes to ${args.id}: "${displayPreview}"`;
    },
  });
}
