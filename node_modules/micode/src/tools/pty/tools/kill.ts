// src/tools/pty/tools/kill.ts
import { tool } from "@opencode-ai/plugin/tool";
import type { PTYManager } from "../manager";

const DESCRIPTION = `Terminates a PTY session and optionally cleans up its buffer.

Use this tool to:
- Stop a running process (sends SIGTERM)
- Clean up an exited session to free memory
- Remove a session from the list

Usage:
- \`id\`: The PTY session ID (from pty_spawn or pty_list)
- \`cleanup\`: If true, removes the session and frees the buffer (default: false)

Behavior:
- If the session is running, it will be killed (status becomes "killed")
- If cleanup=false (default), the session remains in the list with its output buffer intact
- If cleanup=true, the session is removed entirely and the buffer is freed
- Keeping sessions without cleanup allows you to compare logs between runs

Tips:
- Use cleanup=false if you might want to read the output later
- Use cleanup=true when you're done with the session entirely
- To send Ctrl+C instead of killing, use pty_write with data="\\x03"

Examples:
- Kill but keep logs: cleanup=false (or omit)
- Kill and remove: cleanup=true`;

export function createPtyKillTool(manager: PTYManager) {
  return tool({
    description: DESCRIPTION,
    args: {
      id: tool.schema.string().describe("The PTY session ID (e.g., pty_a1b2c3d4)"),
      cleanup: tool.schema
        .boolean()
        .optional()
        .describe("If true, removes the session and frees the buffer (default: false)"),
    },
    execute: async (args) => {
      const session = manager.get(args.id);
      if (!session) {
        throw new Error(`PTY session '${args.id}' not found. Use pty_list to see active sessions.`);
      }

      const wasRunning = session.status === "running";
      const cleanup = args.cleanup ?? false;
      const success = manager.kill(args.id, cleanup);

      if (!success) {
        throw new Error(`Failed to kill PTY session '${args.id}'.`);
      }

      const action = wasRunning ? "Killed" : "Cleaned up";
      const cleanupNote = cleanup ? " (session removed)" : " (session retained for log access)";

      return [
        `<pty_killed>`,
        `${action}: ${args.id}${cleanupNote}`,
        `Title: ${session.title}`,
        `Command: ${session.command} ${session.args.join(" ")}`,
        `Final line count: ${session.lineCount}`,
        `</pty_killed>`,
      ].join("\n");
    },
  });
}
