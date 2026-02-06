// src/tools/pty/tools/spawn.ts
import { tool } from "@opencode-ai/plugin/tool";
import type { PTYManager } from "../manager";

const DESCRIPTION = `Spawns a new interactive PTY (pseudo-terminal) session that runs in the background.

Unlike the built-in bash tool which runs commands synchronously and waits for completion, PTY sessions persist and allow you to:
- Run long-running processes (dev servers, watch modes, etc.)
- Send interactive input (including Ctrl+C, arrow keys, etc.)
- Read output at any time
- Manage multiple concurrent terminal sessions

Usage:
- The \`command\` parameter is required (e.g., "npm", "python", "bash")
- Use \`args\` to pass arguments to the command (e.g., ["run", "dev"])
- Use \`workdir\` to set the working directory (defaults to project root)
- Use \`env\` to set additional environment variables
- Use \`title\` to give the session a human-readable name
- Use \`description\` for a clear, concise 5-10 word description (optional)

Returns the session info including:
- \`id\`: Unique identifier (pty_XXXXXXXX) for use with other pty_* tools
- \`pid\`: Process ID
- \`status\`: Current status ("running")

After spawning, use:
- \`pty_write\` to send input to the PTY
- \`pty_read\` to read output from the PTY
- \`pty_list\` to see all active PTY sessions
- \`pty_kill\` to terminate the PTY

Examples:
- Start a dev server: command="npm", args=["run", "dev"], title="Dev Server"
- Start a Python REPL: command="python3", title="Python REPL"
- Run tests in watch mode: command="npm", args=["test", "--", "--watch"]`;

export function createPtySpawnTool(manager: PTYManager) {
  return tool({
    description: DESCRIPTION,
    args: {
      command: tool.schema.string().describe("The command/executable to run"),
      args: tool.schema.array(tool.schema.string()).optional().describe("Arguments to pass to the command"),
      workdir: tool.schema.string().optional().describe("Working directory for the PTY session"),
      env: tool.schema
        .record(tool.schema.string(), tool.schema.string())
        .optional()
        .describe("Additional environment variables"),
      title: tool.schema.string().optional().describe("Human-readable title for the session"),
      description: tool.schema
        .string()
        .optional()
        .describe("Clear, concise description of what this PTY session is for in 5-10 words"),
    },
    execute: async (args, ctx) => {
      const info = manager.spawn({
        command: args.command,
        args: args.args,
        workdir: args.workdir,
        env: args.env,
        title: args.title,
        parentSessionId: ctx.sessionID,
      });

      const output = [
        `<pty_spawned>`,
        `ID: ${info.id}`,
        `Title: ${info.title}`,
        `Command: ${info.command} ${info.args.join(" ")}`,
        `Workdir: ${info.workdir}`,
        `PID: ${info.pid}`,
        `Status: ${info.status}`,
        `</pty_spawned>`,
      ].join("\n");

      return output;
    },
  });
}
