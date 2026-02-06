// src/tools/pty/tools/list.ts
import { tool } from "@opencode-ai/plugin/tool";
import type { PTYManager } from "../manager";

const DESCRIPTION = `Lists all PTY sessions (active and exited).

Use this tool to:
- See all running and exited PTY sessions
- Get session IDs for use with other pty_* tools
- Check the status and output line count of each session
- Monitor which processes are still running

Returns for each session:
- \`id\`: Unique identifier for use with other tools
- \`title\`: Human-readable name
- \`command\`: The command that was executed
- \`status\`: Current status (running, exited, killed)
- \`exitCode\`: Exit code (if exited/killed)
- \`pid\`: Process ID
- \`lineCount\`: Number of lines in the output buffer
- \`createdAt\`: When the session was created

Tips:
- Use the session ID with pty_read, pty_write, or pty_kill
- Sessions remain in the list after exit until explicitly cleaned up with pty_kill
- This allows you to compare output from multiple sessions`;

export function createPtyListTool(manager: PTYManager) {
  return tool({
    description: DESCRIPTION,
    args: {},
    execute: async () => {
      const sessions = manager.list();

      if (sessions.length === 0) {
        return "<pty_list>\nNo active PTY sessions.\n</pty_list>";
      }

      const lines = ["<pty_list>"];
      for (const session of sessions) {
        const exitInfo = session.exitCode !== undefined ? ` (exit: ${session.exitCode})` : "";
        lines.push(`[${session.id}] ${session.title}`);
        lines.push(`  Command: ${session.command} ${session.args.join(" ")}`);
        lines.push(`  Status: ${session.status}${exitInfo}`);
        lines.push(`  PID: ${session.pid} | Lines: ${session.lineCount} | Workdir: ${session.workdir}`);
        lines.push(`  Created: ${session.createdAt.toISOString()}`);
        lines.push("");
      }
      lines.push(`Total: ${sessions.length} session(s)`);
      lines.push("</pty_list>");

      return lines.join("\n");
    },
  });
}
