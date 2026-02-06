// src/tools/pty/tools/read.ts
import { tool } from "@opencode-ai/plugin/tool";
import type { PTYManager } from "../manager";

const DESCRIPTION = `Reads output from a PTY session's buffer.

The PTY maintains a rolling buffer of output lines. Use offset and limit to paginate through the output, similar to reading a file.

Usage:
- \`id\`: The PTY session ID (from pty_spawn or pty_list)
- \`offset\`: Line number to start reading from (0-based, defaults to 0)
- \`limit\`: Number of lines to read (defaults to 500)
- \`pattern\`: Regex pattern to filter lines (optional)
- \`ignoreCase\`: Case-insensitive pattern matching (default: false)

Returns:
- Numbered lines of output (similar to cat -n format)
- Total line count in the buffer
- Indicator if more lines are available

The buffer stores up to PTY_MAX_BUFFER_LINES (default: 50000) lines. Older lines are discarded when the limit is reached.

Pattern Filtering:
- When \`pattern\` is set, lines are FILTERED FIRST using the regex, then offset/limit apply to the MATCHES
- Original line numbers are preserved so you can see where matches occurred in the buffer
- Supports full regex syntax (e.g., "error", "ERROR|WARN", "failed.*connection", etc.)
- If the pattern is invalid, an error message is returned explaining the issue
- If no lines match the pattern, a clear message indicates zero matches

Tips:
- To see the latest output, use a high offset or omit offset to read from the start
- To tail recent output, calculate offset as (totalLines - N) where N is how many recent lines you want
- Lines longer than 2000 characters are truncated
- Empty output may mean the process hasn't produced output yet

Examples:
- Read first 100 lines: offset=0, limit=100
- Read lines 500-600: offset=500, limit=100
- Read all available: omit both parameters
- Find errors: pattern="error", ignoreCase=true
- Find specific log levels: pattern="ERROR|WARN|FATAL"
- First 10 matches only: pattern="error", limit=10`;

const DEFAULT_LIMIT = 500;
const MAX_LINE_LENGTH = 2000;

export function createPtyReadTool(manager: PTYManager) {
  return tool({
    description: DESCRIPTION,
    args: {
      id: tool.schema.string().describe("The PTY session ID (e.g., pty_a1b2c3d4)"),
      offset: tool.schema.number().optional().describe("Line number to start reading from (0-based, defaults to 0)"),
      limit: tool.schema.number().optional().describe("Number of lines to read (defaults to 500)"),
      pattern: tool.schema.string().optional().describe("Regex pattern to filter lines"),
      ignoreCase: tool.schema.boolean().optional().describe("Case-insensitive pattern matching (default: false)"),
    },
    execute: async (args) => {
      const session = manager.get(args.id);
      if (!session) {
        throw new Error(`PTY session '${args.id}' not found. Use pty_list to see active sessions.`);
      }

      const offset = Math.max(0, args.offset ?? 0);
      const limit = args.limit ?? DEFAULT_LIMIT;

      if (args.pattern) {
        let regex: RegExp;
        try {
          regex = new RegExp(args.pattern, args.ignoreCase ? "i" : "");
        } catch (e) {
          const error = e instanceof Error ? e.message : String(e);
          throw new Error(`Invalid regex pattern '${args.pattern}': ${error}`);
        }

        const result = manager.search(args.id, regex, offset, limit);
        if (!result) {
          throw new Error(`PTY session '${args.id}' not found.`);
        }

        if (result.matches.length === 0) {
          return [
            `<pty_output id="${args.id}" status="${session.status}" pattern="${args.pattern}">`,
            `No lines matched the pattern '${args.pattern}'.`,
            `Total lines in buffer: ${result.totalLines}`,
            `</pty_output>`,
          ].join("\n");
        }

        const formattedLines = result.matches.map((match) => {
          const lineNum = match.lineNumber.toString().padStart(5, "0");
          const truncatedLine =
            match.text.length > MAX_LINE_LENGTH ? `${match.text.slice(0, MAX_LINE_LENGTH)}...` : match.text;
          return `${lineNum}| ${truncatedLine}`;
        });

        const output = [
          `<pty_output id="${args.id}" status="${session.status}" pattern="${args.pattern}">`,
          ...formattedLines,
          "",
        ];

        if (result.hasMore) {
          output.push(
            `(${result.matches.length} of ${result.totalMatches} matches shown. Use offset=${offset + result.matches.length} to see more.)`,
          );
        } else {
          output.push(
            `(${result.totalMatches} match${result.totalMatches === 1 ? "" : "es"} from ${result.totalLines} total lines)`,
          );
        }
        output.push(`</pty_output>`);

        return output.join("\n");
      }

      const result = manager.read(args.id, offset, limit);
      if (!result) {
        throw new Error(`PTY session '${args.id}' not found.`);
      }

      if (result.lines.length === 0) {
        return [
          `<pty_output id="${args.id}" status="${session.status}">`,
          `(No output available - buffer is empty)`,
          `Total lines: ${result.totalLines}`,
          `</pty_output>`,
        ].join("\n");
      }

      const formattedLines = result.lines.map((line, index) => {
        const lineNum = (result.offset + index + 1).toString().padStart(5, "0");
        const truncatedLine = line.length > MAX_LINE_LENGTH ? `${line.slice(0, MAX_LINE_LENGTH)}...` : line;
        return `${lineNum}| ${truncatedLine}`;
      });

      const output = [`<pty_output id="${args.id}" status="${session.status}">`, ...formattedLines];

      if (result.hasMore) {
        output.push("");
        output.push(
          `(Buffer has more lines. Use offset=${result.offset + result.lines.length} to read beyond line ${result.offset + result.lines.length})`,
        );
      } else {
        output.push("");
        output.push(`(End of buffer - total ${result.totalLines} lines)`);
      }
      output.push(`</pty_output>`);

      return output.join("\n");
    },
  });
}
