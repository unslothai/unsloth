import { readFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

import type { PluginInput } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin/tool";

interface FileResult {
  path: string;
  content?: string;
  error?: string;
}

export function createBatchReadTool(ctx: PluginInput) {
  return tool({
    description: `Read multiple files in parallel. Much faster than reading files one at a time.
Use this when you need to read 2+ files - all reads happen concurrently via Promise.all.

Example: batch_read({paths: ["src/index.ts", "src/utils.ts", "package.json"]})

Returns content for each file, or error message if file doesn't exist.`,
    args: {
      paths: tool.schema
        .array(tool.schema.string())
        .describe("Array of file paths to read (relative to project root or absolute)"),
      maxLines: tool.schema
        .number()
        .optional()
        .describe("Optional: limit each file to first N lines (default: no limit)"),
    },
    execute: async (args) => {
      const { paths, maxLines } = args;

      if (!paths || paths.length === 0) {
        return "## batch_read Failed\n\nNo paths specified";
      }

      async function readSingleFile(filePath: string): Promise<FileResult> {
        const fullPath = isAbsolute(filePath) ? filePath : join(ctx.directory, filePath);

        try {
          let content = await readFile(fullPath, "utf-8");

          if (maxLines && maxLines > 0) {
            const lines = content.split("\n");
            if (lines.length > maxLines) {
              content =
                lines.slice(0, maxLines).join("\n") + `\n... (truncated, ${lines.length - maxLines} more lines)`;
            }
          }

          return { path: filePath, content };
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          return { path: filePath, error: msg };
        }
      }

      // Read all files in parallel
      const results = await Promise.all(paths.map(readSingleFile));

      // Format output
      const output: string[] = [`# Batch Read (${paths.length} files)\n`];

      for (const result of results) {
        if (result.error) {
          output.push(`## ${result.path}\n\n**Error**: ${result.error}\n`);
        } else {
          output.push(`## ${result.path}\n\n\`\`\`\n${result.content}\n\`\`\`\n`);
        }
      }

      return output.join("\n");
    },
  });
}
