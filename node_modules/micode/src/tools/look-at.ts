import { tool } from "@opencode-ai/plugin/tool";
import { readFileSync, statSync } from "node:fs";
import { extname, basename } from "node:path";

// File size threshold for triggering extraction (100KB)
const LARGE_FILE_THRESHOLD = 100 * 1024;

// Max lines to return without extraction
const MAX_LINES_WITHOUT_EXTRACT = 200;

// Supported file types for smart extraction
const EXTRACTABLE_EXTENSIONS = [
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".py",
  ".go",
  ".rs",
  ".java",
  ".md",
  ".json",
  ".yaml",
  ".yml",
];

function extractStructure(content: string, ext: string): string {
  const lines = content.split("\n");

  switch (ext) {
    case ".ts":
    case ".tsx":
    case ".js":
    case ".jsx":
      return extractTypeScriptStructure(lines);
    case ".py":
      return extractPythonStructure(lines);
    case ".go":
      return extractGoStructure(lines);
    case ".md":
      return extractMarkdownStructure(lines);
    case ".json":
      return extractJsonStructure(content);
    case ".yaml":
    case ".yml":
      return extractYamlStructure(lines);
    default:
      return extractGenericStructure(lines);
  }
}

function extractTypeScriptStructure(lines: string[]): string {
  const output: string[] = ["## Structure\n"];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Capture exports, classes, interfaces, types, functions
    if (
      trimmed.startsWith("export ") ||
      trimmed.startsWith("class ") ||
      trimmed.startsWith("interface ") ||
      trimmed.startsWith("type ") ||
      trimmed.startsWith("function ") ||
      trimmed.startsWith("const ") ||
      trimmed.startsWith("async function ")
    ) {
      // Get the signature (first line only for multi-line)
      const signature = trimmed.length > 80 ? `${trimmed.slice(0, 80)}...` : trimmed;
      output.push(`Line ${i + 1}: ${signature}`);
    }
  }

  return output.join("\n");
}

function extractPythonStructure(lines: string[]): string {
  const output: string[] = ["## Structure\n"];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Capture classes, functions, decorators
    if (
      trimmed.startsWith("class ") ||
      trimmed.startsWith("def ") ||
      trimmed.startsWith("async def ") ||
      trimmed.startsWith("@")
    ) {
      const signature = trimmed.length > 80 ? `${trimmed.slice(0, 80)}...` : trimmed;
      output.push(`Line ${i + 1}: ${signature}`);
    }
  }

  return output.join("\n");
}

function extractGoStructure(lines: string[]): string {
  const output: string[] = ["## Structure\n"];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Capture types, functions, methods
    if (trimmed.startsWith("type ") || trimmed.startsWith("func ") || trimmed.startsWith("package ")) {
      const signature = trimmed.length > 80 ? `${trimmed.slice(0, 80)}...` : trimmed;
      output.push(`Line ${i + 1}: ${signature}`);
    }
  }

  return output.join("\n");
}

function extractMarkdownStructure(lines: string[]): string {
  const output: string[] = ["## Outline\n"];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Capture headings
    if (line.startsWith("#")) {
      output.push(`Line ${i + 1}: ${line}`);
    }
  }

  return output.join("\n");
}

function extractJsonStructure(content: string): string {
  try {
    const obj = JSON.parse(content);
    const keys = Object.keys(obj);
    return `## Top-level keys (${keys.length})\n\n${keys.slice(0, 50).join(", ")}${keys.length > 50 ? "..." : ""}`;
  } catch {
    return "## Invalid JSON";
  }
}

function extractYamlStructure(lines: string[]): string {
  const output: string[] = ["## Top-level keys\n"];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Top-level keys (no indentation)
    if (line.match(/^[a-zA-Z_][a-zA-Z0-9_]*:/)) {
      output.push(`Line ${i + 1}: ${line}`);
    }
  }

  return output.join("\n");
}

function extractGenericStructure(lines: string[]): string {
  // For unknown files, just show first/last lines and line count
  const total = lines.length;
  const preview = lines.slice(0, 10).join("\n");
  const tail = lines.slice(-5).join("\n");

  return `## File Preview (${total} lines)\n\n### First 10 lines:\n${preview}\n\n### Last 5 lines:\n${tail}`;
}

export const look_at = tool({
  description: `Extract key information from a file to save context tokens.
For large files, returns structure/outline instead of full content.
Use when you need to understand a file without loading all content.
Ideal for: large files, getting file structure, quick overview.`,
  args: {
    filePath: tool.schema.string().describe("Path to the file"),
    extract: tool.schema
      .string()
      .optional()
      .describe("What to extract: 'structure', 'imports', 'exports', 'all' (default: auto)"),
  },
  execute: async (args) => {
    try {
      const stats = statSync(args.filePath);
      const ext = extname(args.filePath).toLowerCase();
      const name = basename(args.filePath);

      // Read file
      const content = readFileSync(args.filePath, "utf-8");
      const lines = content.split("\n");

      // For small files, return full content
      if (stats.size < LARGE_FILE_THRESHOLD && lines.length <= MAX_LINES_WITHOUT_EXTRACT) {
        return `## ${name} (${lines.length} lines)\n\n${content}`;
      }

      // For large files, extract structure
      let output = `## ${name}\n`;
      output += `**Size**: ${Math.round(stats.size / 1024)}KB | **Lines**: ${lines.length}\n\n`;

      if (EXTRACTABLE_EXTENSIONS.includes(ext)) {
        output += extractStructure(content, ext);
      } else {
        output += extractGenericStructure(lines);
      }

      output += `\n\n---\n*Use Read tool with line offset/limit for specific sections*`;

      return output;
    } catch (e) {
      return `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  },
});
