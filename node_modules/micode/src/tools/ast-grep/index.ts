import { spawn, which } from "bun";
import { tool } from "@opencode-ai/plugin/tool";

/**
 * Check if ast-grep CLI (sg) is available on the system.
 * Returns installation instructions if not found.
 */
export async function checkAstGrepAvailable(): Promise<{ available: boolean; message?: string }> {
  const sgPath = which("sg");
  if (sgPath) {
    return { available: true };
  }
  return {
    available: false,
    message:
      "ast-grep CLI (sg) not found. AST-aware search/replace will not work.\n" +
      "Install with one of:\n" +
      "  npm install -g @ast-grep/cli\n" +
      "  cargo install ast-grep --locked\n" +
      "  brew install ast-grep",
  };
}

const LANGUAGES = [
  "c",
  "cpp",
  "csharp",
  "css",
  "dart",
  "elixir",
  "go",
  "haskell",
  "html",
  "java",
  "javascript",
  "json",
  "kotlin",
  "lua",
  "php",
  "python",
  "ruby",
  "rust",
  "scala",
  "sql",
  "swift",
  "tsx",
  "typescript",
  "yaml",
] as const;

interface Match {
  file: string;
  range: { start: { line: number; column: number }; end: { line: number; column: number } };
  text: string;
  replacement?: string;
}

async function runSg(args: string[]): Promise<{ matches: Match[]; error?: string }> {
  try {
    const proc = spawn(["sg", ...args], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);

    if (exitCode !== 0 && !stdout.trim()) {
      if (stderr.includes("No files found")) {
        return { matches: [] };
      }
      return { matches: [], error: stderr.trim() || `Exit code ${exitCode}` };
    }

    if (!stdout.trim()) return { matches: [] };

    try {
      const matches = JSON.parse(stdout) as Match[];
      return { matches };
    } catch {
      return { matches: [], error: "Failed to parse output" };
    }
  } catch (e) {
    const err = e as Error;
    if (err.message?.includes("ENOENT")) {
      return {
        matches: [],
        error:
          "ast-grep CLI not found. Install with:\n" +
          "  npm install -g @ast-grep/cli\n" +
          "  cargo install ast-grep --locked\n" +
          "  brew install ast-grep",
      };
    }
    return { matches: [], error: err.message };
  }
}

function formatMatches(matches: Match[], isDryRun = false): string {
  if (matches.length === 0) return "No matches found";

  const MAX = 100;
  const truncated = matches.length > MAX;
  const shown = matches.slice(0, MAX);

  const lines = shown.map((m) => {
    const loc = `${m.file}:${m.range.start.line}:${m.range.start.column}`;
    const text = m.text.length > 100 ? `${m.text.slice(0, 100)}...` : m.text;
    if (isDryRun && m.replacement) {
      return `${loc}\n  - ${text}\n  + ${m.replacement}`;
    }
    return `${loc}: ${text}`;
  });

  if (truncated) {
    lines.unshift(`Found ${matches.length} matches (showing first ${MAX}):`);
  }

  return lines.join("\n");
}

export const ast_grep_search = tool({
  description:
    "Search code patterns using AST-aware matching. " +
    "Use meta-variables: $VAR (single node), $$$ (multiple nodes). " +
    "Patterns must be complete AST nodes. " +
    "Examples: 'console.log($MSG)', 'def $FUNC($$$):', 'async function $NAME($$$)'",
  args: {
    pattern: tool.schema.string().describe("AST pattern with meta-variables"),
    lang: tool.schema.enum(LANGUAGES).describe("Target language"),
    paths: tool.schema.array(tool.schema.string()).optional().describe("Paths to search"),
  },
  execute: async (args) => {
    const sgArgs = ["run", "-p", args.pattern, "--lang", args.lang, "--json=compact"];
    if (args.paths?.length) {
      sgArgs.push(...args.paths);
    } else {
      sgArgs.push(".");
    }

    const result = await runSg(sgArgs);
    if (result.error) return `Error: ${result.error}`;
    return formatMatches(result.matches);
  },
});

export const ast_grep_replace = tool({
  description:
    "Replace code patterns with AST-aware rewriting. " +
    "Dry-run by default. Use meta-variables in rewrite to preserve matched content. " +
    "Example: pattern='console.log($MSG)' rewrite='logger.info($MSG)'",
  args: {
    pattern: tool.schema.string().describe("AST pattern to match"),
    rewrite: tool.schema.string().describe("Replacement pattern"),
    lang: tool.schema.enum(LANGUAGES).describe("Target language"),
    paths: tool.schema.array(tool.schema.string()).optional().describe("Paths to search"),
    apply: tool.schema.boolean().optional().describe("Apply changes (default: false, dry-run)"),
  },
  execute: async (args) => {
    const sgArgs = ["run", "-p", args.pattern, "-r", args.rewrite, "--lang", args.lang, "--json=compact"];

    if (args.apply) {
      sgArgs.push("--update-all");
    }

    if (args.paths?.length) {
      sgArgs.push(...args.paths);
    } else {
      sgArgs.push(".");
    }

    const result = await runSg(sgArgs);
    if (result.error) return `Error: ${result.error}`;

    const isDryRun = !args.apply;
    const output = formatMatches(result.matches, isDryRun);

    if (isDryRun && result.matches.length > 0) {
      return `${output}\n\n(Dry run - use apply=true to apply changes)`;
    }
    if (args.apply && result.matches.length > 0) {
      return `Applied ${result.matches.length} replacements:\n${output}`;
    }
    return output;
  },
});
