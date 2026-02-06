import type { PluginInput } from "@opencode-ai/plugin";

// Patterns that indicate excessive/unnecessary comments
const EXCESSIVE_COMMENT_PATTERNS = [
  // Obvious comments that explain what code does (not why)
  /\/\/\s*(increment|decrement|add|subtract|set|get|return|call|create|initialize|init)\s+/i,
  /\/\/\s*(the|this|a|an)\s+(following|above|below|next|previous)/i,
  // Section dividers
  /\/\/\s*[-=]{3,}/,
  /\/\/\s*#{3,}/,
  // Empty or whitespace-only comments
  /\/\/\s*$/,
  // "End of" comments
  /\/\/\s*end\s+(of|function|class|method|if|loop|for|while)/i,
];

// Patterns that are valid and should be ignored
const VALID_COMMENT_PATTERNS = [
  // TODO/FIXME/NOTE comments
  /\/\/\s*(TODO|FIXME|NOTE|HACK|XXX|BUG|WARN):/i,
  // JSDoc/TSDoc
  /^\s*\*|\/\*\*/,
  // Directive comments (eslint, prettier, ts, etc.)
  /\/\/\s*@|\/\/\s*eslint|\/\/\s*prettier|\/\/\s*ts-|\/\/\s*type:/i,
  // License headers
  /\/\/\s*(copyright|license|spdx)/i,
  // BDD-style comments (describe, it, given, when, then)
  /\/\/\s*(given|when|then|and|but|describe|it|should|expect)/i,
  // URL references
  /\/\/\s*https?:\/\//i,
  // Regex explanations (often necessary)
  /\/\/\s*regex|\/\/\s*pattern/i,
];

interface CommentIssue {
  line: number;
  comment: string;
  reason: string;
}

function analyzeComments(content: string): CommentIssue[] {
  const issues: CommentIssue[] = [];
  const lines = content.split("\n");

  let consecutiveComments = 0;
  let lastCommentLine = -2;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Check for comment lines
    if (trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith("*")) {
      // Skip valid patterns
      if (VALID_COMMENT_PATTERNS.some((p) => p.test(trimmed))) {
        continue;
      }

      // Check for excessive patterns
      for (const pattern of EXCESSIVE_COMMENT_PATTERNS) {
        if (pattern.test(trimmed)) {
          issues.push({
            line: i + 1,
            comment: trimmed.slice(0, 60) + (trimmed.length > 60 ? "..." : ""),
            reason: "Explains what, not why",
          });
          break;
        }
      }

      // Track consecutive comments (might indicate over-documentation)
      if (i === lastCommentLine + 1) {
        consecutiveComments++;
        if (consecutiveComments > 5) {
          issues.push({
            line: i + 1,
            comment: trimmed.slice(0, 60),
            reason: "Excessive consecutive comments",
          });
        }
      } else {
        consecutiveComments = 1;
      }
      lastCommentLine = i;
    }
  }

  return issues;
}

export function createCommentCheckerHook(_ctx: PluginInput) {
  return {
    // Check after file edits
    "tool.execute.after": async (
      input: { tool: string; args?: Record<string, unknown> },
      output: { output?: string },
    ) => {
      // Only check Edit tool
      if (input.tool !== "Edit" && input.tool !== "edit") return;

      const newString = input.args?.new_string as string | undefined;
      if (!newString) return;

      const issues = analyzeComments(newString);

      if (issues.length > 0) {
        const warning = `\n\n⚠️ **Comment Check**: Found ${issues.length} potentially unnecessary comment(s):\n${issues
          .slice(0, 3)
          .map((i) => `- Line ${i.line}: "${i.comment}" (${i.reason})`)
          .join(
            "\n",
          )}${issues.length > 3 ? `\n...and ${issues.length - 3} more` : ""}\n\nComments should explain WHY, not WHAT. Consider removing obvious comments.`;

        if (output.output) {
          output.output += warning;
        }
      }
    },
  };
}
