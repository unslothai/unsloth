// src/mindmodel/review.ts
export interface Violation {
  file: string;
  line?: number;
  rule: string;
  constraint_file: string;
  found: string;
  expected: string;
}

export interface ReviewResult {
  status: "PASS" | "BLOCKED";
  violations: Violation[];
  summary: string;
}

export function parseReviewResponse(response: string): ReviewResult {
  // Extract JSON from markdown code blocks if present
  const jsonMatch = response.match(/```(?:json)?\s*([\s\S]*?)```/);
  const jsonStr = jsonMatch ? jsonMatch[1].trim() : response.trim();

  try {
    const parsed = JSON.parse(jsonStr);
    return {
      status: parsed.status === "PASS" ? "PASS" : "BLOCKED",
      violations: parsed.violations || [],
      summary: parsed.summary || "",
    };
  } catch {
    // If JSON parsing fails, assume PASS to avoid false blocks
    return {
      status: "PASS",
      violations: [],
      summary: "Failed to parse review response",
    };
  }
}

export function formatViolationsForRetry(violations: Violation[]): string {
  if (violations.length === 0) return "";

  const lines = ["The previous attempt had constraint violations:", ""];

  for (const v of violations) {
    lines.push(`- ${v.file}${v.line ? `:${v.line}` : ""}: ${v.rule}`);
    lines.push(`  Found: ${v.found}`);
    lines.push(`  Expected: ${v.expected}`);
    lines.push(`  See: ${v.constraint_file}`);
    lines.push("");
  }

  lines.push("Please fix these issues in your next attempt.");

  return lines.join("\n");
}

export function formatViolationsForUser(violations: Violation[]): string {
  if (violations.length === 0) return "";

  const lines = ["Blocked: This code violates project constraints:", ""];

  for (const v of violations) {
    lines.push(`- ${v.rule} (see ${v.constraint_file})`);
    lines.push(`  File: ${v.file}${v.line ? `:${v.line}` : ""}`);
  }

  return lines.join("\n");
}
