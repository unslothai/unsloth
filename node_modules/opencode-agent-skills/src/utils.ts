import type { PluginInput } from "@opencode-ai/plugin";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import YAML from "yaml";

/**
 * Result from finding a file in a directory.
 */
export interface FileDiscoveryResult {
  filePath: string;
  relativePath: string;
}

/**
 * Check if a file exists in a directory and return path info.
 *
 * @param directory - Directory to check
 * @param relativePath - Relative path to use in result (caller-specific)
 * @param filename - Name of file to look for (e.g., 'SKILL.md')
 * @returns Path info if file exists, null otherwise
 */
export async function findFile(
  directory: string,
  relativePath: string,
  filename: string
): Promise<FileDiscoveryResult | null> {
  const filePath = path.join(directory, filename);
  try {
    await fs.stat(filePath);
    return { filePath, relativePath };
  } catch {
    return null;
  }
}

/**
 * Parse YAML frontmatter using the yaml library with safe options.
 * Uses strict schema to prevent code execution from malicious YAML.
 * Handles all YAML 1.2 features including multi-line strings (| and >).
 */
export function parseYamlFrontmatter(text: string): Record<string, unknown> {
  try {
    const result = YAML.parse(text, {
      // Use core schema which only supports basic JSON-compatible types
      // This prevents custom tags that could execute code
      schema: "core",
      // Limit recursion depth to prevent DoS attacks
      maxAliasCount: 100,
    });
    return typeof result === "object" && result !== null
      ? (result as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/**
 * Calculate Levenshtein edit distance between two strings.
 * Used for fuzzy matching suggestions when skill/script names are not found.
 * @internal - exported for testing
 */
export function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => i || j)
  );
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i]![j] = Math.min(
        dp[i - 1]![j]! + 1,
        dp[i]![j - 1]! + 1,
        dp[i - 1]![j - 1]! + (a[i - 1] !== b[j - 1] ? 1 : 0)
      );
    }
  }

  return dp[m]![n]!;
}

/**
 * Find the closest matching string from a list of candidates.
 * Uses combined scoring: prefix match (strongest), substring match, then Levenshtein distance.
 * Returns the best match if similarity is above 0.4 threshold, otherwise null.
 * @internal - exported for testing
 */
export function findClosestMatch(input: string, candidates: string[]): string | null {
  if (candidates.length === 0) return null;

  const inputLower = input.toLowerCase();
  let bestMatch: string | null = null;
  let bestScore = 0;

  for (const candidate of candidates) {
    const candidateLower = candidate.toLowerCase();
    let score = 0;

    if (candidateLower.startsWith(inputLower)) {
      score = 0.9 + (inputLower.length / candidateLower.length) * 0.1;

      const nextChar = candidateLower[inputLower.length];
      if (nextChar && /[-_/.]/.test(nextChar)) {
        score += 0.05;
      }
    } else if (inputLower.startsWith(candidateLower)) {
      score = 0.8;
    }
    else if (candidateLower.includes(inputLower) || inputLower.includes(candidateLower)) {
      score = 0.7;
    }
    else {
      const distance = levenshtein(inputLower, candidateLower);
      const maxLength = Math.max(inputLower.length, candidateLower.length);
      score = 1 - (distance / maxLength);
    }

    if (score > bestScore) {
      bestScore = score;
      bestMatch = candidate;
    }
  }

  return bestScore >= 0.4 ? bestMatch : null;
}

/**
 * Check if a path is safely within a base directory (no escape via ..)
 */
export function isPathSafe(basePath: string, requestedPath: string): boolean {
  const resolved = path.resolve(basePath, requestedPath);
  return resolved.startsWith(basePath + path.sep) || resolved === basePath;
}

/**
 * Inject content into session via noReply + synthetic.
 * Content persists across context compaction.
 * Must pass model and agent to prevent mode/model switching.
 */
export type OpencodeClient = PluginInput["client"];

export interface SessionContext {
  model?: { providerID: string; modelID: string };
  agent?: string;
}

export async function injectSyntheticContent(
  client: OpencodeClient,
  sessionID: string,
  text: string,
  context?: SessionContext
): Promise<void> {
  await client.session.prompt({
    path: { id: sessionID },
    body: {
      noReply: true,
      model: context?.model,
      agent: context?.agent,
      parts: [{ type: "text", text, synthetic: true }],
    },
  });
}

/**
 * Get the current context (model + agent) for a session by querying messages.
 * This mirrors OpenCode's internal lastModel() logic to find the most recent
 * user message and extract its model/agent.
 *
 * Used during tool execution when we don't have direct access to the
 * current user message's context.
 */
export async function getSessionContext(
  client: OpencodeClient,
  sessionID: string,
  limit: number = 50
): Promise<SessionContext | undefined> {
  try {
    const response = await client.session.messages({
      path: { id: sessionID },
      query: { limit }
    });

    if (response.data) {
      for (const msg of response.data) {
        if (msg.info.role === "user" && "model" in msg.info && msg.info.model) {
          return {
            model: msg.info.model,
            agent: msg.info.agent
          };
        }
      }
    }
  } catch { }

  return undefined;
}
