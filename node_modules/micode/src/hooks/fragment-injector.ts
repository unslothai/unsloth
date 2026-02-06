// src/hooks/fragment-injector.ts
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import type { PluginInput } from "@opencode-ai/plugin";

import type { MicodeConfig } from "../config-loader";

/**
 * Load project-level fragments from .micode/fragments.json
 * Returns empty object if file doesn't exist or is invalid
 */
export async function loadProjectFragments(projectDir: string): Promise<Record<string, string[]>> {
  const fragmentsPath = join(projectDir, ".micode", "fragments.json");

  try {
    const content = await readFile(fragmentsPath, "utf-8");
    const parsed = JSON.parse(content) as Record<string, unknown>;

    const fragments: Record<string, string[]> = {};

    for (const [agentName, fragmentList] of Object.entries(parsed)) {
      if (Array.isArray(fragmentList)) {
        const validFragments = fragmentList.filter((f): f is string => typeof f === "string" && f.trim().length > 0);
        if (validFragments.length > 0) {
          fragments[agentName] = validFragments;
        }
      }
    }

    return fragments;
  } catch {
    return {};
  }
}

/**
 * Merge global and project fragments
 * Global fragments come first, project fragments are appended
 */
export function mergeFragments(
  global: Record<string, string[]>,
  project: Record<string, string[]>,
): Record<string, string[]> {
  const allAgents = new Set([...Object.keys(global), ...Object.keys(project)]);
  const merged: Record<string, string[]> = {};

  for (const agent of allAgents) {
    const globalFragments = global[agent] ?? [];
    const projectFragments = project[agent] ?? [];
    const combined = [...globalFragments, ...projectFragments];

    if (combined.length > 0) {
      merged[agent] = combined;
    }
  }

  return merged;
}

/**
 * Format fragments as an XML block for injection into system prompt
 */
export function formatFragmentsBlock(fragments: string[]): string {
  if (fragments.length === 0) {
    return "";
  }

  const bullets = fragments.map((f) => `- ${f}`).join("\n");
  return `<user-instructions>\n${bullets}\n</user-instructions>\n\n`;
}

/**
 * Simple Levenshtein distance for typo detection
 */
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1, // insertion
          matrix[i - 1][j] + 1, // deletion
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

/**
 * Find closest matching agent name
 */
function findClosestAgent(unknown: string, knownAgents: Set<string>): string | null {
  let closest: string | null = null;
  let minDistance = Infinity;

  for (const known of knownAgents) {
    const distance = levenshteinDistance(unknown, known);
    // Only suggest if distance is reasonable (less than half the length)
    if (distance < minDistance && distance <= Math.ceil(known.length / 2)) {
      minDistance = distance;
      closest = known;
    }
  }

  return closest;
}

/**
 * Generate warnings for unknown agent names in fragments config
 */
export function warnUnknownAgents(fragmentAgents: string[], knownAgents: Set<string>): string[] {
  const warnings: string[] = [];

  for (const agent of fragmentAgents) {
    if (!knownAgents.has(agent)) {
      const closest = findClosestAgent(agent, knownAgents);
      if (closest) {
        warnings.push(`[micode] Unknown agent "${agent}" in fragments config. Did you mean "${closest}"?`);
      } else {
        warnings.push(`[micode] Unknown agent "${agent}" in fragments config.`);
      }
    }
  }

  return warnings;
}

/**
 * Create fragment injector hook
 * Injects user-defined fragments at the beginning of agent system prompts
 */
export function createFragmentInjectorHook(ctx: PluginInput, globalConfig: MicodeConfig | null) {
  // Cache for project fragments (loaded once per session)
  let projectFragmentsCache: Record<string, string[]> | null = null;

  async function getProjectFragments(): Promise<Record<string, string[]>> {
    if (projectFragmentsCache === null) {
      projectFragmentsCache = await loadProjectFragments(ctx.directory);
    }
    return projectFragmentsCache;
  }

  return {
    "chat.params": async (
      _input: { sessionID: string },
      output: { options?: Record<string, unknown>; system?: string },
    ) => {
      const agent = output.options?.agent as string | undefined;
      if (!agent) return;

      const globalFragments = globalConfig?.fragments ?? {};
      const projectFragments = await getProjectFragments();
      const mergedFragments = mergeFragments(globalFragments, projectFragments);

      const agentFragments = mergedFragments[agent];
      if (!agentFragments || agentFragments.length === 0) return;

      const fragmentBlock = formatFragmentsBlock(agentFragments);

      if (output.system) {
        output.system = fragmentBlock + output.system;
      } else {
        output.system = fragmentBlock;
      }
    },
  };
}
