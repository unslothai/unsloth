// src/tools/mindmodel-lookup.ts
import type { PluginInput } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin/tool";

import { formatExamplesForInjection, type LoadedMindmodel, loadExamples, loadMindmodel } from "../mindmodel";
import { log } from "../utils/logger";

let cachedMindmodel: LoadedMindmodel | null | undefined;

async function getMindmodel(directory: string): Promise<LoadedMindmodel | null> {
  if (cachedMindmodel === undefined) {
    cachedMindmodel = await loadMindmodel(directory);
  }
  return cachedMindmodel;
}

// Simple keyword-based category matching (no LLM needed)
export function matchCategories(query: string, manifest: LoadedMindmodel["manifest"]): string[] {
  const queryLower = query.toLowerCase();
  const matched: string[] = [];

  for (const category of manifest.categories) {
    // Extract keywords from path and description
    const pathParts = category.path.toLowerCase().replace(".md", "").split("/");
    const descLower = (category.description || "").toLowerCase();

    // Check if any keyword matches
    const keywords = [...pathParts, ...descLower.split(/\s+/)];
    for (const keyword of keywords) {
      if (keyword.length > 2 && queryLower.includes(keyword)) {
        matched.push(category.path);
        break;
      }
    }
  }

  return matched;
}

export function createMindmodelLookupTool(ctx: PluginInput) {
  const mindmodel_lookup = tool({
    description: `Look up coding patterns and examples from the project's .mindmodel/ directory.
Call this tool when you need to understand how to implement something in this codebase.
Provide a brief description of what you're trying to do (e.g., "create a form component", "add error handling", "write a test").
Returns relevant code examples and patterns to follow.`,
    args: {
      query: tool.schema
        .string()
        .describe("What you're trying to implement (e.g., 'create a button component', 'add form validation')"),
    },
    execute: async ({ query }) => {
      try {
        const mindmodel = await getMindmodel(ctx.directory);
        if (!mindmodel) {
          return "No .mindmodel/ directory found in this project. Proceed without specific patterns.";
        }

        log.info("mindmodel", `Looking up patterns for: "${query.slice(0, 100)}..."`);

        // Match categories using keywords
        const categories = matchCategories(query, mindmodel.manifest);

        if (categories.length === 0) {
          return "No specific patterns found for this task. Proceed using general best practices.";
        }

        log.info("mindmodel", `Matched categories: ${categories.join(", ")}`);

        // Load examples
        const examples = await loadExamples(mindmodel, categories);
        if (examples.length === 0) {
          return "Categories matched but no examples found. Proceed using general best practices.";
        }

        const formatted = formatExamplesForInjection(examples);
        log.info("mindmodel", `Returning ${examples.length} examples`);

        return formatted;
      } catch (error) {
        log.warn("mindmodel", `Lookup failed: ${error instanceof Error ? error.message : "unknown"}`);
        return "Failed to load patterns. Proceed using general best practices.";
      }
    },
  });

  return { mindmodel_lookup };
}
