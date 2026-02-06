// src/hooks/mindmodel-injector.ts
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import type { PluginInput } from "@opencode-ai/plugin";

import { formatExamplesForInjection, type LoadedMindmodel, loadExamples, loadMindmodel } from "../mindmodel";
import { matchCategories } from "../tools/mindmodel-lookup";
import { config } from "../utils/config";

interface MessagePart {
  type: string;
  text?: string;
}

interface MessageWithParts {
  info: { role: string };
  parts: MessagePart[];
}

// Simple hash function for task strings
function hashTask(task: string): string {
  let hash = 0;
  for (let i = 0; i < task.length; i++) {
    const char = task.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash.toString(36);
}

// Simple LRU cache for matched tasks
class LRUCache<V> {
  private cache = new Map<string, V>();
  constructor(private maxSize: number) {}

  get(key: string): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: string, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Delete oldest (first) entry
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }
}

export function createMindmodelInjectorHook(ctx: PluginInput) {
  let cachedMindmodel: LoadedMindmodel | null | undefined;
  let cachedSystemMd: string | null | undefined;

  // Pending injection content (shared across hooks for current request)
  let pendingInjection: string | null = null;

  // LRU cache for matched tasks (uses hashed keys)
  const matchedTasks = new LRUCache<string>(2000);

  async function getMindmodel(): Promise<LoadedMindmodel | null> {
    if (cachedMindmodel === undefined) {
      cachedMindmodel = await loadMindmodel(ctx.directory);
    }
    return cachedMindmodel;
  }

  async function getSystemMd(): Promise<string | null> {
    if (cachedSystemMd === undefined) {
      try {
        const systemPath = join(ctx.directory, config.paths.mindmodelDir, config.paths.mindmodelSystem);
        cachedSystemMd = await readFile(systemPath, "utf-8");
      } catch {
        cachedSystemMd = null;
      }
    }
    return cachedSystemMd;
  }

  function extractTaskFromMessages(messages: MessageWithParts[]): string {
    // Get the last user message
    const lastUserMessage = [...messages].reverse().find((m) => m.info.role === "user");
    if (!lastUserMessage) return "";

    // Extract text from parts
    return lastUserMessage.parts
      .filter((p) => p.type === "text" && p.text)
      .map((p) => p.text)
      .join(" ");
  }

  return {
    // Hook 1: Extract task from messages and prepare injection
    "experimental.chat.messages.transform": async (
      _input: Record<string, unknown>,
      output: { messages: MessageWithParts[] },
    ) => {
      try {
        const mindmodel = await getMindmodel();
        if (!mindmodel) {
          return;
        }

        const task = extractTaskFromMessages(output.messages);
        if (!task) {
          return;
        }

        // Check cache first
        const taskHash = hashTask(task);
        const cachedInjection = matchedTasks.get(taskHash);
        if (cachedInjection !== undefined) {
          pendingInjection = cachedInjection || null;
          return;
        }

        // Match categories using keywords (instant, no LLM)
        const categories = matchCategories(task, mindmodel.manifest);

        if (categories.length === 0) {
          matchedTasks.set(taskHash, ""); // Cache empty result
          return;
        }

        // Load and format examples
        const examples = await loadExamples(mindmodel, categories);
        if (examples.length === 0) {
          matchedTasks.set(taskHash, ""); // Cache empty result
          return;
        }

        const formatted = formatExamplesForInjection(examples);

        // Store for the system transform hook and cache for future requests
        pendingInjection = formatted;
        matchedTasks.set(taskHash, formatted);
      } catch {
        // Silently ignore errors - don't break the main flow
      }
    },

    // Hook 2: Inject into system prompt
    "experimental.chat.system.transform": async (_input: { sessionID: string }, output: { system: string[] }) => {
      // Always inject system.md as base context
      const systemMd = await getSystemMd();
      if (systemMd) {
        output.system.unshift(`<mindmodel-constraints>\n${systemMd}\n</mindmodel-constraints>`);
      }

      // Add keyword-matched patterns if any
      if (pendingInjection) {
        const injection = pendingInjection;
        pendingInjection = null;
        output.system.unshift(injection);
      }
    },
  };
}
