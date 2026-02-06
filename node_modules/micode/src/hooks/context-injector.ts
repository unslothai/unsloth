import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import type { PluginInput } from "@opencode-ai/plugin";

import { config } from "../utils/config";

// Tools that trigger directory-aware context injection
const FILE_ACCESS_TOOLS = ["Read", "read", "Edit", "edit"];

// Cache for file contents
interface ContextCache {
  rootContent: Map<string, string>;
  directoryContent: Map<string, Map<string, string>>; // path -> filename -> content
  lastRootCheck: number;
}

export function createContextInjectorHook(ctx: PluginInput) {
  const cache: ContextCache = {
    rootContent: new Map(),
    directoryContent: new Map(),
    lastRootCheck: 0,
  };

  async function loadRootContextFiles(): Promise<Map<string, string>> {
    const now = Date.now();

    if (now - cache.lastRootCheck < config.limits.contextCacheTtlMs && cache.rootContent.size > 0) {
      return cache.rootContent;
    }

    cache.rootContent.clear();
    cache.lastRootCheck = now;

    for (const filename of config.paths.rootContextFiles) {
      try {
        const filepath = join(ctx.directory, filename);
        const content = await readFile(filepath, "utf-8");
        if (content.trim()) {
          cache.rootContent.set(filename, content);
        }
      } catch {
        // File doesn't exist - skip
      }
    }

    return cache.rootContent;
  }

  async function walkUpForContextFiles(filePath: string): Promise<Map<string, string>> {
    const absPath = resolve(filePath);
    const projectRoot = resolve(ctx.directory);

    // Check cache
    const cacheKey = dirname(absPath);
    if (cache.directoryContent.has(cacheKey)) {
      return cache.directoryContent.get(cacheKey)!;
    }

    const collected = new Map<string, string>();
    let currentDir = dirname(absPath);

    // Walk up from file directory to project root
    while (currentDir.startsWith(projectRoot) || currentDir === projectRoot) {
      for (const filename of config.paths.dirContextFiles) {
        const contextPath = join(currentDir, filename);
        const relPath = currentDir.replace(projectRoot, "").replace(/^\//, "") || ".";
        const key = `${relPath}/${filename}`;

        // Skip if already have this file from a closer directory
        if (!collected.has(key)) {
          try {
            const content = await readFile(contextPath, "utf-8");
            if (content.trim()) {
              collected.set(key, content);
            }
          } catch {
            // File doesn't exist - skip
          }
        }
      }

      // Stop at project root
      if (currentDir === projectRoot) break;

      // Move up
      const parent = dirname(currentDir);
      if (parent === currentDir) break; // Reached filesystem root
      currentDir = parent;
    }

    // Cache for this directory
    cache.directoryContent.set(cacheKey, collected);

    // Limit cache size
    if (cache.directoryContent.size > config.limits.contextCacheMaxSize) {
      const firstKey = cache.directoryContent.keys().next().value;
      if (firstKey) cache.directoryContent.delete(firstKey);
    }

    return collected;
  }

  function formatContextBlock(files: Map<string, string>, label: string): string {
    if (files.size === 0) return "";

    const blocks: string[] = [];

    for (const [filename, content] of files) {
      blocks.push(`<context file="${filename}">\n${content}\n</context>`);
    }

    return `\n<${label}>\n${blocks.join("\n\n")}\n</${label}>\n`;
  }

  return {
    // Inject project root context into every chat
    "chat.params": async (
      _input: { sessionID: string },
      output: { options?: Record<string, unknown>; system?: string },
    ) => {
      const files = await loadRootContextFiles();
      if (files.size === 0) return;

      const contextBlock = formatContextBlock(files, "project-context");

      if (output.system) {
        output.system = output.system + contextBlock;
      } else {
        output.system = contextBlock;
      }
    },

    // Inject directory-specific context when reading/editing files
    "tool.execute.after": async (
      input: { tool: string; args?: Record<string, unknown> },
      output: { output?: string },
    ) => {
      if (!FILE_ACCESS_TOOLS.includes(input.tool)) return;

      const filePath = input.args?.filePath as string | undefined;
      if (!filePath) return;

      try {
        const directoryFiles = await walkUpForContextFiles(filePath);
        if (directoryFiles.size === 0) return;

        const contextBlock = formatContextBlock(directoryFiles, "directory-context");

        if (output.output) {
          output.output = output.output + contextBlock;
        }
      } catch {
        // Ignore errors in context injection
      }
    },
  };
}
