import { tool } from "@opencode-ai/plugin";

import type { MemoryScope, MemoryStore } from "./memory";

export function MemoryList(store: MemoryStore) {
  return tool({
    description: "List available memory blocks (labels, descriptions, sizes).",
    args: {
      scope: tool.schema.enum(["all", "global", "project"]).optional(),
    },
    async execute(args) {
      // Default to "all" for list (show everything)
      const scope = (args.scope ?? "all") as MemoryScope | "all";
      const blocks = await store.listBlocks(scope);
      if (blocks.length === 0) {
        return "No memory blocks found.";
      }

      return blocks
        .map(
          (b) =>
            `${b.scope}:${b.label}\n  read_only=${b.readOnly} chars=${b.value.length}/${b.limit}\n  ${b.description}`,
        )
        .join("\n\n");
    },
  });
}

export function MemorySet(store: MemoryStore) {
  return tool({
    description: "Create or update a memory block (full overwrite).",
    args: {
      label: tool.schema.string(),
      scope: tool.schema.enum(["global", "project"]).optional(),
      value: tool.schema.string(),
      description: tool.schema.string().optional(),
      limit: tool.schema.number().int().positive().optional(),
    },
    async execute(args) {
      // Default to "project" for mutations (safer default)
      const scope = (args.scope ?? "project") as MemoryScope;
      await store.setBlock(scope, args.label, args.value, {
        description: args.description,
        limit: args.limit,
      });
      return `Updated memory block ${scope}:${args.label}.`;
    },
  });
}

export function MemoryReplace(store: MemoryStore) {
  return tool({
    description: "Replace a substring within a memory block.",
    args: {
      label: tool.schema.string(),
      scope: tool.schema.enum(["global", "project"]).optional(),
      oldText: tool.schema.string(),
      newText: tool.schema.string(),
    },
    async execute(args) {
      // Default to "project" for mutations (safer default)
      const scope = (args.scope ?? "project") as MemoryScope;
      await store.replaceInBlock(scope, args.label, args.oldText, args.newText);
      return `Updated memory block ${scope}:${args.label}.`;
    },
  });
}
