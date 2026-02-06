import type { Plugin } from "@opencode-ai/plugin";

import { createMemoryStore } from "./memory";
import { renderMemoryBlocks } from "./prompt";
import { MemoryList, MemoryReplace, MemorySet } from "./tools";

export const MemoryPlugin: Plugin = async ({ directory }) => {
  const store = createMemoryStore(directory);
  await store.ensureSeed();

  return {
    "experimental.chat.system.transform": async (_input, output) => {
      const blocks = await store.listBlocks("all");
      const xml = renderMemoryBlocks(blocks);
      if (!xml) return;

      // Insert early (right after provider header) for salience.
      // OpenCode will re-join system chunks to preserve caching.
      const insertAt = output.system.length > 0 ? 1 : 0;
      output.system.splice(insertAt, 0, xml);
    },

    tool: {
      memory_list: MemoryList(store),
      memory_set: MemorySet(store),
      memory_replace: MemoryReplace(store),
    },
  };
};
