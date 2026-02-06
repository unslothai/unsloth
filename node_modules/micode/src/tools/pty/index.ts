// src/tools/pty/index.ts
export { PTYManager } from "./manager";
export { RingBuffer } from "./buffer";
export { createPtySpawnTool } from "./tools/spawn";
export { createPtyWriteTool } from "./tools/write";
export { createPtyReadTool } from "./tools/read";
export { createPtyListTool } from "./tools/list";
export { createPtyKillTool } from "./tools/kill";
export type {
  PTYSession,
  PTYSessionInfo,
  PTYStatus,
  SpawnOptions,
  ReadResult,
  SearchMatch,
  SearchResult,
} from "./types";

import type { PTYManager } from "./manager";
import { createPtySpawnTool } from "./tools/spawn";
import { createPtyWriteTool } from "./tools/write";
import { createPtyReadTool } from "./tools/read";
import { createPtyListTool } from "./tools/list";
import { createPtyKillTool } from "./tools/kill";

export function createPtyTools(manager: PTYManager) {
  return {
    pty_spawn: createPtySpawnTool(manager),
    pty_write: createPtyWriteTool(manager),
    pty_read: createPtyReadTool(manager),
    pty_list: createPtyListTool(manager),
    pty_kill: createPtyKillTool(manager),
  };
}
