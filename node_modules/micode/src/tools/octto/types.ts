// src/tools/octto/types.ts

import type { ToolContext } from "@opencode-ai/plugin/tool";
import type { createOpencodeClient } from "@opencode-ai/sdk";

// Using `any` to avoid exposing zod types in declaration files.
// The actual tools are typesafe via zod schemas.
export interface OcttoTool {
  description: string;
  args: any;
  execute: (args: any, context: ToolContext) => Promise<string>;
}

export type OcttoTools = Record<string, OcttoTool>;

export type OpencodeClient = ReturnType<typeof createOpencodeClient>;

export interface OcttoSessionTracker {
  onCreated?: (parentSessionId: string, octtoSessionId: string) => void;
  onEnded?: (parentSessionId: string, octtoSessionId: string) => void;
}
