import type { Plugin } from "@opencode-ai/plugin";
import type { McpLocalConfig } from "@opencode-ai/sdk";

// Agents
import { agents, PRIMARY_AGENT_NAME } from "./agents";
// Config loader
import { loadMicodeConfig, loadModelContextLimits, mergeAgentConfigs } from "./config-loader";
import { createArtifactAutoIndexHook } from "./hooks/artifact-auto-index";
// Hooks
import { createAutoCompactHook } from "./hooks/auto-compact";
import { createCommentCheckerHook } from "./hooks/comment-checker";
import { createConstraintReviewerHook } from "./hooks/constraint-reviewer";
import { createContextInjectorHook } from "./hooks/context-injector";
import { createContextWindowMonitorHook } from "./hooks/context-window-monitor";
import { createFileOpsTrackerHook, getFileOps } from "./hooks/file-ops-tracker";
import { createFragmentInjectorHook, warnUnknownAgents } from "./hooks/fragment-injector";
import { createLedgerLoaderHook } from "./hooks/ledger-loader";
import { createMindmodelInjectorHook } from "./hooks/mindmodel-injector";
import { createSessionRecoveryHook } from "./hooks/session-recovery";
import { createTokenAwareTruncationHook } from "./hooks/token-aware-truncation";
import { artifact_search } from "./tools/artifact-search";
// Tools
import { ast_grep_replace, ast_grep_search, checkAstGrepAvailable } from "./tools/ast-grep";
import { createBatchReadTool } from "./tools/batch-read";
import { btca_ask, checkBtcaAvailable } from "./tools/btca";
import { look_at } from "./tools/look-at";
import { milestone_artifact_search } from "./tools/milestone-artifact-search";
import { createMindmodelLookupTool } from "./tools/mindmodel-lookup";
import { createOcttoTools, createSessionStore } from "./tools/octto";
// PTY System
import { createPtyTools, PTYManager } from "./tools/pty";
import { createSpawnAgentTool } from "./tools/spawn-agent";
import { log } from "./utils/logger";

// Think mode: detect keywords and enable extended thinking
const THINK_KEYWORDS = [
  /\bthink\s*(hard|deeply|carefully|through)\b/i,
  /\bthink\b.*\b(about|on|through)\b/i,
  /\b(deeply|carefully)\s*think\b/i,
  /\blet('s|s)?\s*think\b/i,
];

function detectThinkKeyword(text: string): boolean {
  return THINK_KEYWORDS.some((pattern) => pattern.test(text));
}

// MCP server configurations
const MCP_SERVERS: Record<string, McpLocalConfig> = {
  context7: {
    type: "local",
    command: ["npx", "-y", "@upstash/context7-mcp@latest"],
  },
};

// Environment-gated research MCP servers
if (process.env.PERPLEXITY_API_KEY) {
  MCP_SERVERS.perplexity = {
    type: "local",
    command: ["npx", "-y", "@anthropic/mcp-perplexity"],
  };
}

if (process.env.FIRECRAWL_API_KEY) {
  MCP_SERVERS.firecrawl = {
    type: "local",
    command: ["npx", "-y", "firecrawl-mcp"],
  };
}

const OpenCodeConfigPlugin: Plugin = async (ctx) => {
  // Validate external tool dependencies at startup
  const astGrepStatus = await checkAstGrepAvailable();
  if (!astGrepStatus.available) {
    console.warn(`[micode] ${astGrepStatus.message}`);
  }

  const btcaStatus = await checkBtcaAvailable();
  if (!btcaStatus.available) {
    console.warn(`[micode] ${btcaStatus.message}`);
  }

  // Load user config for agent overrides and feature flags
  const userConfig = await loadMicodeConfig();

  // Load model context limits from opencode.json
  const modelContextLimits = loadModelContextLimits();

  // Think mode state per session
  const thinkModeState = new Map<string, boolean>();

  // Hooks
  const autoCompactHook = createAutoCompactHook(ctx, {
    compactionThreshold: userConfig?.compactionThreshold,
    modelContextLimits,
  });
  const contextInjectorHook = createContextInjectorHook(ctx);
  const ledgerLoaderHook = createLedgerLoaderHook(ctx);
  const sessionRecoveryHook = createSessionRecoveryHook(ctx);
  const tokenAwareTruncationHook = createTokenAwareTruncationHook(ctx);
  const contextWindowMonitorHook = createContextWindowMonitorHook(ctx, { modelContextLimits });
  const commentCheckerHook = createCommentCheckerHook(ctx);
  const artifactAutoIndexHook = createArtifactAutoIndexHook(ctx);
  const fileOpsTrackerHook = createFileOpsTrackerHook(ctx);

  // Fragment injector hook - injects user-defined prompt fragments
  const fragmentInjectorHook = createFragmentInjectorHook(ctx, userConfig);

  // Warn about unknown agent names in fragments config
  if (userConfig?.fragments) {
    const knownAgentNames = new Set(Object.keys(agents));
    const fragmentAgentNames = Object.keys(userConfig.fragments);
    const warnings = warnUnknownAgents(fragmentAgentNames, knownAgentNames);
    for (const warning of warnings) {
      console.warn(warning);
    }
  }

  // Track internal sessions to prevent hook recursion (used by reviewer)
  const internalSessions = new Set<string>();

  // Mindmodel injector hook - matches tasks to patterns via keywords and injects them
  // Feature-flagged: set features.mindmodelInjection=true in micode.json to enable
  const mindmodelInjectorHook = userConfig?.features?.mindmodelInjection ? createMindmodelInjectorHook(ctx) : null;

  // Mindmodel lookup tool - agents call this when they need coding patterns
  const mindmodelLookupTool = createMindmodelLookupTool(ctx);

  // Constraint reviewer hook - reviews generated code against .mindmodel/ constraints
  const constraintReviewerHook = createConstraintReviewerHook(ctx, async (reviewPrompt) => {
    let sessionId: string | undefined;
    try {
      const sessionResult = await ctx.client.session.create({
        body: { title: "constraint-reviewer" },
      });

      if (!sessionResult.data?.id) {
        log.warn("mindmodel", "Failed to create reviewer session");
        return '{"status": "PASS", "violations": [], "summary": "Review skipped"}';
      }
      sessionId = sessionResult.data.id;

      // Mark as internal to prevent hook recursion
      internalSessions.add(sessionId);

      const promptResult = await ctx.client.session.prompt({
        path: { id: sessionId },
        body: {
          agent: "mm-constraint-reviewer",
          tools: {},
          parts: [{ type: "text", text: reviewPrompt }],
        },
      });

      if (!promptResult.data?.parts) {
        return '{"status": "PASS", "violations": [], "summary": "Empty response"}';
      }

      let responseText = "";
      for (const part of promptResult.data.parts) {
        if (part.type === "text" && "text" in part) {
          responseText += (part as { text: string }).text;
        }
      }

      return responseText;
    } catch (error) {
      log.warn("mindmodel", `Reviewer failed: ${error instanceof Error ? error.message : "unknown"}`);
      return '{"status": "PASS", "violations": [], "summary": "Review failed"}';
    } finally {
      if (sessionId) {
        internalSessions.delete(sessionId);
        await ctx.client.session.delete({ path: { id: sessionId } }).catch(() => {});
      }
    }
  });

  // PTY System
  const ptyManager = new PTYManager();
  const ptyTools = createPtyTools(ptyManager);

  // Spawn agent tool (for subagents to spawn other subagents)
  const spawn_agent = createSpawnAgentTool(ctx);

  // Batch read tool (for parallel file reads)
  const batch_read = createBatchReadTool(ctx);

  // Octto (browser-based brainstorming) tools
  const octtoSessionStore = createSessionStore();

  // Track octto sessions per opencode session for cleanup
  const octtoSessionsMap = new Map<string, Set<string>>();

  const octtoTools = createOcttoTools(octtoSessionStore, ctx.client, {
    onCreated: (parentSessionId, octtoSessionId) => {
      const sessions = octtoSessionsMap.get(parentSessionId) ?? new Set<string>();
      sessions.add(octtoSessionId);
      octtoSessionsMap.set(parentSessionId, sessions);
    },
    onEnded: (parentSessionId, octtoSessionId) => {
      const sessions = octtoSessionsMap.get(parentSessionId);
      if (!sessions) return;
      sessions.delete(octtoSessionId);
      if (sessions.size === 0) {
        octtoSessionsMap.delete(parentSessionId);
      }
    },
  });

  return {
    // Tools
    tool: {
      ast_grep_search,
      ast_grep_replace,
      btca_ask,
      look_at,
      artifact_search,
      milestone_artifact_search,
      spawn_agent,
      batch_read,
      ...mindmodelLookupTool,
      ...ptyTools,
      ...octtoTools,
    },

    config: async (config) => {
      // Allow all permissions globally - no prompts
      config.permission = {
        ...config.permission,
        edit: "allow",
        bash: "allow",
        webfetch: "allow",
        doom_loop: "allow",
        external_directory: "allow",
      };

      // Merge user config overrides into plugin agents
      const mergedAgents = mergeAgentConfigs(agents, userConfig);

      // Add our agents - our agents override OpenCode defaults, demote built-in build/plan to subagent
      config.agent = {
        ...config.agent, // OpenCode defaults first
        build: { ...config.agent?.build, mode: "subagent" },
        plan: { ...config.agent?.plan, mode: "subagent" },
        triage: { ...config.agent?.triage, mode: "subagent" },
        docs: { ...config.agent?.docs, mode: "subagent" },
        // Our agents override - spread these LAST so they take precedence
        ...Object.fromEntries(Object.entries(mergedAgents).filter(([k]) => k !== PRIMARY_AGENT_NAME)),
        [PRIMARY_AGENT_NAME]: mergedAgents[PRIMARY_AGENT_NAME],
      };

      // Add MCP servers (plugin servers override defaults)
      config.mcp = {
        ...config.mcp,
        ...MCP_SERVERS,
      };

      // Add commands
      config.command = {
        ...config.command,
        init: {
          description: "Initialize project with ARCHITECTURE.md and CODE_STYLE.md",
          agent: "project-initializer",
          template: `Initialize this project. $ARGUMENTS`,
        },
        mindmodel: {
          description: "Generate .mindmodel/ constraints for this project",
          agent: "mm-orchestrator",
          template: `Generate mindmodel for this project. $ARGUMENTS`,
        },
        ledger: {
          description: "Create or update continuity ledger for session state",
          agent: "ledger-creator",
          template: `Update the continuity ledger. $ARGUMENTS`,
        },
        search: {
          description: "Search past handoffs, plans, and ledgers",
          agent: "artifact-searcher",
          template: `Search for: $ARGUMENTS`,
        },
      };
    },

    "chat.message": async (input, output) => {
      // Extract text from user message
      const text = output.parts
        .filter((p) => p.type === "text" && "text" in p)
        .map((p) => (p as { text: string }).text)
        .join(" ");

      // Track if think mode was requested
      thinkModeState.set(input.sessionID, detectThinkKeyword(text));

      // Check for override command
      await constraintReviewerHook["chat.message"](input, output);
    },

    "chat.params": async (input, output) => {
      // Inject user-defined fragments FIRST (highest priority, beginning of prompt)
      await fragmentInjectorHook["chat.params"](input, output);

      // Inject ledger context (high priority)
      await ledgerLoaderHook["chat.params"](input, output);

      // Inject project context files
      await contextInjectorHook["chat.params"](input, output);

      // Inject context window status
      await contextWindowMonitorHook["chat.params"](input, output);

      // If think mode was requested, increase thinking budget
      if (thinkModeState.get(input.sessionID)) {
        output.options = {
          ...output.options,
          thinking: {
            type: "enabled",
            budget_tokens: 32000,
          },
        };
      }
    },

    // Structured compaction prompt (Factory.ai / pi-mono best practices)
    "experimental.session.compacting": async (
      input: { sessionID: string },
      output: { context: string[]; prompt?: string },
    ) => {
      // Get file operations for this session
      const fileOps = getFileOps(input.sessionID);
      const readPaths = Array.from(fileOps.read).sort();
      const modifiedPaths = Array.from(fileOps.modified).sort();

      const fileOpsSection = `
## File Operations
### Read
${readPaths.length > 0 ? readPaths.map((p) => `- \`${p}\``).join("\n") : "- (none)"}

### Modified
${modifiedPaths.length > 0 ? modifiedPaths.map((p) => `- \`${p}\``).join("\n") : "- (none)"}`;

      output.prompt = `Create a structured summary for continuing this conversation. Use this EXACT format:

# Session Summary

## Goal
{The core objective being pursued - one sentence describing success criteria}

## Constraints & Preferences
{Technical requirements, patterns to follow, things to avoid - or "(none)"}

## Progress
### Done
- [x] {Completed items with specific details}

### In Progress
- [ ] {Current work - what's actively being worked on}

### Blocked
- {Issues preventing progress, if any - or "(none)"}

## Key Decisions
- **{Decision}**: {Rationale - why this choice was made}

## Next Steps
1. {Ordered list of what to do next - be specific}

## Critical Context
- {Data, examples, references, or findings needed to continue work}
- {Important discoveries or insights from this session}
${fileOpsSection}

IMPORTANT:
- Preserve EXACT file paths and function names
- Focus on information needed to continue seamlessly
- Be specific about what was done, not vague summaries
- Include any error messages or issues encountered`;
    },

    // Tool output processing
    "tool.execute.after": async (
      input: { tool: string; sessionID: string; callID: string; args?: Record<string, unknown> },
      output: { output?: string },
    ) => {
      // Token-aware truncation
      await tokenAwareTruncationHook["tool.execute.after"]({ name: input.tool, sessionID: input.sessionID }, output);

      // Comment checker for Edit tool
      await commentCheckerHook["tool.execute.after"]({ tool: input.tool, args: input.args }, output);

      // Directory-aware context injection for Read/Edit
      await contextInjectorHook["tool.execute.after"]({ tool: input.tool, args: input.args }, output);

      // Auto-index artifacts when written to thoughts/ directories
      await artifactAutoIndexHook["tool.execute.after"]({ tool: input.tool, args: input.args }, output);

      // Track file operations for ledger
      await fileOpsTrackerHook["tool.execute.after"](
        { tool: input.tool, sessionID: input.sessionID, args: input.args },
        output,
      );

      // Constraint review for Edit/Write
      await constraintReviewerHook["tool.execute.after"](
        { tool: input.tool, sessionID: input.sessionID, args: input.args },
        output,
      );
    },

    // Transform messages: match task keywords and prepare mindmodel injection
    "experimental.chat.messages.transform": async (input, output) => {
      if (!mindmodelInjectorHook) return;
      // Skip internal sessions (reviewer)
      const sessionID = (input as { sessionID?: string }).sessionID;
      if (sessionID && internalSessions.has(sessionID)) return;

      await mindmodelInjectorHook["experimental.chat.messages.transform"](input, output);
    },

    // Transform system prompt: filter CLAUDE.md/AGENTS.md + inject mindmodel
    "experimental.chat.system.transform": async (input, output) => {
      // Filter out CLAUDE.md/AGENTS.md from system prompt for our agents
      output.system = output.system.filter((s) => {
        // Keep entries that don't come from CLAUDE.md or AGENTS.md
        if (s.startsWith("Instructions from:")) {
          const path = s.split("\n")[0];
          if (path.includes("CLAUDE.md") || path.includes("AGENTS.md")) {
            return false;
          }
        }
        return true;
      });

      // Inject mindmodel patterns into system prompt (if enabled)
      if (mindmodelInjectorHook) {
        await mindmodelInjectorHook["experimental.chat.system.transform"](input, output);
      }
    },

    event: async ({ event }) => {
      // Session cleanup (think mode + PTY + octto + constraint reviewer)
      if (event.type === "session.deleted") {
        const props = event.properties as { info?: { id?: string } } | undefined;
        if (props?.info?.id) {
          const sessionId = props.info.id;
          thinkModeState.delete(sessionId);
          ptyManager.cleanupBySession(sessionId);
          constraintReviewerHook.cleanupSession(sessionId);

          // Cleanup octto sessions
          const octtoSessions = octtoSessionsMap.get(sessionId);
          if (octtoSessions) {
            for (const octtoSessionId of octtoSessions) {
              await octtoSessionStore.endSession(octtoSessionId).catch(() => {});
            }
            octtoSessionsMap.delete(sessionId);
          }
        }
      }

      // Run all event hooks
      await autoCompactHook.event({ event });
      await sessionRecoveryHook.event({ event });
      await tokenAwareTruncationHook.event({ event });
      await contextWindowMonitorHook.event({ event });

      // File ops tracker cleanup
      await fileOpsTrackerHook.event({ event });
    },
  };
};

export default OpenCodeConfigPlugin;
