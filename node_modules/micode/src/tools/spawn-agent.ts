import type { PluginInput } from "@opencode-ai/plugin";
import { type ToolContext, tool } from "@opencode-ai/plugin/tool";

// Extended context with metadata (available but not typed in plugin API)
// Using intersection to add optional metadata without type conflict
type ExtendedContext = ToolContext & {
  metadata?: (input: { title?: string; metadata?: Record<string, unknown> }) => void;
};

interface SessionCreateResponse {
  data?: { id?: string };
}

interface MessagePart {
  type: string;
  text?: string;
}

interface SessionMessage {
  info?: { role?: "user" | "assistant" };
  parts?: MessagePart[];
}

interface SessionMessagesResponse {
  data?: SessionMessage[];
}

interface AgentTask {
  agent: string;
  prompt: string;
  description: string;
}

export function createSpawnAgentTool(ctx: PluginInput) {
  async function runAgent(
    task: AgentTask,
    toolCtx: ExtendedContext,
    progressState?: { completed: number; total: number; startTime: number },
  ): Promise<string> {
    const { agent, prompt, description } = task;
    const agentStartTime = Date.now();

    const updateProgress = (status: string) => {
      if (toolCtx.metadata && progressState) {
        const elapsed = ((Date.now() - progressState.startTime) / 1000).toFixed(0);
        toolCtx.metadata({
          title: `[${progressState.completed}/${progressState.total}] ${status} (${elapsed}s)`,
        });
      }
    };

    updateProgress(`Running ${agent}...`);

    try {
      const sessionResp = (await ctx.client.session.create({
        body: {},
        query: { directory: ctx.directory },
      })) as SessionCreateResponse;

      const sessionID = sessionResp.data?.id;
      if (!sessionID) {
        return `## ${description}\n\n**Agent**: ${agent}\n**Error**: Failed to create session`;
      }

      await ctx.client.session.prompt({
        path: { id: sessionID },
        body: {
          parts: [{ type: "text", text: prompt }],
          agent: agent,
        },
        query: { directory: ctx.directory },
      });

      const messagesResp = (await ctx.client.session.messages({
        path: { id: sessionID },
        query: { directory: ctx.directory },
      })) as SessionMessagesResponse;

      const messages = messagesResp.data || [];
      const lastAssistant = messages.filter((m) => m.info?.role === "assistant").pop();

      const result =
        lastAssistant?.parts
          ?.filter((p) => p.type === "text" && p.text)
          .map((p) => p.text)
          .join("\n") || "(No response from agent)";

      await ctx.client.session
        .delete({
          path: { id: sessionID },
          query: { directory: ctx.directory },
        })
        .catch(() => {});

      const agentTime = ((Date.now() - agentStartTime) / 1000).toFixed(1);
      return `## ${description} (${agentTime}s)\n\n**Agent**: ${agent}\n\n### Result\n\n${result}`;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return `## ${description}\n\n**Agent**: ${agent}\n**Error**: ${errorMsg}`;
    }
  }

  return tool({
    description: `Spawn subagents to execute tasks in PARALLEL.
All agents in the array run concurrently via Promise.all.

Example:
spawn_agent({
  agents: [
    {agent: "mm-stack-detector", prompt: "...", description: "Detect stack"},
    {agent: "mm-dependency-mapper", prompt: "...", description: "Map deps"}
  ]
})`,
    args: {
      agents: tool.schema
        .array(
          tool.schema.object({
            agent: tool.schema.string().describe("Agent to spawn"),
            prompt: tool.schema.string().describe("Full prompt/instructions"),
            description: tool.schema.string().describe("Short description"),
          }),
        )
        .describe("Agents to spawn in parallel"),
    },
    execute: async (args, toolCtx) => {
      const { agents } = args;
      const extCtx = toolCtx as ExtendedContext;

      if (!agents || agents.length === 0) {
        return "## spawn_agent Failed\n\nNo agents specified.";
      }

      if (agents.length === 1) {
        extCtx.metadata?.({ title: `Running ${agents[0].agent}...` });
        return runAgent(agents[0], extCtx);
      }

      // Multiple agents - run in parallel
      const startTime = Date.now();
      const progressState = { completed: 0, total: agents.length, startTime };

      extCtx.metadata?.({
        title: `Running ${agents.length} agents in parallel...`,
      });

      const runWithProgress = async (task: AgentTask): Promise<string> => {
        const result = await runAgent(task, extCtx, progressState);
        progressState.completed++;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        extCtx.metadata?.({
          title: `[${progressState.completed}/${agents.length}] ${task.agent} done (${elapsed}s)`,
        });
        return result;
      };

      const results = await Promise.all(agents.map(runWithProgress));
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);

      extCtx.metadata?.({
        title: `${agents.length} agents completed in ${totalTime}s`,
      });

      return `# ${agents.length} agents completed in ${totalTime}s (parallel)\n\n${results.join("\n\n---\n\n")}`;
    },
  });
}
