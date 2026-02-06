import { spawn, which } from "bun";
import { tool } from "@opencode-ai/plugin/tool";

/**
 * Check if btca CLI is available on the system.
 * Returns installation instructions if not found.
 */
export async function checkBtcaAvailable(): Promise<{ available: boolean; message?: string }> {
  const btcaPath = which("btca");
  if (btcaPath) {
    return { available: true };
  }
  return {
    available: false,
    message:
      "btca CLI not found. Library source code search will not work.\n" +
      "Install from: https://github.com/davis7dotsh/better-context\n" +
      "  bun add -g btca",
  };
}

const BTCA_TIMEOUT_MS = 120000; // 2 minutes for long clones

async function runBtca(args: string[]): Promise<{ output: string; error?: string }> {
  try {
    const proc = spawn(["btca", ...args], {
      stdout: "pipe",
      stderr: "pipe",
    });

    // Create timeout promise
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        proc.kill();
        reject(new Error("btca command timed out after 2 minutes"));
      }, BTCA_TIMEOUT_MS);
    });

    // Race between process completion and timeout
    const [stdout, stderr, exitCode] = await Promise.race([
      Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]),
      timeoutPromise,
    ]);

    if (exitCode !== 0) {
      const errorMsg = stderr.trim() || `Exit code ${exitCode}`;
      return { output: "", error: errorMsg };
    }

    return { output: stdout.trim() };
  } catch (e) {
    const err = e as Error;
    if (err.message?.includes("ENOENT")) {
      return {
        output: "",
        error:
          "btca CLI not found. Install from: https://github.com/davis7dotsh/better-context\n" + "  bun add -g btca",
      };
    }
    return { output: "", error: err.message };
  }
}

export const btca_ask = tool({
  description:
    "Ask questions about library/framework source code using btca. " +
    "Clones repos locally and searches source code to answer questions. " +
    "Use for understanding library internals, finding implementation details, or debugging.",
  args: {
    tech: tool.schema.string().describe("Resource name configured in btca (e.g., 'react', 'express')"),
    question: tool.schema.string().describe("Question to ask about the library source code"),
  },
  execute: async (args) => {
    const result = await runBtca(["ask", "-t", args.tech, "-q", args.question]);

    if (result.error) {
      return `Error: ${result.error}`;
    }

    return result.output || "No answer found";
  },
});
