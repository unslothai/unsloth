import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const provider = "unsloth";
const maxResultCharacters = 100_000;
const model = process.env.UNSLOTH_PI_SUBAGENT_MODEL || "";
const baseUrl = process.env.UNSLOTH_PI_SUBAGENT_BASE_URL || "";
const contextWindow = positiveInt(process.env.UNSLOTH_PI_SUBAGENT_CONTEXT_WINDOW, 32768);
const maxTokens = positiveInt(
	process.env.UNSLOTH_PI_SUBAGENT_MAX_TOKENS,
	Math.min(Math.floor(contextWindow / 4), 8192),
);

function positiveInt(value: string | undefined, fallback: number): number {
	const parsed = Number.parseInt(value || "", 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function finalText(message: any): string {
	if (message?.role !== "assistant" || !Array.isArray(message.content)) return "";
	return message.content
		.filter((part: any) => part?.type === "text" && typeof part.text === "string")
		.map((part: any) => part.text)
		.join("\n")
		.trim();
}

function boundedResult(text: string): string {
	if (text.length <= maxResultCharacters) return text;
	return `${text.slice(0, maxResultCharacters)}\n\n[Local agent output truncated]`;
}

function piInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const bunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !bunVirtualScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	const executable = path.basename(process.execPath).toLowerCase();
	if (!/^(node|bun)(\.exe)?$/.test(executable)) return { command: process.execPath, args };
	return { command: "pi", args };
}

export default function unslothSubagent(pi: ExtensionAPI): void {
	if (!model || !baseUrl) {
		throw new Error("Unsloth subagent configuration is incomplete.");
	}

	pi.registerProvider(provider, {
		name: "Unsloth Studio",
		baseUrl,
		apiKey: "$UNSLOTH_PI_SUBAGENT_API_KEY",
		api: "openai-completions",
		authHeader: true,
		models: [
			{
				id: model,
				name: `${model} via Unsloth`,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens,
			},
		],
	});

	if (process.env.UNSLOTH_PI_SUBAGENT_CHILD === "1") return;

	pi.registerTool({
		name: "unsloth_agent",
		label: "Unsloth agent",
		description:
			"Local coding subagent powered by Unsloth for debugging, implementation, and codebase research. Use when the user asks to spawn an Unsloth or local agent.",
		parameters: Type.Object({
			task: Type.String({ description: "The complete task for the local Unsloth agent." }),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const extension = fileURLToPath(import.meta.url);
			const args = [
				"--mode",
				"json",
				"--print",
				"--no-session",
				"--provider",
				provider,
				"--model",
				model,
				"--no-extensions",
				"--extension",
				extension,
				`Task: ${params.task}`,
			];
			const invocation = piInvocation(args);
			let output = "";
			let stderr = "";
			let lastResponse = "";
			let aborted = false;
			const processLine = (line: string) => {
				try {
					const event = JSON.parse(line);
					if (event.type === "message_end") {
						const response = finalText(event.message);
						if (response) lastResponse = boundedResult(response);
					}
				} catch {
					// Ignore non-JSON diagnostic lines. The exit status still reports failures.
				}
			};

			const exitCode = await new Promise<number>((resolve, reject) => {
				const child = spawn(invocation.command, invocation.args, {
					cwd: ctx.cwd,
					shell: false,
					stdio: ["ignore", "pipe", "pipe"],
					env: { ...process.env, UNSLOTH_PI_SUBAGENT_CHILD: "1" },
				});
				const cancel = () => {
					aborted = true;
					child.kill();
				};
				child.on("error", reject);
				child.stdout.on("data", (chunk) => {
					output += chunk.toString();
					const lines = output.split("\n");
					output = lines.pop() || "";
					for (const line of lines) processLine(line);
				});
				child.stderr.on("data", (chunk) => {
					stderr = (stderr + chunk.toString()).slice(-100_000);
				});
				child.on("close", (code) => {
					signal?.removeEventListener("abort", cancel);
					if (output.trim()) processLine(output);
					resolve(code ?? 1);
				});
				signal?.addEventListener("abort", cancel, { once: true });
				if (signal?.aborted) cancel();
			});

			if (aborted) throw new Error("The local Unsloth agent was cancelled.");
			if (exitCode !== 0) {
				throw new Error(stderr.trim() || `The local Unsloth agent exited with code ${exitCode}.`);
			}
			return {
				content: [{ type: "text", text: lastResponse || "The local agent returned no text." }],
				details: { provider, model },
			};
		},
	});
}
