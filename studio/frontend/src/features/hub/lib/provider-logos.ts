// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Provider-logo registry for Unsloth re-uploads. Unsloth re-uploads upstream
 * models (e.g. unsloth/Qwen2.5-7B is Alibaba's Qwen); we render the upstream
 * provider's logo in place of the Unsloth profile picture.
 *
 * Providers are evaluated in declaration order; a provider matches if any of its
 * `prefixes` is a prefix of the repo name (the part after "owner/"). Repo names
 * are case-sensitive — match the publisher's exact casing.
 */

/**
 * Logo coloring. "original" = file as-is; "mono-theme" = CSS mask in current
 * text color (follows theme); "mono-black" = mask always pure black.
 */
export type LogoTreatment = "original" | "mono-theme" | "mono-black";

/** Tile background. "white" keeps colored marks readable; "transparent" shows the surface. */
export type LogoBackground = "white" | "transparent";

/**
 * Logo sizing (only for "original"; mono always pads). "contain" pads at ~75%;
 * "cover" is full-bleed.
 */
export type LogoFit = "contain" | "cover";

export interface ProviderLogo {
	id: string;
	name: string;
	logoPath: string;
	treatment: LogoTreatment;
	background: LogoBackground;
	fit?: LogoFit;
	/**
	 * Repo-name prefixes (after `owner/`) mapping to this provider. A prefix
	 * match suffices — variants (-Instruct, -bnb-4bit, -GGUF) ride along. Match
	 * on the family stem so future minor versions are picked up automatically.
	 */
	prefixes: readonly string[];
}

export const PROVIDER_LOGOS: readonly ProviderLogo[] = [
	// First so Llama-3.x-Nemotron/Minitron and Mistral-NeMo beat meta-llama / mistralai.
	{
		id: "nvidia",
		name: "NVIDIA",
		logoPath: "/hub/profile/logo/nvidia.svg",
		treatment: "original",
		background: "white",
		prefixes: [
			"Llama-3.1-Nemotron-",
			"Llama-3.3-Nemotron-",
			"Llama-3.1-Minitron-",
			"NVIDIA-Nemotron-",
			"Nemotron-3-",
			"Nemotron-4-",
			"Nemotron-H-",
			"Minitron-",
			"Mistral-NeMo-",
			"OpenReasoning-Nemotron-",
			"OpenCodeReasoning",
			"Cosmos-"
		],
	},

	// `DeepSeek-R1-Distill-*` must beat Qwen/meta-llama on the suffix family.
	{
		id: "deepseek-ai",
		name: "DeepSeek",
		logoPath: "/hub/profile/logo/deepseek.svg",
		treatment: "original",
		background: "white",
		prefixes: [
			"DeepSeek-R1-Distill-",
			"DeepSeek-",
			"deepseek-",
			"deepseek-llm-",
			"deepseek-coder-",
		],
	},

	{
		id: "microsoft",
		name: "Microsoft",
		logoPath: "/hub/profile/logo/microsoft.svg",
		treatment: "original",
		background: "white",
		prefixes: [
			"MAI-DS-R1",
			"NextCoder-",
			"Phi-3-",
			"Phi-3.5-",
			"Phi-4",
			"phi-1",
			"phi-2",
			"phi-",
		],
	},

	// The broad `Qwen` prefix catches Qwen1.5/2/2.5/3/future versions without explicit entries.
	{
		id: "qwen",
		name: "Qwen",
		logoPath: "/hub/profile/logo/qwen.png",
		treatment: "original",
		background: "white",
		prefixes: ["Qwen", "QwQ-", "QVQ-"],
	},

	{
		id: "moonshotai",
		name: "Moonshot AI",
		logoPath: "/hub/profile/logo/moonshot.jpg",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["Kimi-", "Moonlight-"],
	},

	// Z.ai (THUDM successor).
	{
		id: "zai-org",
		name: "Z.ai",
		logoPath: "/hub/profile/logo/zai.svg",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["GLM-", "glm-", "chatglm", "codegeex"],
	},

	{
		id: "xai-org",
		name: "xAI",
		logoPath: "/hub/profile/logo/xai.svg",
		treatment: "mono-black",
		background: "white",
		prefixes: ["grok-"],
	},

	{
		id: "minimax",
		name: "MiniMax AI",
		logoPath: "/hub/profile/logo/minimax-color.png",
		treatment: "original",
		background: "white",
		prefixes: ["MiniMax-"],
	},

	// SmolLM family lives under HuggingFaceTB.
	{
		id: "huggingface",
		name: "Hugging Face",
		logoPath: "/hub/profile/logo/hf.svg",
		treatment: "original",
		background: "white",
		prefixes: ["SmolLM"],
	},

	{
		id: "ibm",
		name: "IBM",
		logoPath: "/hub/profile/logo/ibm.png",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["granite-", "granitelib-"],
	},

	{
		id: "cohere",
		name: "Cohere Labs",
		logoPath: "/hub/profile/logo/cohere.png",
		treatment: "original",
		background: "white",
		prefixes: ["c4ai-command", "aya-"],
	},

	{
		id: "openai",
		name: "OpenAI",
		logoPath: "/hub/profile/logo/openai.svg",
		treatment: "mono-theme",
		background: "transparent",
		prefixes: ["gpt-oss-"],
	},

	{
		id: "google",
		name: "Google",
		logoPath: "/hub/profile/logo/google.png",
		treatment: "original",
		background: "white",
		prefixes: [
			"gemma-",
			"codegemma-",
			"recurrentgemma-",
			"shieldgemma-",
			"medgemma-",
			"functiongemma-",
			"translategemma-",
			"alphagenome-",
			"t5gemma-",
			"tipsv2-",
			"embeddinggemma-",
			"videoprism-",
			"txgemma-",
			"paligemma-",
			"metricx-",
			"bert-",
		],
	},

	// After NVIDIA so `Mistral-NeMo-` wins; generic Mistral-/Mixtral- fall through here.
	{
		id: "mistralai",
		name: "Mistral AI",
		logoPath: "/hub/profile/logo/mistral.svg",
		treatment: "original",
		background: "white",
		prefixes: ["Mistral-", "Mixtral-", "Codestral-", "Pixtral-", "Devstral-", "Ministral-", "Voxtral-", "Magistral-"],
	},

	// Last among Llama-prefix providers so NVIDIA's Llama-3.x-Nemotron/Minitron match first.
	{
		id: "meta-llama",
		name: "Meta",
		logoPath: "/hub/profile/logo/meta.svg",
		treatment: "original",
		background: "white",
		prefixes: [
			"Meta-Llama-",
			"Llama-Guard-",
			"LlamaGuard-",
			"CodeLlama-",
			"Llama-",
			"llama-",
			"meta-"
		],
	},
];

/**
 * Resolve a repo name (after `owner/`) to its upstream provider, or null.
 * Iterates PROVIDER_LOGOS in declaration order; first prefix match wins.
 */
export function matchProviderLogo(repoName: string): ProviderLogo | null {
	if (!repoName) return null;
	for (const provider of PROVIDER_LOGOS) {
		if (provider.prefixes.some((prefix) => repoName.startsWith(prefix))) {
			return provider;
		}
	}
	return null;
}

// Owners whose avatars get swapped for the matched provider's logo.
const RELABELED_OWNERS: ReadonlySet<string> = new Set(["unsloth"]);

export function isProviderRelabeledOwner(
	owner: string | null | undefined,
): boolean {
	if (!owner) return false;
	return RELABELED_OWNERS.has(owner.toLowerCase());
}

/**
 * Provider logo to render in place of the owner's profile picture, or null.
 * Only owners in RELABELED_OWNERS are eligible.
 */
export function resolveOwnerProviderLogo(
	owner: string | null | undefined,
	repoName: string | null | undefined,
): ProviderLogo | null {
	if (!isProviderRelabeledOwner(owner) || !repoName) return null;
	return matchProviderLogo(repoName);
}
