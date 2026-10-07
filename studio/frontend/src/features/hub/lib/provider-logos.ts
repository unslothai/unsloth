// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shows the upstream provider's logo for Unsloth re-uploads. First prefix match wins, so
 * most-specific providers must come first; prefixes are case-sensitive.
 */

/** "mono-theme" masks in the current text color; "mono-black" masks in pure black. */
export type LogoTreatment = "original" | "mono-theme" | "mono-black";

export type LogoBackground = "white" | "transparent";

/** Only for "original" (mono always pads). "contain" pads at ~75%; "cover" is full-bleed. */
export type LogoFit = "contain" | "cover";

export interface ProviderLogo {
	id: string;
	name: string;
	logoPath: string;
	treatment: LogoTreatment;
	background: LogoBackground;
	/** Only consulted when treatment is "original". Defaults to "contain". */
	fit?: LogoFit;
	/** Repo-name prefixes (after `owner/`); match the family stem so variants ride along. */
	prefixes: readonly string[];
	/** Case-insensitive word-boundary fallback after all prefixes miss. Use provider-unique stems. */
	stems?: readonly string[];
	/** The provider's own Hub orgs, matched in full and case-insensitively, never as a prefix. */
	owners?: readonly string[];
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

	// `DeepSeek-R1-Distill-*` must beat Qwen/meta-llama.
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
		stems: ["gemma"],
	},

	// After NVIDIA so `Mistral-NeMo-` wins.
	{
		id: "mistralai",
		name: "Mistral AI",
		logoPath: "/hub/profile/logo/mistral.svg",
		treatment: "original",
		background: "white",
		prefixes: ["Mistral-", "Mixtral-", "Codestral-", "Pixtral-", "Devstral-", "Ministral-", "Voxtral-", "Magistral-"],
	},

	// Last among Llama-prefix providers so NVIDIA's Nemotron/Minitron match first.
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
			"meta-",
			"Muse-Glimmer"
		],
		// Not facebookresearch, which is an unrelated account.
		owners: ["meta-models", "meta-llama", "facebook"],
	},
];

function stemMatchesAtBoundary(haystack: string, stem: string): boolean {
	for (let at = haystack.indexOf(stem); at !== -1; at = haystack.indexOf(stem, at + 1)) {
		const next = haystack[at + stem.length];
		if (next === undefined || next < "a" || next > "z") return true;
	}
	return false;
}

/** Resolve a repo name to its provider: prefixes in declaration order, then `stems`. */
export function matchProviderLogo(repoName: string): ProviderLogo | null {
	if (!repoName) return null;
	for (const provider of PROVIDER_LOGOS) {
		if (provider.prefixes.some((prefix) => repoName.startsWith(prefix))) {
			return provider;
		}
	}
	const lower = repoName.toLowerCase();
	for (const provider of PROVIDER_LOGOS) {
		if (provider.stems?.some((stem) => stemMatchesAtBoundary(lower, stem))) {
			return provider;
		}
	}
	return null;
}

const RELABELED_OWNERS: ReadonlySet<string> = new Set(["unsloth"]);

export function isProviderRelabeledOwner(
	owner: string | null | undefined,
): boolean {
	if (!owner) return false;
	return RELABELED_OWNERS.has(owner.toLowerCase());
}

export function matchProviderLogoByOwner(
	owner: string | null | undefined,
): ProviderLogo | null {
	const needle = owner?.trim().toLowerCase();
	if (!needle) return null;
	for (const provider of PROVIDER_LOGOS) {
		if (provider.owners?.some((org) => org.toLowerCase() === needle)) {
			return provider;
		}
	}
	return null;
}

export function resolveOwnerProviderLogo(
	owner: string | null | undefined,
	repoName: string | null | undefined,
): ProviderLogo | null {
	const byOwner = matchProviderLogoByOwner(owner);
	if (byOwner) return byOwner;
	if (!isProviderRelabeledOwner(owner) || !repoName) return null;
	return matchProviderLogo(repoName);
}
