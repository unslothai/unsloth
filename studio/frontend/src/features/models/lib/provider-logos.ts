// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Provider-logo registry for Unsloth re-uploads.
 *
 * Unsloth re-uploads models from upstream providers (e.g. unsloth/Qwen2.5-7B
 * is Alibaba's Qwen). When we render an avatar for one of these repos we want
 * to show the *upstream* provider's logo, not the Unsloth profile picture —
 * the username stays "unsloth", only the picture swaps.
 *
 * --- Adding a new family ---
 *   1. Drop the logo into /public/hub/profile/logo/
 *   2. Append an entry to PROVIDER_LOGOS or add a prefix to an existing one
 *   3. Pick `treatment`, `background`, and (for "original") `fit`. See the
 *      type docs below for the available options and when to use each.
 *
 * --- Matching rules ---
 * Providers are evaluated in declaration order, and a provider matches if any
 * of its `prefixes` is a prefix of the repo name (the part after "owner/").
 *
 * IMPORTANT — most-specific providers MUST be declared first. Several
 * families share leading substrings, and the first match wins:
 *
 *   - NVIDIA's `Llama-3.1-Nemotron-`, `Llama-3.3-Nemotron-`,
 *     `Llama-3.1-Minitron-` and `Mistral-NeMo-` must come before meta-llama
 *     and mistralai, otherwise the latter win on the shorter prefix.
 *
 *   - DeepSeek's `DeepSeek-R1-Distill-` must come before Qwen / meta-llama
 *     so the distill variants don't get mis-attributed to the distillation
 *     target.
 *
 * Repo names from the HF API are case-sensitive — match the exact casing the
 * upstream publisher uses (e.g. `Qwen` not `qwen`, `phi-` lowercase for v1/v2
 * vs `Phi-` capitalized for v3+).
 */

/**
 * How the logo is colored.
 *   - "original": render the file as-is via <img>. Use for PNG/JPG logos and
 *     any SVG whose own colors you want to preserve.
 *   - "mono-theme": render the SVG via a CSS mask painted with the current
 *     text color (`text-foreground` → light in dark theme, dark in light).
 *     Use for single-color SVGs that should follow the app theme.
 *   - "mono-black": same as mono-theme but the fill is always pure black,
 *     regardless of theme. Use for marks that should read "ink on paper" on
 *     a white tile in both themes.
 */
export type LogoTreatment = "original" | "mono-theme" | "mono-black";

/**
 * Tile background behind the logo. "white" gives consistent contrast for
 * colored brand marks across light & dark themes; "transparent" lets the
 * surrounding surface show through (useful for logos with their own
 * background or when the surface already provides contrast).
 */
export type LogoBackground = "white" | "transparent";

/**
 * How the logo image is sized within the tile.
 *   - "contain" (default): padded inside the tile at ~75% — gives the logo
 *     breathing room.
 *   - "cover": fills the tile edge-to-edge. Use only for full-bleed assets
 *     (e.g. portrait-style brand cards).
 * Only consulted when `treatment` is `"original"`. Mono treatments always
 * pad the silhouette inside the tile.
 */
export type LogoFit = "contain" | "cover";

export interface ProviderLogo {
	/** Stable identifier (kebab-case). Used for debug/telemetry only. */
	id: string;
	/** Display name used as the avatar's accessible label. */
	name: string;
	/** Path to the logo under /public (Vite serves it at the same URL). */
	logoPath: string;
	treatment: LogoTreatment;
	background: LogoBackground;
	/** Only consulted when treatment is "original". Defaults to "contain". */
	fit?: LogoFit;
	/**
	 * Prefixes of the repo name (the part after `owner/`) that map to this
	 * provider. A prefix match is sufficient — variants like `-Instruct`,
	 * `-bnb-4bit`, `-GGUF` etc. ride along automatically. Match on the family
	 * stem (e.g. `Qwen`, `gemma-`, `DeepSeek-`) so future minor versions are
	 * picked up without an extra entry here.
	 */
	prefixes: readonly string[];
}

export const PROVIDER_LOGOS: readonly ProviderLogo[] = [
	// --- NVIDIA ---------------------------------------------------------------
	// Declared first so `Llama-3.x-Nemotron-`, `Llama-3.x-Minitron-` and
	// `Mistral-NeMo-` steal priority from meta-llama / mistralai prefixes.
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

	// --- DeepSeek -------------------------------------------------------------
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

	// --- Microsoft ------------------------------------------------------------
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

	// --- Qwen -----------------------------------------------------------------
	// The broad `Qwen` prefix also catches Qwen-, Qwen1.5-, Qwen2-, Qwen2.5-,
	// Qwen3-, future Qwen3.x-, Qwen4-, etc. without explicit entries.
	{
		id: "qwen",
		name: "Qwen",
		logoPath: "/hub/profile/logo/qwen.png",
		treatment: "original",
		background: "white",
		prefixes: ["Qwen", "QwQ-", "QVQ-"],
	},

	// --- Moonshot AI ----------------------------------------------------------
	// Full-bleed cover render so the brand asset reads correctly without
	// padding around it.
	{
		id: "moonshotai",
		name: "Moonshot AI",
		logoPath: "/hub/profile/logo/moonshot.jpg",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["Kimi-", "Moonlight-"],
	},

	// --- Z.ai (THUDM successor) ----------------------------------------------
	// Full-bleed cover render so the brand asset fills the entire avatar tile.
	{
		id: "zai-org",
		name: "Z.ai",
		logoPath: "/hub/profile/logo/zai.svg",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["GLM-", "glm-", "chatglm", "codegeex"],
	},

	// --- xAI ------------------------------------------------------------------
	// Black silhouette on a white tile — fixed coloring, ignores app theme.
	{
		id: "xai-org",
		name: "xAI",
		logoPath: "/hub/profile/logo/xai.svg",
		treatment: "mono-black",
		background: "white",
		prefixes: ["grok-"],
	},

	// --- MiniMax AI -----------------------------------------------------------
	{
		id: "minimax",
		name: "MiniMax AI",
		logoPath: "/hub/profile/logo/minimax-color.png",
		treatment: "original",
		background: "white",
		prefixes: ["MiniMax-"],
	},

	// --- Hugging Face (SmolLM family lives under HuggingFaceTB) --------------
	{
		id: "huggingface",
		name: "Hugging Face",
		logoPath: "/hub/profile/logo/hf.svg",
		treatment: "original",
		background: "white",
		prefixes: ["SmolLM"],
	},

	// --- IBM (Granite family) -----------------------------------------------
	// Full-bleed cover so the IBM mark fills the entire avatar tile.
	{
		id: "ibm",
		name: "IBM",
		logoPath: "/hub/profile/logo/ibm.png",
		treatment: "original",
		background: "transparent",
		fit: "cover",
		prefixes: ["granite-", "granitelib-"],
	},

	// --- Cohere Labs ----------------------------------------------------------
	{
		id: "cohere",
		name: "Cohere Labs",
		logoPath: "/hub/profile/logo/cohere.png",
		treatment: "original",
		background: "white",
		prefixes: ["c4ai-command", "aya-"],
	},

	// --- OpenAI ---------------------------------------------------------------
	// Mono mark — recolors to match the app theme on a transparent tile.
	{
		id: "openai",
		name: "OpenAI",
		logoPath: "/hub/profile/logo/openai.svg",
		treatment: "mono-theme",
		background: "transparent",
		prefixes: ["gpt-oss-"],
	},

	// --- Google (Gemma family + variants) ------------------------------------
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

	// --- Mistral AI -----------------------------------------------------------
	// Declared AFTER NVIDIA so `Mistral-NeMo-` (an NVIDIA collab) wins, while
	// generic `Mistral-` / `Mixtral-` / etc. fall through to Mistral AI.
	{
		id: "mistralai",
		name: "Mistral AI",
		logoPath: "/hub/profile/logo/mistral.svg",
		treatment: "original",
		background: "white",
		prefixes: ["Mistral-", "Mixtral-", "Codestral-", "Pixtral-", "Devstral-", "Ministral-", "Voxtral-", "Magistral-"],
	},

	// --- Meta (Llama family) ------------------------------------------------
	// Declared LAST among Llama-prefix providers so NVIDIA's
	// Llama-3.x-Nemotron / Llama-3.x-Minitron variants match first.
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
 * Resolve a repo name (the part after `owner/`) to its upstream provider, or
 * null if it doesn't match anything in the registry. Iterates PROVIDER_LOGOS
 * in declaration order; the first provider with any prefix that the repo
 * name starts with wins.
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

/**
 * Owners whose models we re-upload from upstream providers, and whose
 * avatars should therefore be swapped for the matched provider's logo. Kept
 * as a set so additional Unsloth-controlled orgs can be added without
 * touching call sites.
 */
const RELABELED_OWNERS: ReadonlySet<string> = new Set(["unsloth"]);

/**
 * True if the given owner is one whose avatars get replaced with the
 * upstream provider's logo (see PROVIDER_LOGOS for the mapping).
 */
export function isProviderRelabeledOwner(
	owner: string | null | undefined,
): boolean {
	if (!owner) return false;
	return RELABELED_OWNERS.has(owner.toLowerCase());
}

/**
 * Given an owner and repo name, return the upstream provider logo to render
 * in place of the owner's profile picture, or null to fall back to the
 * standard avatar. Only owners listed in RELABELED_OWNERS are eligible.
 */
export function resolveOwnerProviderLogo(
	owner: string | null | undefined,
	repoName: string | null | undefined,
): ProviderLogo | null {
	if (!isProviderRelabeledOwner(owner) || !repoName) return null;
	return matchProviderLogo(repoName);
}
