


import { cn } from "@/lib/utils";
import { DashboardSquare01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/**
 * Registry logos at `public/provider-logos/{provider_type}.{ext}`; key matches
 * `PROVIDER_REGISTRY` (lowercase). Extension varies per asset (svg preferred).
 */
const PROVIDER_LOGO_FILE: Record<string, string> = {
  openai: "openai.svg",
  mistral: "mistral.svg",
  gemini: "gemini.svg",
  anthropic: "anthropic.svg",
  deepseek: "deepseek.svg",
  huggingface: "huggingface.svg",
  kimi: "kimi.jpg",
  moonshot: "kimi.jpg",
  qwen: "qwen.png",
  "tongyi-qianwen": "qwen.png",
  openrouter: "openrouter.svg",
  vllm: "vllm.svg",
  ollama: "ollama.svg",
  llama_cpp: "llama_cpp.svg",
  "azure-openai": "misc/microsoft.svg",
  minimax: "misc/minimax.png",
  nvidia: "misc/nvidia.svg",
  perplexity: "misc/perplexity.png",
  xai: "misc/xai.svg",
  "zhipu-ai": "misc/z-ai.svg",
};

export function apiProviderLogoSrc(
  providerType: string | undefined | null,
): string | undefined {
  if (!providerType) return undefined;
  const file = PROVIDER_LOGO_FILE[providerType];
  if (!file) return undefined;
  return `${import.meta.env.BASE_URL}provider-logos/${file}`;
}

interface ApiProviderLogoProps {
  providerType: string | undefined | null;
  className?: string;
  title?: string;
}

const DARK_INVERT_LOGOS = new Set(["openai", "ollama", "openrouter"]);

/** Provider logo from `public/provider-logos/`; monochrome ones invert in dark mode. */
export function ApiProviderLogo({ providerType, className, title }: ApiProviderLogoProps) {
  const src = apiProviderLogoSrc(providerType);
  if (!src) {
    return (
      <span title={title} aria-hidden className="inline-flex shrink-0">
        <HugeiconsIcon icon={DashboardSquare01Icon} className={cn("shrink-0", className)} />
      </span>
    );
  }
  return (
    <img
      src={src}
      alt=""
      title={title}
      aria-hidden
      className={cn(
        "shrink-0 object-contain",
        providerType && DARK_INVERT_LOGOS.has(providerType) && "dark:invert",
        className,
      )}
    />
  );
}
