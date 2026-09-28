import { createRoot } from "react-dom/client";
import { ComposerFastControl } from "./src/features/chat/components/composer-fast-control";
import { FastModelSelectionContext } from "./src/features/chat/lib/fast-selection-context";
import { setOpenRouterFastTier } from "./src/features/chat/lib/openrouter-fast-tier";
import { setProviderModelCatalog } from "./src/features/chat/model-catalog";
import { useChatRuntimeStore } from "./src/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "./src/features/chat/stores/external-providers-store";
import { buildExternalModelId } from "./src/features/chat/external-providers";
import "./src/index.css";
setProviderModelCatalog("openrouter", [{ id: "xiaomi/mimo-v2.6-pro", pricing: { rates: { prompt: "0.000001", completion: "0.000002", input_cache_read: "0.0000001" }, overrides: [{ min_prompt_tokens: 200000, prompt: "0.000003" }] } }]);
const smokeParams = new URLSearchParams(location.search);
const standard = smokeParams.has("tier") ? "anthropic/claude-opus-5" : "xiaomi/mimo-v2.6-pro";
const ultraspeed = "xiaomi/mimo-v2.6-pro-ultraspeed";
if (smokeParams.has("fast")) {
  setProviderModelCatalog("openrouter", [
    { id: standard, reasoning: { supported_efforts: ["low", "high"], mandatory: true }, pricing: { rates: { prompt: "0.000001", completion: "0.000002" } } },
    { id: ultraspeed, description: "The fast speed edition, built from the same 1T MiMo-V2.6-Pro checkpoint.", reasoning: { supported_efforts: ["high"], mandatory: true }, pricing: { rates: { prompt: "0.00000435", completion: "0.0000087" } } },
  ]);
  setOpenRouterFastTier(standard, { supported: smokeParams.has("tier"), available: true, endpoints: smokeParams.has("tier") ? [{ tag: "anthropic/fast", pricing: { rates: { prompt: "0.00001", completion: "0.00005" } } }] : [], fetchedAt: Date.now(), source: "https://openrouter.ai/api/v1/models/anthropic/claude-opus-5/endpoints" });
  setOpenRouterFastTier(ultraspeed, { supported: false, available: false, endpoints: [], fetchedAt: Date.now(), source: "https://openrouter.ai" });
  useExternalProvidersStore.setState({ connectionsEnabled: true, providers: [{ id: "smoke", providerType: "openrouter", name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", models: smokeParams.has("gated") ? [standard] : [standard, ultraspeed], availableModels: smokeParams.has("unavailable") ? [standard] : [standard, ultraspeed], createdAt: 0, updatedAt: 0 }] });
  useChatRuntimeStore.getState().setCheckpoint(buildExternalModelId("smoke", smokeParams.has("direct") ? ultraspeed : standard), null);
  useChatRuntimeStore.setState({ reasoningEnabled: true, reasoningEffort: smokeParams.has("low") ? "low" : "high", modelLoading: false, runningByThreadId: smokeParams.has("busy") ? { smoke: true } : {} });
}

document.documentElement.classList.toggle("dark", smokeParams.get("theme") === "dark");
function Probe() {
  const checkpoint = useChatRuntimeStore(s => s.params.checkpoint);
  return <main className="min-h-screen bg-background p-6 text-foreground"><div className="flex justify-end pt-[680px]">
    <FastModelSelectionContext.Provider value={id => useChatRuntimeStore.getState().setCheckpoint(id, null)}><ComposerFastControl /></FastModelSelectionContext.Provider>
  </div><output aria-label="Selected model">{checkpoint}</output></main>;
}
createRoot(document.getElementById("root")!).render(<Probe />);
