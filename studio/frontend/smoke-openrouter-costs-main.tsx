import { createRoot } from "react-dom/client";
import { ComposerModelPricing } from "./src/features/chat/components/composer-model-pricing";
import { CostReceiptDetails } from "./src/features/chat/components/recorded-cost";
import { setProviderModelCatalog } from "./src/features/chat/model-catalog";
import { buildExternalModelId } from "./src/features/chat/external-providers";
import { useChatRuntimeStore } from "./src/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "./src/features/chat/stores/external-providers-store";
import "./src/index.css";
const params = new URLSearchParams(location.search);
document.documentElement.classList.toggle("dark", params.get("theme") === "dark");
const model = "vendor/example";
setProviderModelCatalog("openrouter", [{ id: model, pricing: { rates: { prompt: "0.000001", completion: "0.000002", input_cache_read: "0.0000001" }, overrides: [{ min_prompt_tokens: 200000, prompt: "0.000003" }] } }]);
useExternalProvidersStore.setState({ connectionsEnabled: true, providers: [{ id: "smoke", name: "OpenRouter", providerType: "openrouter", baseUrl: "https://openrouter.ai/api/v1", models: [model], createdAt: 0, updatedAt: 0 }] });
useChatRuntimeStore.getState().setCheckpoint(buildExternalModelId("smoke", model), null);
createRoot(document.getElementById("root")!).render(<main className="min-h-screen bg-background p-6 text-foreground"><div className="flex justify-end pt-[680px]">
{params.has("receipt") ? <CostReceiptDetails custom={{ costReceipts: [{ provider: "openrouter", attemptId: "attempt", generationId: "gen-example", requestedModel: model, cost: 0.000021, usage: { prompt_tokens: 12, completion_tokens: 5, cost_details: { upstream_inference_cost: 0.0001 } } }] }} /> : <ComposerModelPricing />}
</div></main>);
