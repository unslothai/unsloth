import type { TourStep } from "@/features/tour";

export function buildChatTourSteps({
  canCompare,
  openModelSelector,
  closeModelSelector,
  openSettings,
  closeSettings,
  enterCompare,
  exitCompare,
}: {
  canCompare: boolean;
  openModelSelector: () => void;
  closeModelSelector: () => void;
  openSettings: () => void;
  closeSettings: () => void;
  enterCompare: () => void;
  exitCompare: () => void;
}): TourStep[] {
  const steps: TourStep[] = [
    {
      id: "model",
      target: "chat-model-selector",
      title: "Pick a model",
      body: <>Hub models vs fine-tuned adapters live here.</>,
    },
    {
      id: "model-tabs",
      target: "chat-model-selector-popover",
      title: "Two tabs",
      body: <>Hub: search HF. Fine-tuned: your local LoRA adapters.</>,
      onEnter: openModelSelector,
      onExit: closeModelSelector,
    },
    {
      id: "settings",
      target: "chat-settings",
      title: "Settings sidebar",
      body: <>Sampling + system prompt live in the right sidebar.</>,
      onEnter: openSettings,
      onExit: closeSettings,
    },
  ];

  if (canCompare) {
    steps.push(
      {
        id: "compare-btn",
        target: "chat-compare",
        title: "Compare mode",
        body: <>When a LoRA is selected, you can compare base vs fine-tuned.</>,
      },
      {
        id: "compare-view",
        target: "chat-compare-view",
        title: "Side-by-side threads",
        body: <>Same prompt, 2 threads. Compose at bottom.</>,
        onEnter: enterCompare,
        onExit: exitCompare,
      },
    );
  }

  return steps;
}

