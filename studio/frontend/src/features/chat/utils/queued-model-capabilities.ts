import type { ChatModelRow } from "../types/runtime";

export type QueuedModelCapabilities = Pick<
  ChatModelRow,
  | "isVision"
  | "isGguf"
  | "isMlx"
  | "isAudio"
  | "audioType"
  | "hasAudioInput"
  | "hasVideoInput"
>;

export function mergeQueuedModelCapabilities(
  models: ChatModelRow[],
  checkpoint: string,
  capabilities: QueuedModelCapabilities | null,
): ChatModelRow[] {
  if (!capabilities) {
    return models;
  }

  const index = models.findIndex((model) => model.id === checkpoint);
  if (index < 0) {
    return [
      ...models,
      {
        id: checkpoint,
        name: checkpoint,
        isLora: false,
        ...capabilities,
      },
    ];
  }

  return models.map((model, modelIndex) =>
    modelIndex === index ? { ...model, ...capabilities } : model,
  );
}
