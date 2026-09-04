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

/** Give a queued run an accurate private model entry without changing the visible chat's model
 *  catalog. Status-derived capabilities override stale catalog values, and a model loaded outside
 *  Unsloth gets a minimal entry. */
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
