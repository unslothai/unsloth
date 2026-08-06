


import { getModelConfig } from "@/features/training";

export async function fetchModelMaxPositionEmbeddings(
  modelName: string,
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<number | null> {
  const config = await getModelConfig(
    modelName,
    signal,
    hfToken?.trim() || undefined,
  );
  const value = config.max_position_embeddings;
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : null;
}
