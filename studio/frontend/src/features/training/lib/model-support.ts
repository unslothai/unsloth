


const UNTRAINABLE_MODEL_FORMATS = new Set(["gguf", "adapter"]);
const TRAINABLE_MODEL_FORMATS = new Set(["safetensors", "checkpoint"]);

export function isUntrainableModelFormat(
  format: string | null | undefined,
): boolean {
  return format != null && UNTRAINABLE_MODEL_FORMATS.has(format);
}

export function isTrainableModelFormat(
  format: string | null | undefined,
): boolean {
  return format != null && TRAINABLE_MODEL_FORMATS.has(format);
}
