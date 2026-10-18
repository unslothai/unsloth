export type ContextUsageReasoningFields = {
  enable_thinking?: unknown;
  thinking?: unknown;
  reasoning_effort?: unknown;
  preserve_thinking?: unknown;
};

export function contextUsageReasoningKey(fields: ContextUsageReasoningFields): string {
  const thinking = fields.thinking as { type?: unknown } | undefined;
  const enabled = typeof fields.enable_thinking === "boolean"
    ? fields.enable_thinking
    : thinking?.type === "enabled"
      ? true
      : thinking?.type === "disabled"
        ? false
        : null;
  return JSON.stringify([
    enabled,
    typeof fields.reasoning_effort === "string" ? fields.reasoning_effort : null,
    typeof fields.preserve_thinking === "boolean" ? fields.preserve_thinking : null,
  ]);
}
