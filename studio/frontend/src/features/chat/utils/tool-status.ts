


/** Mirrors NUDGE_TOOL_CALLS_STATUS in backend core/inference/tool_call_parser.py; keep in sync. */
export const NUDGE_TOOL_CALLS_STATUS = "Nudging tool calls";

export type ToolStatusKind = "nudge" | "terminal" | "web";

/** Which glyph the badge shows: exact match for the nudge, "Running" prefix for sandbox tools, globe otherwise. */
export function toolStatusKind(status: string): ToolStatusKind {
  if (status === NUDGE_TOOL_CALLS_STATUS) {
    return "nudge";
  }
  return status.startsWith("Running") ? "terminal" : "web";
}
