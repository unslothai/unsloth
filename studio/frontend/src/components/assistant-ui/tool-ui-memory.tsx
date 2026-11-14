// SPDX-License-Identifier: AGPL-3.0-only
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { stringifyToolResult } from "@/lib/strip-ansi";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";

export const MemoryToolUI: ToolCallMessagePartComponent = ({
  result,
  status,
}) => {
  const text = result == null ? "" : stringifyToolResult(result);
  const saved = text.startsWith("Memory updated:");
  return (
    <div className="my-2 rounded border px-3 py-2 text-sm" aria-live="polite">
      <span>
        {status?.type === "running"
          ? "Updating memory…"
          : saved
            ? text
            : text || "Memory was not saved."}
      </span>
      {saved && (
        <button
          type="button"
          className="ml-3 underline"
          onClick={() =>
            useSettingsDialogStore
              .getState()
              .openDialog("chat", { scrollTarget: "chat-memory" })
          }
        >
          Manage memory
        </button>
      )}
    </div>
  );
};
