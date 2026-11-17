// SPDX-License-Identifier: AGPL-3.0-only
import { useChatProjectScope } from "@/features/chat/chat-project-scope";
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { partsHaveNonEmptyText } from "./message-derived";
import {
  isToolCallCancelled,
  isToolCallRunning,
  toolArgText,
} from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

type MemoryArgs = { scope?: unknown; key?: unknown; value?: unknown };

export const MemoryToolUI: ToolCallMessagePartComponent<MemoryArgs> = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const scope = toolArgText(args?.scope);
  const key = toolArgText(args?.key);
  const value = toolArgText(args?.value);
  const text = result == null ? "" : stringifyToolResult(result);
  const saved = text.startsWith("Memory updated:");
  const unchanged = text.startsWith("Memory unchanged:");
  const forgetting = args?.value === null;
  const isRunning = isToolCallRunning(status);
  const cancelled = isToolCallCancelled(status);
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const projectId = useChatProjectScope();
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);
  const label = awaitingApproval
    ? "Waiting for approval…"
    : isRunning
      ? forgetting
        ? "Forgetting memory…"
        : "Remembering…"
      : cancelled
        ? "Memory update cancelled"
        : saved
          ? forgetting
            ? "Memory forgotten"
            : "Memory saved"
          : unchanged
            ? "Memory already up to date"
            : "Memory was not saved";

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger toolName="Memory" status={status} />
      <ToolFallbackContent>
        <div className="space-y-2 text-sm" aria-live="polite">
          <p className="text-muted-foreground">{label}</p>
          {(value || key) && (
            <p className="whitespace-pre-wrap break-words">
              {forgetting
                ? key.replaceAll("_", " ")
                : value || key.replaceAll("_", " ")}
            </p>
          )}
          {!isRunning && !cancelled && !saved && !unchanged && text && (
            <p className="whitespace-pre-wrap break-words text-muted-foreground">
              {text.replace(/^Error:\s*/, "")}
            </p>
          )}
          {(saved || unchanged) && (
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>
                {scope === "project" ? "This project" : "Across your chats"}
              </span>
              <button
                type="button"
                className="underline underline-offset-4 hover:text-foreground"
                onClick={() =>
                  useSettingsDialogStore.getState().openDialog("chat", {
                    scrollTarget: "chat-memory",
                    memoryScope: {
                      projectId: scope === "project" ? projectId : null,
                    },
                  })
                }
              >
                Manage memory
              </button>
            </div>
          )}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};
