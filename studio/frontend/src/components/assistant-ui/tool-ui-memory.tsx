// SPDX-License-Identifier: AGPL-3.0-only
import { useChatProjectScope } from "@/features/chat/chat-project-scope";
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { Brain03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memoryToolFeedback } from "./memory-tool-feedback";
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

function MemoryIcon(props: { className?: string }) {
  return <HugeiconsIcon icon={Brain03Icon} strokeWidth={1.75} {...props} />;
}

type MemoryArgs = { scope?: unknown; value?: unknown };

export const MemoryToolUI: ToolCallMessagePartComponent<MemoryArgs> = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const scope = toolArgText(args?.scope);
  const value = toolArgText(args?.value);
  const text = result == null ? "" : stringifyToolResult(result);
  const forgetting = args?.value === null;
  const isRunning = isToolCallRunning(status);
  const cancelled = isToolCallCancelled(status);
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const projectId = useChatProjectScope();
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);
  const feedback = memoryToolFeedback({
    phase: awaitingApproval
      ? "approval"
      : isRunning
        ? "running"
        : cancelled
          ? "cancelled"
          : "settled",
    forgetting,
    project: scope === "project",
    resultText: text,
  });

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName="Memory"
        status={status}
        icon={status?.type === "complete" ? MemoryIcon : undefined}
      />
      <ToolFallbackContent>
        <div className="space-y-2 text-sm" aria-live="polite">
          <p className="text-muted-foreground">{feedback.label}</p>
          {!forgetting && value && (
            <p className="whitespace-pre-wrap wrap-break-word">{value}</p>
          )}
          {feedback.detail && (
            <p className="text-muted-foreground">{feedback.detail}</p>
          )}
          {feedback.canManage && (
            <div className="text-xs text-muted-foreground">
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
