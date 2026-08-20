


import { Badge } from "@/components/assistant-ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { File02Icon, Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { XIcon } from "lucide-react";
import type { DocumentStatus } from "../types/rag";

export function DocumentStatusChip({
  filename,
  status,
  progress,
  error,
  onRemove,
  shared = false,
}: {
  filename: string;
  status: DocumentStatus;
  progress?: number | null;
  error?: string | null;
  onRemove?: () => void;
  /** Indexed for the whole project rather than this one chat: swap the file
   * glyph for a folder so the two scopes are told apart at a glance. */
  shared?: boolean;
}) {
  const processing = status === "pending" || status === "running";
  return (
    <Badge
      variant="outline"
      size="sm"
      title={
        error ??
        (shared
          ? `${filename} — shared with every chat in this project`
          : filename)
      }
      className={cn(
        "rounded-full inline-flex items-center gap-1.5 max-w-[16rem]",
        status === "failed" && "border-destructive/40 text-destructive",
      )}
    >
      {/* file, or folder when the doc is a project-wide source */}
      <HugeiconsIcon
        icon={shared ? Folder02Icon : File02Icon}
        strokeWidth={2}
        className="size-3 shrink-0"
      />
      <span className="truncate">{filename}</span>
      {/* spinner while indexing, else close button */}
      {processing ? (
        <span className="flex shrink-0 items-center gap-1 text-ui-10 text-muted-foreground">
          {progress != null
            ? `${Math.round(progress <= 1 ? progress * 100 : progress)}%`
            : null}
          <Spinner className="size-3.5" />
        </span>
      ) : onRemove ? (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${filename}`}
          className="shrink-0 rounded-full text-muted-foreground hover:text-foreground"
        >
          <XIcon className="size-3" />
        </button>
      ) : null}
    </Badge>
  );
}
