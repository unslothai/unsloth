import type { ReactElement } from "react";
import { EyeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";

type StatusTone = "success" | "error";

type RecipeStudioHeaderProps = {
  previewLoading: boolean;
  statusMessage: {
    tone: StatusTone;
    text: string;
  } | null;
  onPreview: () => void;
};

const STATUS_MESSAGE_CLASS: Record<StatusTone, string> = {
  success: "mt-2 text-xs text-emerald-600",
  error: "mt-2 text-xs text-rose-600",
};

export function RecipeStudioHeader({
  previewLoading,
  statusMessage,
  onPreview,
}: RecipeStudioHeaderProps): ReactElement {
  return (
    <div className="mb-6 flex flex-col gap-4">
      <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[1fr_auto] lg:items-center">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Create Data Recipe</h1>
          <p className="text-sm text-muted-foreground">
            Design synthetic-data pipelines with Data Designer.
          </p>
          {statusMessage && (
            <p className={STATUS_MESSAGE_CLASS[statusMessage.tone]}>
              {statusMessage.text}
            </p>
          )}
        </div>
        <div className="flex items-center justify-start gap-2 lg:justify-end">
          <Button
            type="button"
            size="sm"
            onClick={onPreview}
            disabled={previewLoading}
            className="gap-2 text-xs"
          >
            {previewLoading ? (
              <Spinner className="size-3.5" />
            ) : (
              <HugeiconsIcon icon={EyeIcon} className="size-3.5" />
            )}
            Preview
          </Button>
        </div>
      </div>
    </div>
  );
}
