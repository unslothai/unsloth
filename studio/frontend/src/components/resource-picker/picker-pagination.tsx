


import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { Ref } from "react";
import { PICKER_OPTION_FOCUS_VISIBLE_CLASS } from "./picker-focus";

function PickerLoadMoreButton({
  disabled,
  onLoadMore,
}: {
  disabled: boolean;
  onLoadMore: () => void;
}) {
  const t = useT();
  return (
    <div className="flex justify-center py-2">
      <button
        type="button"
        disabled={disabled}
        onClick={onLoadMore}
        className={cn(
          "inline-flex h-7 items-center rounded-full border border-border/70 px-3 text-ui-11 font-medium text-foreground transition-colors hover:bg-foreground/[0.05] disabled:cursor-not-allowed disabled:opacity-50",
          PICKER_OPTION_FOCUS_VISIBLE_CLASS,
        )}
      >
        {t("picker.loadMore")}
      </button>
    </div>
  );
}

export function PickerHubPaginationFooter({
  isLoading,
  isLoadingMore,
  onLoadMore,
  sentinelRef,
  showLoadMore,
}: {
  isLoading: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
  sentinelRef: Ref<HTMLDivElement>;
  showLoadMore: boolean;
}) {
  return (
    <>
      <div ref={sentinelRef} className="h-px" />
      {isLoadingMore && (
        <div className="flex items-center justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      )}
      {showLoadMore && (
        <PickerLoadMoreButton
          disabled={isLoading || isLoadingMore}
          onLoadMore={onLoadMore}
        />
      )}
    </>
  );
}
