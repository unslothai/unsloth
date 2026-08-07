


import { isHfAuthError } from "@/components/resource-picker/hf-error";
import { hubResourceIdsEqual } from "@/components/resource-picker/hub-resource-id";
import { PickerHubPaginationFooter } from "@/components/resource-picker/picker-pagination";
import {
  PickerSearchError,
  RetryButton,
} from "@/components/resource-picker/picker-states";
import { resolvePickerDeviceListState } from "@/components/resource-picker/picker-tab-policy";
import { SelectablePickerItem } from "@/components/resource-picker/selectable-picker-item";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import { FolderSearchIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Ref } from "react";
import type { TrainModelDeviceItem } from "./train-model-picker-view-model";

export interface TrainModelVramView {
  est: number;
  status: VramFitStatus | null;
  detail: string | null;
}

export interface TrainModelDeviceListProps {
  items: readonly TrainModelDeviceItem[];
  isLoading: boolean;
  error: string | null;
  warning: boolean;
  activeKey: string | null;
  hasQuery: boolean;
  onPick: (model: TrainModelDeviceItem) => void;
  onRetry: () => void;
}

export interface TrainModelHubListProps {
  ids: readonly string[];
  value: string | null;
  vramMap: ReadonlyMap<string, TrainModelVramView>;
  isLoading: boolean;
  isLoadingMore: boolean;
  gpuTotalGb: number | null;
  hasQuery: boolean;
  error: string | null;
  onPick: (id: string) => void;
  onLoadMore: () => void;
  onRetry: () => void;
  showLoadMore: boolean;
  sentinelRef: Ref<HTMLDivElement>;
}

export function TrainModelDeviceList({
  items,
  isLoading,
  error,
  warning,
  activeKey,
  hasQuery,
  onPick,
  onRetry,
}: TrainModelDeviceListProps) {
  const t = useT();
  const listState = resolvePickerDeviceListState({
    error,
    hasItems: items.length > 0,
    hasQuery,
    isLoading,
    warning,
  });
  const warningContent = (
    <>
      <span>{t("studio.modelPicker.someLocationsUnscanned")}</span>
      <RetryButton onRetry={onRetry} />
    </>
  );
  if (listState === "loading") {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.modelPicker.scanningLocal")}
      </div>
    );
  }
  if (listState === "error") {
    return (
      <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
        <p className="text-ui-12p5 font-medium text-foreground">
          {t("studio.modelPicker.couldntScan")}
        </p>
        <p className="text-ui-11 leading-snug text-muted-foreground">{error}</p>
        <RetryButton onRetry={onRetry} />
      </div>
    );
  }
  if (listState === "warning") {
    return (
      <div className="flex items-center justify-between gap-3 px-2 py-2 text-ui-10p5 text-muted-foreground/80">
        {warningContent}
      </div>
    );
  }
  if (listState === "no-results") {
    return null;
  }
  if (listState === "empty") {
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <HugeiconsIcon
          icon={FolderSearchIcon}
          strokeWidth={1.5}
          className="size-5 text-muted-foreground/70"
        />
        <p className="text-xs text-muted-foreground">
          {t("studio.modelPicker.noLocalModels")}
        </p>
        <p className="text-ui-10p5 text-muted-foreground/70">
          {t("studio.modelPicker.noLocalModelsHint")}
        </p>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((model) => (
        <li key={model.key}>
          <SelectablePickerItem
            active={activeKey === model.key}
            onSelect={() => onPick(model)}
            values={[model.id, model.path]}
          >
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <span className="block min-w-0 flex-1 truncate">
                  {model.title || model.id}
                </span>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs break-all">
                {model.path}
              </TooltipContent>
            </Tooltip>
            <span className="ml-2 shrink-0 rounded-[6px] border border-border/60 px-1.5 py-0.5 text-ui-10 font-medium leading-none text-muted-foreground">
              {model.sourceLabel}
            </span>
          </SelectablePickerItem>
        </li>
      ))}
      {warning && (
        <li className="flex items-center justify-between gap-3 px-2 py-1 text-ui-10p5 text-muted-foreground/80">
          {warningContent}
        </li>
      )}
    </ul>
  );
}

export function TrainModelHubList({
  ids,
  value,
  vramMap,
  isLoading,
  isLoadingMore,
  gpuTotalGb,
  hasQuery,
  error,
  onPick,
  onLoadMore,
  onRetry,
  showLoadMore,
  sentinelRef,
}: TrainModelHubListProps) {
  const t = useT();
  const isAuthError = error ? isHfAuthError(error) : false;
  const hubError = error
    ? {
        title: isAuthError
          ? t("studio.modelPicker.tokenRejectedTitle")
          : t("studio.modelPicker.hubUnreachable"),
        detail: isAuthError ? t("studio.modelPicker.tokenRejectedBody") : error,
      }
    : null;
  if (isLoading && ids.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.modelPicker.searchingHub")}
      </div>
    );
  }
  if (ids.length === 0) {
    if (hubError) {
      return (
        <PickerSearchError
          title={hubError.title}
          detail={hubError.detail}
          onRetry={onRetry}
        />
      );
    }
    if (!hasQuery) {
      return (
        <>
          <div className="px-4 py-8 text-center text-xs text-muted-foreground">
            {t("studio.modelPicker.noModelsFound")}
          </div>
          <PickerHubPaginationFooter
            isLoading={isLoading}
            isLoadingMore={isLoadingMore}
            onLoadMore={onLoadMore}
            sentinelRef={sentinelRef}
            showLoadMore={showLoadMore}
          />
        </>
      );
    }
  }
  return (
    <>
      {ids.length === 0 && hasQuery && (
        <div className="px-4 py-8 text-center text-xs text-muted-foreground">
          {t("studio.modelPicker.noModelsFound")}
        </div>
      )}
      <ul className="flex flex-col gap-0.5 p-0.5">
        {ids.map((id) => {
          const fit = vramMap.get(id);
          const exceeds = fit?.status === "exceeds";
          const tight = fit?.status === "tight";
          return (
            <li key={id}>
              <SelectablePickerItem
                active={hubResourceIdsEqual(value, id)}
                onSelect={() => onPick(id)}
                values={[id]}
              >
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <span
                      className={cn(
                        "block min-w-0 flex-1 truncate",
                        exceeds && "text-muted-foreground",
                      )}
                    >
                      {id}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="max-w-xs break-all">
                    {id}
                    {fit && fit.est > 0 && gpuTotalGb != null && (
                      <span className="mt-1 block text-ui-10">
                        {exceeds
                          ? t("studio.modelPicker.vramNeeds", {
                              est: fit.est,
                              total: gpuTotalGb,
                            })
                          : tight
                            ? t("studio.modelPicker.vramTight", {
                                est: fit.est,
                                total: gpuTotalGb,
                              })
                            : t("studio.modelPicker.vramApprox", {
                                est: fit.est,
                              })}
                      </span>
                    )}
                  </TooltipContent>
                </Tooltip>
                <span className="ml-auto flex shrink-0 items-center gap-1.5">
                  {exceeds && (
                    <span className="rounded bg-red-50 px-1.5 py-0.5 text-ui-9 font-semibold text-red-700 dark:bg-red-950 dark:text-red-400">
                      {t("studio.modelPicker.vramOomBadge")}
                    </span>
                  )}
                  {tight && (
                    <span className="text-ui-9 font-semibold text-amber-500">
                      {t("studio.modelPicker.vramTightBadge")}
                    </span>
                  )}
                  {fit?.detail && (
                    <span className="text-ui-10 text-muted-foreground">
                      {fit.detail}
                    </span>
                  )}
                </span>
              </SelectablePickerItem>
            </li>
          );
        })}
      </ul>
      {hubError && (
        <PickerSearchError
          title={hubError.title}
          detail={hubError.detail}
          onRetry={onRetry}
          compact={true}
        />
      )}
      <PickerHubPaginationFooter
        isLoading={isLoading}
        isLoadingMore={isLoadingMore}
        onLoadMore={onLoadMore}
        sentinelRef={sentinelRef}
        showLoadMore={showLoadMore}
      />
    </>
  );
}
