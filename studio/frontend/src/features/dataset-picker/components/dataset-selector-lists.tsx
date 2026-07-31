// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isHfAuthError } from "@/components/resource-picker/hf-error";
import { hubResourceIdsEqual } from "@/components/resource-picker/hub-resource-id";
import {
  PickerHubPaginationFooter,
  PickerSearchError,
  RetryButton,
} from "@/components/resource-picker/picker-tab-toggle";
import { SelectablePickerItem } from "@/components/resource-picker/selectable-picker-item";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cacheLocalPathMatchesSelection } from "@/features/training";
import { useT } from "@/i18n";
import type { Ref } from "react";

export type DatasetDeviceItem =
  | {
      kind: "local";
      key: string;
      title: string;
      detail: string;
      path: string;
    }
  | {
      kind: "cached";
      key: string;
      title: string;
      detail: string;
      repoId: string;
      cachePath: string | null;
    };

export function DatasetDeviceList({
  items,
  isLoading,
  error,
  warning,
  hasQuery,
  onRetry,
  onOpenDataRecipes,
  selectedLocalPath,
  selectedHfRepoId,
  onPick,
}: {
  items: DatasetDeviceItem[];
  isLoading: boolean;
  error: string | null;
  warning: boolean;
  hasQuery: boolean;
  onRetry: () => void;
  onOpenDataRecipes: () => void;
  selectedLocalPath: string | null;
  selectedHfRepoId: string | null;
  onPick: (item: DatasetDeviceItem) => void;
}) {
  const t = useT();
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.datasetPicker.scanningLocal")}
      </div>
    );
  }
  if (items.length === 0) {
    if (error) {
      return (
        <div className="flex flex-col items-center gap-1.5 px-4 py-8 text-center">
          <p className="text-ui-12p5 font-medium text-foreground">
            {t("studio.datasetPicker.couldntScan")}
          </p>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {error}
          </p>
          <RetryButton onRetry={onRetry} />
        </div>
      );
    }
    if (hasQuery) {
      return (
        <div className="px-4 py-8 text-center text-xs text-muted-foreground">
          {t("studio.datasetPicker.noDatasetsFound")}
        </div>
      );
    }
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <p className="text-xs text-muted-foreground">
          {t("studio.datasetPicker.noLocalDatasets")}
        </p>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onOpenDataRecipes}
        >
          {t("studio.datasetPicker.openDataRecipes")}
        </Button>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5 p-0.5">
      {items.map((item) => {
        const active =
          item.kind === "local"
            ? cacheLocalPathMatchesSelection(selectedLocalPath, item.path)
            : hubResourceIdsEqual(selectedHfRepoId, item.repoId);
        return (
          <li key={item.key}>
            <SelectablePickerItem
              active={active}
              onSelect={() => onPick(item)}
              values={[item.kind === "local" ? item.path : item.repoId]}
            >
              <span className="block min-w-0 flex-1 truncate">
                {item.title}
              </span>
              <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                {item.detail}
              </span>
            </SelectablePickerItem>
          </li>
        );
      })}
      {warning && (
        <li className="px-2 py-1 text-ui-10p5 text-muted-foreground/80">
          {t("studio.datasetPicker.someLocationsUnscanned")}
        </li>
      )}
    </ul>
  );
}

export function DatasetHubList({
  items,
  isLoading,
  isLoadingMore,
  value,
  hasQuery,
  error,
  onPick,
  onLoadMore,
  onRetry,
  showLoadMore,
  sentinelRef,
}: {
  items: ReadonlyArray<{ id: string }>;
  isLoading: boolean;
  isLoadingMore: boolean;
  value: string | null;
  hasQuery: boolean;
  error: string | null;
  onPick: (id: string) => void;
  onLoadMore: () => void;
  onRetry: () => void;
  showLoadMore: boolean;
  sentinelRef: Ref<HTMLDivElement>;
}) {
  const t = useT();
  const isAuthError = error ? isHfAuthError(error) : false;
  const hubError = error
    ? {
        title: isAuthError
          ? t("studio.datasetPicker.tokenRejectedTitle")
          : t("studio.datasetPicker.hubUnreachable"),
        detail: isAuthError
          ? t("studio.datasetPicker.tokenRejectedBody")
          : error,
      }
    : null;
  if (isLoading && items.length === 0) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
        <Spinner className="size-4" /> {t("studio.datasetPicker.searchingHub")}
      </div>
    );
  }
  if (items.length === 0) {
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
            {t("studio.datasetPicker.noDatasetsFound")}
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
      {items.length === 0 && hasQuery && (
        <div className="px-4 py-8 text-center text-xs text-muted-foreground">
          {t("studio.datasetPicker.noDatasetsFound")}
        </div>
      )}
      <ul className="flex flex-col gap-0.5 p-0.5">
        {items.map((item) => {
          const active = hubResourceIdsEqual(value, item.id);
          return (
            <li key={item.id}>
              <SelectablePickerItem
                active={active}
                onSelect={() => onPick(item.id)}
                values={[item.id]}
              >
                <span className="block min-w-0 flex-1 truncate">{item.id}</span>
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
