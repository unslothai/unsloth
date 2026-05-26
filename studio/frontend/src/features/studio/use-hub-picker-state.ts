// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDebouncedValue } from "@/hooks";
import { useCallback, useRef, useState } from "react";
import {
  PICKER_TAB,
  type PickerTab,
  defaultPickerTab,
  readPickerTabPreference,
  writePickerTabPreference,
} from "./picker-tab-toggle";

const PICKER_VIEW_CACHE_MAX_ENTRIES = 16;

type HubPickerViewInput = {
  hasDeviceItems: boolean;
  isLoadingDevice: boolean;
};

type HubPickerViewState = {
  activeQuery: string;
  handleQueryChange: (next: string) => void;
  tab: PickerTab;
};

function resolvePickerTab({
  hasDeviceItems,
  hasExplicitTabPreference,
  isLoadingDevice,
  lockedInferredTab,
  online,
  selectedTab,
}: HubPickerViewInput & {
  hasExplicitTabPreference: boolean;
  lockedInferredTab: PickerTab | null;
  online: boolean;
  selectedTab: PickerTab;
}): { shouldForceDeviceTab: boolean; tab: PickerTab } {
  const shouldUseDeviceTab = !online || (!isLoadingDevice && hasDeviceItems);
  const shouldForceDeviceTab = !online && selectedTab === PICKER_TAB.HUB;
  const inferredTab =
    shouldUseDeviceTab && (!hasExplicitTabPreference || shouldForceDeviceTab)
      ? PICKER_TAB.DEVICE
      : selectedTab;
  return {
    shouldForceDeviceTab,
    tab: shouldForceDeviceTab
      ? PICKER_TAB.DEVICE
      : (lockedInferredTab ?? inferredTab),
  };
}

function pickerViewCacheKey({
  activeQuery,
  hasDeviceItems,
  isLoadingDevice,
  shouldLockInferredTab,
  tab,
}: HubPickerViewInput & {
  activeQuery: string;
  shouldLockInferredTab: boolean;
  tab: PickerTab;
}): string {
  return [
    hasDeviceItems ? "1" : "0",
    isLoadingDevice ? "1" : "0",
    tab,
    activeQuery,
    shouldLockInferredTab ? "1" : "0",
  ].join("\0");
}

export function useHubPickerState({
  storageKey,
  hfToken,
  online,
}: {
  storageKey: string;
  hfToken?: string | null;
  online: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [initialTabPreference] = useState(() =>
    readPickerTabPreference(storageKey),
  );
  const [selectedTab, setTabState] = useState<PickerTab>(
    () => initialTabPreference ?? defaultPickerTab(),
  );
  const [lockedInferredTab, setLockedInferredTab] = useState<PickerTab | null>(
    null,
  );
  const [hubQuery, setHubQuery] = useState("");
  const [deviceQuery, setDeviceQuery] = useState("");
  const [hasExplicitTabPreference, setHasExplicitTabPreference] = useState(
    initialTabPreference !== null,
  );

  const debouncedHubQuery = useDebouncedValue(hubQuery);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const viewCacheRef = useRef(new Map<string, HubPickerViewState>());

  const handleTabChange = useCallback(
    (next: PickerTab) => {
      setLockedInferredTab(null);
      setHasExplicitTabPreference(true);
      setTabState(next);
      writePickerTabPreference(storageKey, next);
    },
    [storageKey],
  );

  const closePicker = useCallback(() => {
    setOpen(false);
    setLockedInferredTab(null);
  }, []);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (nextOpen) {
        setOpen(true);
        return;
      }
      closePicker();
    },
    [closePicker],
  );

  const getViewState = useCallback(
    ({
      hasDeviceItems,
      isLoadingDevice,
    }: HubPickerViewInput): HubPickerViewState => {
      const { shouldForceDeviceTab, tab } = resolvePickerTab({
        hasDeviceItems,
        hasExplicitTabPreference,
        isLoadingDevice,
        lockedInferredTab,
        online,
        selectedTab,
      });
      const activeQuery = (
        tab === PICKER_TAB.HUB ? hubQuery : deviceQuery
      ).trim();
      const shouldLockInferredTab =
        !shouldForceDeviceTab &&
        (!hasExplicitTabPreference || tab !== selectedTab);
      const cacheKey = pickerViewCacheKey({
        activeQuery,
        hasDeviceItems,
        isLoadingDevice,
        shouldLockInferredTab,
        tab,
      });
      const cache = viewCacheRef.current;
      const cached = cache.get(cacheKey);
      if (cached) {
        cache.delete(cacheKey);
        cache.set(cacheKey, cached);
        return cached;
      }
      const setQuery = tab === PICKER_TAB.HUB ? setHubQuery : setDeviceQuery;
      const handleQueryChange = (next: string) => {
        if (shouldLockInferredTab) {
          setLockedInferredTab((current) => current ?? tab);
        }
        setQuery(next);
      };
      const view = { activeQuery, handleQueryChange, tab };
      cache.set(cacheKey, view);
      while (cache.size > PICKER_VIEW_CACHE_MAX_ENTRIES) {
        const oldest = cache.keys().next().value;
        if (!oldest) {
          break;
        }
        cache.delete(oldest);
      }
      return view;
    },
    [
      deviceQuery,
      hasExplicitTabPreference,
      hubQuery,
      lockedInferredTab,
      online,
      selectedTab,
    ],
  );

  return {
    closePicker,
    debouncedHfToken,
    debouncedHubQuery,
    deviceQuery,
    getViewState,
    handleOpenChange,
    handleTabChange,
    hubQuery,
    open,
  };
}
