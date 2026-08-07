


import { useDebouncedValue } from "@/hooks";
import { useCallback, useState } from "react";
import {
  type PickerDeviceInventoryState,
  resolveInferredPickerTabLock,
  resolvePickerTab,
} from "./picker-tab-policy";
import {
  PICKER_TAB,
  type PickerTab,
  readPickerTabPreference,
  writePickerTabPreference,
} from "./picker-tab-state";

type PickerViewInput = PickerDeviceInventoryState;

type PickerViewState = {
  activeQuery: string;
  handleOpenChange: (nextOpen: boolean) => void;
  handleQueryChange: (next: string) => void;
  tab: PickerTab;
};

export function usePickerState({
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
    () => initialTabPreference ?? (online ? PICKER_TAB.hub : PICKER_TAB.device),
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

  const getViewState = useCallback(
    ({
      hasDeviceItems,
      isDeviceInventorySettled,
    }: PickerViewInput): PickerViewState => {
      const tab = resolvePickerTab({
        hasDeviceItems,
        hasExplicitTabPreference,
        isDeviceInventorySettled,
        lockedInferredTab,
        online,
        selectedTab,
      });
      const activeQuery = (
        tab === PICKER_TAB.hub ? hubQuery : deviceQuery
      ).trim();
      const shouldLockInferredTab =
        !hasExplicitTabPreference || tab !== selectedTab;
      const setQuery = tab === PICKER_TAB.hub ? setHubQuery : setDeviceQuery;
      const handleQueryChange = (next: string) => {
        if (shouldLockInferredTab) {
          setLockedInferredTab((current) => current ?? tab);
        }
        setQuery(next);
      };
      const handleOpenChange = (nextOpen: boolean) => {
        if (nextOpen) {
          const inferredTab = resolveInferredPickerTabLock({
            hasDeviceItems,
            hasExplicitTabPreference,
            isDeviceInventorySettled,
            online,
            selectedTab,
          });
          if (inferredTab !== null) {
            setLockedInferredTab(inferredTab);
          }
          setOpen(true);
          return;
        }
        closePicker();
      };
      return { activeQuery, handleOpenChange, handleQueryChange, tab };
    },
    [
      deviceQuery,
      closePicker,
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
    handleTabChange,
    hubQuery,
    open,
  };
}
