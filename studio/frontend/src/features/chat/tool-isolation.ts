import { create } from "zustand";
import { authFetch } from "@/features/auth";

export type IsolationMode = "auto" | "required";
export type IsolationCapability = {
  available: boolean; environment: string; backend: string; reason: string;
  remediation: string; limitations: string[]; diagnostic?: Record<string, unknown>;
};
const storageKey = "unsloth_tool_execution_mode";
export function migrateIsolationMode(value: unknown): { mode: IsolationMode; reselect: boolean } {
  if (value == null || value === "auto") return { mode: "auto", reselect: false };
  if (value === "required" || value === "os_isolation_required") return { mode: "required", reselect: false };
  return { mode: "required", reselect: true };
}
function initialMode() {
  try {
    const selection = migrateIsolationMode(localStorage.getItem(storageKey));
    for (const key of ["unsloth_limited_tool_grant", "unsloth_nested_tool_grant"]) localStorage.removeItem(key);
    return selection;
  } catch { return { mode: "auto" as const, reselect: false }; }
}
let flight: Promise<void> | undefined;
let forcedRefresh: Promise<void> | undefined;
export const useIsolationStore = create<{
  mode: IsolationMode; reselect: boolean; capability: IsolationCapability | null;
  checking: boolean; error: string | null; setMode: (mode: IsolationMode) => void;
  check: (force?: boolean) => Promise<void>;
}>(() => ({
  ...initialMode(), capability: null, checking: false, error: null,
  setMode(mode) {
    try { localStorage.setItem(storageKey, mode); } catch { /* Keep the current selection in memory. */ }
    useIsolationStore.setState({ mode, reselect: false });
  },
  check(force = false): Promise<void> {
    if (flight) {
      if (!force) return flight;
      forcedRefresh ??= flight
        .then(() => useIsolationStore.getState().check(true))
        .finally(() => { forcedRefresh = undefined; });
      return forcedRefresh;
    }
    useIsolationStore.setState({ checking: true, error: null });
    flight = (async () => {
      try {
        const response = await authFetch(`/api/inference/tool-isolation/capability?force=${force}`);
        if (!response.ok) throw new Error("Could not check Python and Terminal isolation.");
        useIsolationStore.setState({ capability: await response.json() });
      } catch (error) { useIsolationStore.setState({ error: String(error) }); }
      finally { useIsolationStore.setState({ checking: false }); flight = undefined; }
    })();
    return flight;
  },
}));
