import { useEffect, useState } from "react";
import { authFetch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import { DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { useIsolationStore } from "./tool-isolation";

export function IsolationControls() {
  const state = useIsolationStore();
  const [setup, setSetup] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const check = state.check;
  useEffect(() => { void check(); }, [check]);
  async function install(repair = false) {
    setBusy(true);
    try {
      const response = await authFetch(`/api/inference/tool-isolation/windows-setup?repair_existing=${repair}`, { method: "POST" }, { retryNetworkErrors: false });
      const result = await response.json();
      setSetup(result.message ?? result.detail ?? "Setup failed.");
      await state.check(true);
    } catch (error) { setSetup(String(error)); }
    finally { setBusy(false); }
  }
  return <>
    <DropdownMenuSeparator />
    <DropdownMenuLabel>Python and Terminal · Sandbox</DropdownMenuLabel>
    {(["auto", "required"] as const).map(mode => <DropdownMenuItem key={mode} onSelect={event => { event.preventDefault(); state.setMode(mode); }}>
      <span className="flex-1">{mode === "auto" ? "Auto" : "Require sandbox"}</span>{state.mode === mode ? "✓" : null}
    </DropdownMenuItem>)}
    <div className="px-2 py-2 text-xs text-muted-foreground">
      <p>{state.checking ? "Checking" : state.capability?.available ? `${state.capability.backend} available` : "No OS isolation"}</p>
      <p className="mt-1">{state.mode === "auto" ? "Uses OS isolation when available, otherwise software safeguards." : "Python and Terminal will not run without OS isolation."}</p>
      {state.reselect && <p className="mt-1">Your previous mode is no longer supported. Choose Auto or Require sandbox. Required stays on until you choose.</p>}
      <details className="mt-2"><summary>Sandbox details</summary>
        <p className="mt-1">{state.capability?.reason}</p><p>{state.capability?.remediation}</p>
        {state.capability?.environment !== "win32" && <p>Network access is unrestricted.</p>}
        {state.capability?.limitations.map(item => <p key={item}>{item.replaceAll("_", " ")}</p>)}
        {state.capability?.diagnostic && <pre className="whitespace-pre-wrap break-words">{JSON.stringify(state.capability.diagnostic, null, 2)}</pre>}
      </details>
      {state.error && <p role="alert">{state.error}</p>}
      <Button variant="ghost" size="sm" disabled={state.checking || busy} onClick={() => void state.check(true)}>Check again</Button>
      {state.capability?.environment === "win32" && <>
        <p>Windows setup may request administrator approval. It never retries a tool call.</p>
        <Button variant="outline" size="sm" disabled={busy} onClick={() => void install()}>{busy ? "Setting up…" : "Set up Windows sandbox"}</Button>
        {setup?.includes("different network settings") && <Button variant="outline" size="sm" disabled={busy} onClick={() => void install(true)}>Repair Windows sandbox</Button>}
      </>}
      {setup && <p role="status" className="mt-1">{setup}</p>}
    </div>
  </>;
}
