import { isolationLimitation } from "./isolation-labels";
import { create } from "zustand";
import { useContext } from "react";
import { toolOutputKey, ToolPaneScopeContext } from "./tool-output-scope";

export const useExecutionRecords = create<{ records: Record<string, string> }>(() => ({ records: {} }));
export function recordExecution(key: string, value: unknown) {
  if (!value || typeof value !== "object") return;
  const record = value as Record<string, unknown>;
  if (typeof record.os_isolation !== "boolean" || typeof record.backend !== "string" || typeof record.network_policy !== "string") return;
  const records = { ...useExecutionRecords.getState().records, [key]: JSON.stringify(record) };
  while (Object.keys(records).length > 512) delete records[Object.keys(records)[0]];
  useExecutionRecords.setState({ records });
}
export function clearExecution(key: string) {
  const records = { ...useExecutionRecords.getState().records };
  delete records[key];
  useExecutionRecords.setState({ records });
}
export function ToolExecutionDetails({ toolCallId }: { toolCallId: string }) {
  const records = useExecutionRecords(s => s.records);
  const paneScope = useContext(ToolPaneScopeContext);
  const text = records[toolOutputKey(paneScope, toolCallId)];
  if (!text) return null;
  const record = JSON.parse(text);
  return <details className="mx-4 mb-2 text-xs text-muted-foreground" data-slot="tool-execution-protection">
    <summary>{record.os_isolation ? `Sandbox · ${record.backend}` : record.effective_mode === "full" ? "Full access · No OS isolation" : "No OS isolation"}</summary>
    <p>Network: {record.network_policy === "unrestricted" ? "unrestricted" : "SRT deny policy; system DNS remains available"}</p>
    {Array.isArray(record.limitations) && record.limitations.map((item: string) => <p key={item}>{isolationLimitation(item)}</p>)}
  </details>;
}
