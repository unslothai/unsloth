import { isolationLimitation } from "./isolation-labels";
import { create } from "zustand";
import { useContext } from "react";
import { useAuiState } from "@assistant-ui/react";
import { toolOutputKey, ToolPaneScopeContext } from "./tool-output-scope";

export const useExecutionRecords = create<{ records: Record<string, string> }>(() => ({ records: {} }));
export function executionRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object") return;
  const record = value as Record<string, unknown>;
  if (typeof record.os_isolation !== "boolean" || typeof record.backend !== "string" || typeof record.network_policy !== "string") return;
  return record;
}
export function recordExecution(key: string, value: unknown) {
  const record = executionRecord(value);
  if (!record) return;
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
  const saved = useAuiState(s => s.message.metadata.custom.toolExecutions) as Record<string, unknown> | undefined;
  const paneScope = useContext(ToolPaneScopeContext);
  const text = records[toolOutputKey(paneScope, toolCallId)];
  const record = executionRecord(saved?.[toolCallId]) ?? (text ? executionRecord(JSON.parse(text)) : undefined);
  if (!record) return null;
  const limitations = Array.isArray(record.limitations)
    ? record.limitations.filter((item: string) => item !== "srt_windows_system_dns_unfenced" && item !== "srt_windows_shared_account_grants")
    : [];
  const label = record.os_isolation ? `Sandbox · ${record.backend}` : record.effective_mode === "full" ? "Full access · No OS isolation" : "No OS isolation";
  if (record.network_policy !== "unrestricted" && limitations.length === 0) {
    return <p className="mx-4 mb-2 text-xs text-muted-foreground" data-slot="tool-execution-protection">{label}</p>;
  }
  return <details className="mx-4 mb-2 text-xs text-muted-foreground" data-slot="tool-execution-protection">
    <summary>{label}</summary>
    {record.network_policy === "unrestricted" && <p>Network: unrestricted</p>}
    {limitations.map((item: string) => <p key={item}>{isolationLimitation(item)}</p>)}
  </details>;
}
