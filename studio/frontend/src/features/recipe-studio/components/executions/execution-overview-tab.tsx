import type { ReactElement, RefObject, UIEvent } from "react";
import {
  Database01Icon,
  Database02Icon,
  Flag02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import type { RecipeExecutionRecord } from "../../execution-types";
import type { ModelUsageRow } from "./executions-view-helpers";
import { formatMetricValue } from "./executions-view-helpers";

type ExecutionOverviewTabProps = {
  execution: RecipeExecutionRecord;
  showSummaryCards: boolean;
  recordsMetric: number | null;
  totalMetric: number | null;
  runDuration: string;
  columnCount: number;
  llmColumnCount: number;
  nullRate: number | null;
  sideEffects: string[];
  lowUniquenessColumns: string[];
  modelUsageRows: ModelUsageRow[];
  totalInputTokens: number;
  totalOutputTokens: number;
  terminalLines: string[];
  terminalRef: RefObject<HTMLDivElement | null>;
  onTerminalScroll: (event: UIEvent<HTMLDivElement>) => void;
};

export function ExecutionOverviewTab({
  execution,
  showSummaryCards,
  recordsMetric,
  totalMetric,
  runDuration,
  columnCount,
  llmColumnCount,
  nullRate,
  sideEffects,
  lowUniquenessColumns,
  modelUsageRows,
  totalInputTokens,
  totalOutputTokens,
  terminalLines,
  terminalRef,
  onTerminalScroll,
}: ExecutionOverviewTabProps): ReactElement {
  return (
    <div className="mt-3 space-y-3">
      {showSummaryCards && (
        <div className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="h-full rounded-lg bg-muted/20 p-3">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Run summary</p>
                <HugeiconsIcon
                  icon={Database01Icon}
                  className="size-4 text-muted-foreground"
                />
              </div>
              <div className="space-y-1 text-xs">
                <p>
                  Records:{" "}
                  <span className="font-semibold">
                    {formatMetricValue(recordsMetric)} / {formatMetricValue(totalMetric)}
                  </span>
                </p>
                <p>
                  Duration: <span className="font-semibold">{runDuration}</span>
                </p>
                <p>
                  Columns analyzed:{" "}
                  <span className="font-semibold">{formatMetricValue(columnCount)}</span>
                </p>
                <p>
                  Final stage:{" "}
                  <span className="font-semibold">{execution.stage ?? "--"}</span>
                </p>
              </div>
            </div>
            <div className="h-full rounded-lg bg-muted/20 p-3">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Insights</p>
                <HugeiconsIcon
                  icon={Database02Icon}
                  className="size-4 text-muted-foreground"
                />
              </div>
              <div className="space-y-2 text-xs">
                <p>
                  LLM columns:{" "}
                  <span className="font-semibold">{formatMetricValue(llmColumnCount)}</span>
                </p>
                <p>
                  Null rate: <span className="font-semibold">{nullRate?.toFixed(1) ?? "--"}%</span>
                </p>
                <p>
                  Dropped columns:{" "}
                  <span className="font-semibold">{formatMetricValue(sideEffects.length)}</span>
                </p>
                {sideEffects.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {sideEffects.map((name) => (
                      <Badge key={name} variant="outline">
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
                <p>
                  Low uniqueness flags:{" "}
                  <span className="font-semibold">
                    {formatMetricValue(lowUniquenessColumns.length)}
                  </span>
                </p>
                {lowUniquenessColumns.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {lowUniquenessColumns.slice(0, 3).map((name) => (
                      <Badge key={name} variant="secondary">
                        {name}
                      </Badge>
                    ))}
                    {lowUniquenessColumns.length > 3 && (
                      <Badge variant="secondary">
                        +{lowUniquenessColumns.length - 3} more
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="rounded-lg bg-muted/20 p-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs text-muted-foreground">Model usage</p>
              <HugeiconsIcon icon={Flag02Icon} className="size-4 text-muted-foreground" />
            </div>
            {modelUsageRows.length === 0 ? (
              <p className="text-xs text-muted-foreground">No model usage yet.</p>
            ) : (
              <div className="space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded border bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground">Total input</p>
                    <p className="text-sm font-semibold">
                      {formatMetricValue(totalInputTokens)}
                    </p>
                  </div>
                  <div className="rounded border bg-muted/30 px-2 py-1.5">
                    <p className="text-muted-foreground">Total output</p>
                    <p className="text-sm font-semibold">
                      {formatMetricValue(totalOutputTokens)}
                    </p>
                  </div>
                </div>
                <div className="overflow-hidden rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Model</TableHead>
                        <TableHead className="text-right">Input</TableHead>
                        <TableHead className="text-right">Output</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {modelUsageRows.map((usage) => (
                        <TableRow key={usage.model}>
                          <TableCell className="max-w-[320px] truncate">{usage.model}</TableCell>
                          <TableCell className="text-right">
                            {formatMetricValue(usage.input)}
                          </TableCell>
                          <TableCell className="text-right">
                            {formatMetricValue(usage.output)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      <div className="overflow-hidden rounded-xl corner-squircle border">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <p className="text-sm font-semibold">Terminal output</p>
          <p className="text-xs text-muted-foreground">{terminalLines.length} lines</p>
        </div>
        <div
          ref={terminalRef}
          className="max-h-72 overflow-auto bg-zinc-900/80 px-3 py-2 font-mono text-xs text-zinc-200"
          onScroll={onTerminalScroll}
        >
          {terminalLines.length === 0 ? (
            <p className="text-zinc-400">
              {isExecutionInProgress(execution.status)
                ? "Waiting for logs..."
                : "No logs captured."}
            </p>
          ) : (
            terminalLines.map((line, index) => (
              <p
                key={`${index}-${line.slice(0, 24)}`}
                className="whitespace-pre-wrap break-words leading-relaxed"
              >
                {line}
              </p>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
