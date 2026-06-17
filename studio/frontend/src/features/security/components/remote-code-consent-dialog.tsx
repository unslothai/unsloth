// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  SecurityWarningIcon,
  ShieldBanIcon,
  SourceCodeSquareIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { severityTone } from "../lib/severity-tone";
import { useRemoteCodeConsentDialogStore } from "../stores/remote-code-consent-dialog-store";
import type {
  RemoteCodeFinding,
  RemoteCodeSeverity,
  RemoteCodeSnippetRow,
  UnsafeFile,
} from "../types";

/** Background tint for the flagged line + matched span, by severity. */
function matchTone(severity: RemoteCodeSeverity | string): {
  row: string;
  span: string;
} {
  switch (severity) {
    case "CRITICAL":
    case "HIGH":
      return {
        row: "bg-red-500/10 border-l-2 border-red-500/60",
        span: "bg-red-500/25 text-foreground",
      };
    case "MEDIUM":
      return {
        row: "bg-amber-500/10 border-l-2 border-amber-500/60",
        span: "bg-amber-500/25 text-foreground",
      };
    default:
      return {
        row: "bg-muted border-l-2 border-border",
        span: "bg-foreground/15 text-foreground",
      };
  }
}

/** One code line: line-number gutter + source, with the match span highlighted. */
function SnippetLine({
  row,
  severity,
}: {
  row: RemoteCodeSnippetRow;
  severity: RemoteCodeSeverity;
}) {
  const tone = matchTone(severity);
  const hasSpan =
    row.isMatch &&
    typeof row.matchStart === "number" &&
    typeof row.matchEnd === "number" &&
    row.matchEnd > row.matchStart;

  let body;
  if (hasSpan) {
    const start = row.matchStart as number;
    const end = row.matchEnd as number;
    body = (
      <>
        {row.text.slice(0, start)}
        <span className={cn("rounded-[3px] px-0.5 font-semibold", tone.span)}>
          {row.text.slice(start, end)}
        </span>
        {row.text.slice(end)}
      </>
    );
  } else {
    body = row.text || " ";
  }

  return (
    <div
      className={cn(
        "flex",
        row.isMatch ? tone.row : "border-l-2 border-transparent",
      )}
    >
      <span className="w-10 shrink-0 select-none pr-3 text-right tabular-nums text-muted-foreground/50">
        {row.number}
      </span>
      <code
        className={cn(
          "whitespace-pre pr-3",
          row.isMatch ? "text-foreground" : "text-muted-foreground",
        )}
      >
        {body}
      </code>
    </div>
  );
}

function UnsafeFileCard({ file }: { file: UnsafeFile }) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border bg-muted/40 px-3 py-2">
      <Badge
        variant="outline"
        className={cn(
          "shrink-0 text-[10px] font-semibold uppercase tracking-wide",
          severityTone("CRITICAL"),
        )}
      >
        {file.level}
      </Badge>
      <span className="flex min-w-0 items-center gap-1 font-mono text-xs text-muted-foreground">
        <HugeiconsIcon icon={SourceCodeSquareIcon} className="size-3.5 shrink-0" />
        <span className="truncate">{file.path}</span>
      </span>
    </div>
  );
}

function FindingCard({ finding }: { finding: RemoteCodeFinding }) {
  const fileLabel = finding.line ? `${finding.file}:${finding.line}` : finding.file;
  const snippet = finding.snippet ?? [];
  return (
    <div className="overflow-hidden rounded-lg border">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b bg-muted/40 px-3 py-2">
        <Badge
          variant="outline"
          className={cn(
            "shrink-0 text-[10px] font-semibold tracking-wide",
            severityTone(finding.severity),
          )}
        >
          {finding.severity}
        </Badge>
        <span className="text-sm font-medium">{finding.check}</span>
        <span className="ml-auto flex shrink-0 items-center gap-1 font-mono text-xs text-muted-foreground">
          <HugeiconsIcon icon={SourceCodeSquareIcon} className="size-3.5" />
          {fileLabel}
        </span>
      </div>
      {snippet.length > 0 ? (
        <div className="overflow-x-auto bg-background py-2 font-mono text-xs leading-relaxed">
          {snippet.map((row) => (
            <SnippetLine key={row.number} row={row} severity={finding.severity} />
          ))}
        </div>
      ) : finding.evidence ? (
        <div className="px-3 py-2 font-mono text-xs text-muted-foreground">
          {finding.evidence}
        </div>
      ) : null}
    </div>
  );
}

/**
 * App-wide consent dialog for loads that need trust_remote_code. Shows the
 * security scan findings with the flagged code in context; CRITICAL is a hard
 * block (no Enable). Mounted once in the root layout; driven by
 * useRemoteCodeConsentDialogStore.
 */
export function RemoteCodeConsentDialog() {
  const open = useRemoteCodeConsentDialogStore((s) => s.open);
  const scan = useRemoteCodeConsentDialogStore((s) => s.scan);
  const resolve = useRemoteCodeConsentDialogStore((s) => s.resolve);

  const displayName = scan?.modelName?.split("/").pop() || "This model";
  const blocked = scan ? !scan.approvable : false;
  const findings = scan?.findings ?? [];
  const unsafeFiles = scan?.unsafeFiles ?? [];
  // Malware (an unsafe serialized file flagged by Hugging Face's scan) is a hard
  // block with its own copy, distinct from a CRITICAL custom-code finding.
  const malware = unsafeFiles.length > 0;

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) resolve(false);
      }}
    >
      <AlertDialogContent className="max-w-2xl">
        <AlertDialogHeader>
          <div className="flex items-start gap-3">
            <div
              className={cn(
                "flex size-9 shrink-0 items-center justify-center rounded-full",
                blocked
                  ? "bg-red-500/10 text-red-600 dark:text-red-400"
                  : "bg-amber-500/10 text-amber-600 dark:text-amber-400",
              )}
            >
              <HugeiconsIcon
                icon={blocked ? ShieldBanIcon : SecurityWarningIcon}
                className="size-5"
              />
            </div>
            <div className="min-w-0 space-y-1">
              <AlertDialogTitle>
                {malware
                  ? "Unsafe files detected"
                  : blocked
                    ? "Custom code blocked"
                    : "Enable custom code for this model?"}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {malware ? (
                  <>
                    <span className="font-medium text-foreground">
                      {displayName}
                    </span>{" "}
                    contains files that Hugging Face's security scan flagged as
                    unsafe (for example, a malicious pickle that would run code
                    when the model loads). It cannot be loaded. The flagged files
                    were never downloaded.
                  </>
                ) : (
                  <>
                    <span className="font-medium text-foreground">
                      {displayName}
                    </span>{" "}
                    declares custom Python code in its repository.{" "}
                    {blocked
                      ? "A security scan flagged CRITICAL issues, so it cannot be enabled."
                      : findings.length > 0
                        ? "Review the security scan below. Continue only if you trust the model source."
                        : "Continue only if you trust the model source."}
                  </>
                )}
              </AlertDialogDescription>
            </div>
          </div>
        </AlertDialogHeader>

        {malware ? (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">
              Flagged by Hugging Face's security scan:
            </p>
            <div className="max-h-[14rem] space-y-2 overflow-y-auto pr-1">
              {unsafeFiles.map((f, i) => (
                <UnsafeFileCard key={`${f.path}-${i}`} file={f} />
              ))}
            </div>
          </div>
        ) : null}

        {findings.length > 0 ? (
          <div className="max-h-[22rem] space-y-3 overflow-y-auto pr-1">
            {findings.map((f, i) => (
              <FindingCard key={i} finding={f} />
            ))}
          </div>
        ) : null}

        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          {blocked ? null : (
            <AlertDialogAction onClick={() => resolve(true)}>
              Enable and continue
            </AlertDialogAction>
          )}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
