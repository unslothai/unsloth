// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense, useEffect, useState } from "react";
import { LazyImportBoundary, LazyImportFailure } from "@/components/lazy-import-boundary";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { ChatMcpServersDialogProps } from "./chat-mcp-servers-dialog";

const McpEditor = lazy(() => import("./chat-mcp-servers-dialog").then((module) => ({
  default: module.ChatMcpServersDialog,
})));

function PendingEditor({ open, onOpenChange, failed = false }: ChatMcpServersDialogProps & { failed?: boolean }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>MCP Servers</DialogTitle>
          <DialogDescription>{failed ? "The MCP editor could not load." : "Loading MCP editor…"}</DialogDescription>
        </DialogHeader>
        {failed ? (
          <LazyImportFailure message="Reload to retry loading the MCP editor." reloadLabel="Reload"
            dismissLabel="Cancel" onDismiss={() => onOpenChange(false)}
            testId="mcp-editor-load-error" className="p-2" />
        ) : (
          <div><p role="status">Loading MCP editor…</p>
            <button type="button" onClick={() => onOpenChange(false)}>Cancel</button></div>
        )}
      </DialogContent>
    </Dialog>
  );
}

export function ChatMcpServersDialog(props: ChatMcpServersDialogProps) {
  const [activated, setActivated] = useState(props.open);
  useEffect(() => { if (props.open) setActivated(true); }, [props.open]);
  // Load only on first use, but retain the editor across subsequent closes.
  // Pending mutations and their ownership/generation guards must not be reset.
  if (!activated && !props.open) return null;
  return (
    <LazyImportBoundary fallback={<PendingEditor {...props} failed />}>
      <Suspense fallback={<PendingEditor {...props} />}>
        <McpEditor {...props} />
      </Suspense>
    </LazyImportBoundary>
  );
}
