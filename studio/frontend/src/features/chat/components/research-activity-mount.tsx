// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense, type ComponentProps } from "react";
import { LazyImportBoundary, LazyImportFailure } from "@/components/lazy-import-boundary";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
const Panel = lazy(() => import("./research-activity-panel").then(module => ({default: module.ResearchActivityPanel})));
type PanelProps = ComponentProps<typeof import("./research-activity-panel").ResearchActivityPanel>;
type SheetProps = ComponentProps<typeof import("./research-activity-panel").ResearchActivitySheet>;

export function ResearchActivityPanel(props: PanelProps) {
  return (
    <LazyImportBoundary key={props.runId} fallback={<LazyImportFailure message="Research activity could not load."
      reloadLabel="Reload" dismissLabel="Close" onDismiss={props.onClose}
      testId="research-activity-load-error" className="p-4" />}>
      <Suspense fallback={<aside aria-label="Research activity" className="p-4">
        <p role="status">Loading research activity…</p><button type="button" onClick={props.onClose}>Close</button>
      </aside>}><Panel {...props} /></Suspense>
    </LazyImportBoundary>
  );
}

export function ResearchActivitySheet({runId,open,onOpenChange}: SheetProps) {
  // Keep the accessible modal shell eager; only the optional editor is deferred.
  return <Sheet open={open} onOpenChange={onOpenChange}>
    <SheetContent side="right" className="w-screen max-w-none p-0 sm:max-w-none" showCloseButton={false}>
      <SheetHeader className="sr-only"><SheetTitle>Deep research</SheetTitle>
        <SheetDescription>Chronological research activity</SheetDescription></SheetHeader>
      {open ? <ResearchActivityPanel key={runId} runId={runId} variant="sheet" onClose={() => onOpenChange(false)} /> : null}
    </SheetContent>
  </Sheet>;
}
