// SPDX-License-Identifier: AGPL-3.0-only
import { type ComponentProps, lazy, Suspense } from "react";
import { LazyImportBoundary } from "@/components/lazy-import-boundary";

type Props = ComponentProps<typeof import("./release-notes-panel").ReleaseNotesPanel>;
const Notes = lazy(() => import("./release-notes-panel").then((module) => ({
  default: module.ReleaseNotesPanel,
})));

export function ReleaseNotesPanel(props: Props) {
  // The popup owns visibility. `open` means expanded notes, not an open popup:
  // collapsed previews must still load. Update checks/actions stay in their owner.
  return (
    <LazyImportBoundary fallback={
      <div role="alert" className="px-1 py-2 text-ui-11 text-muted-foreground" data-testid="release-notes-load-error">
        <p>Release notes could not load. Update controls remain available.</p>
        <button type="button" onClick={() => window.location.reload()} className="mt-1 underline">Reload to retry notes</button>
      </div>
    }>
      <Suspense fallback={props.open ? <p role="status" className="px-1 py-2 text-ui-11 text-muted-foreground">Loading release notes…</p> : null}>
        <Notes {...props} />
      </Suspense>
    </LazyImportBoundary>
  );
}
