// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense, useEffect, useState, type ComponentProps } from "react";
import { LazyImportBoundary, LazyImportFailure } from "@/components/lazy-import-boundary";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
const ProjectDialog = lazy(() => import("./new-project-dialog").then(module => ({default: module.NewProjectDialog})));
type Props = ComponentProps<typeof import("./new-project-dialog").NewProjectDialog>;

function PendingDialog({props,failed=false}: {props: Props; failed?: boolean}) {
  return <Dialog open={props.open} onOpenChange={props.onOpenChange}>
    <DialogContent><DialogHeader><DialogTitle>{props.title ?? "Create project"}</DialogTitle>
      <DialogDescription>{failed ? "The project editor could not load." : "Loading project editor…"}</DialogDescription></DialogHeader>
      {failed ? <LazyImportFailure message="Reload to retry loading the project editor." reloadLabel="Reload"
        dismissLabel="Cancel" onDismiss={() => props.onOpenChange(false)} testId="project-editor-load-error" className="p-2" /> :
        <button type="button" onClick={() => props.onOpenChange(false)}>Cancel</button>}
    </DialogContent>
  </Dialog>;
}

export function NewProjectDialog(props: Props) {
  const [activated,setActivated] = useState(props.open);
  useEffect(() => { if (props.open) setActivated(true); }, [props.open]);
  // After first use, keep the original controller mounted across close/reopen.
  // Outstanding native folder leases and create-operation cleanup retain their owner.
  if (!activated && !props.open) return null;
  return <LazyImportBoundary fallback={<PendingDialog props={props} failed />}>
    <Suspense fallback={<PendingDialog props={props} />}><ProjectDialog {...props} /></Suspense>
  </LazyImportBoundary>;
}
