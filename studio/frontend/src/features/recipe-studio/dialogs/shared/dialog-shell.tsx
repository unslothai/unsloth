import {
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ReactElement } from "react";

export function DialogShell(): ReactElement {
  return (
    <DialogHeader>
      <DialogTitle>Configure block</DialogTitle>
      <DialogDescription>
        Adjust block params before running the flow.
      </DialogDescription>
    </DialogHeader>
  );
}
