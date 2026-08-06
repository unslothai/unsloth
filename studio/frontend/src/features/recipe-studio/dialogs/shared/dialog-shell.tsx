


import {
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ReactElement } from "react";

type DialogShellProps = {
  title?: string;
  description?: string;
};

export function DialogShell({
  title = "Edit step",
  description = "Update this step before you run the recipe.",
}: DialogShellProps): ReactElement {
  return (
    <DialogHeader>
      <DialogTitle>{title}</DialogTitle>
      <DialogDescription>{description}</DialogDescription>
    </DialogHeader>
  );
}
