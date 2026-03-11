// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

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
  title = "Configure block",
  description = "Adjust block params before running the flow.",
}: DialogShellProps): ReactElement {
  return (
    <DialogHeader>
      <DialogTitle>{title}</DialogTitle>
      <DialogDescription>{description}</DialogDescription>
    </DialogHeader>
  );
}
