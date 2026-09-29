// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
