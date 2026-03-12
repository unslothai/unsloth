// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils"
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading03Icon } from "@hugeicons/core-free-icons"

function Spinner({ className }: { className?: string }) {
  return (
    <HugeiconsIcon icon={Loading03Icon} strokeWidth={2} role="status" aria-label="Loading" className={cn("size-4 animate-spin", className)} />
  )
}

export { Spinner }
