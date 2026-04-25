// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client"

import { cn } from "@/lib/utils"
import { HugeiconsIcon } from "@hugeicons/react"
import { Loading03Icon } from "@hugeicons/core-free-icons"
import { useI18nStore, translate } from "@/features/i18n/store"

function Spinner({ className }: { className?: string }) {
  const locale = useI18nStore((s) => s.locale)
  return (
    <HugeiconsIcon icon={Loading03Icon} strokeWidth={2} role="status" aria-label={translate(locale, "common.loading")} className={cn("size-4 animate-spin", className)} />
  )
}

export { Spinner }
