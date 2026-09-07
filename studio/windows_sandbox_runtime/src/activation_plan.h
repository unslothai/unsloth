/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#ifndef UNSLOTH_ACTIVATION_PLAN_H
#define UNSLOTH_ACTIVATION_PLAN_H
#include "gate.h"

/* Trusted broker channel only. Caller holds the protected generation directory
   lease through process exit. No image initializers execute in this function. */
DWORD us_activation_plan_prepare(const BYTE *blob, DWORD bytes,
    const UsBinding *binding, const wchar_t *runtime_home, DWORD *count);
/* Bounded startup diagnostics: 1=wire, 2=provider, 3=entry, 4=path,
   5=manifest hash, 6=file open, 7=metadata, 8=final path, 9=image hash,
   10=context adapter, 11=complete. No paths or payload bytes are exposed. */
DWORD us_activation_plan_diagnostic(void);
#endif
