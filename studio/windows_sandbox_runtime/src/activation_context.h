/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#ifndef UNSLOTH_ACTIVATION_CONTEXT_H
#define UNSLOTH_ACTIVATION_CONTEXT_H
#include <windows.h>

#define US_ACTIVATION_LIMIT 64u
#define US_ACTIVATION_BYTES 65536u

/* Caller supplies immutable, broker-approved empty manifests from the final
   protected snapshot. Preparation compares those exact bytes without executing
   the target image. The caller must retain the snapshot directory lease too. */
typedef struct {
    const wchar_t *path;
    const void *manifest;
    DWORD manifest_bytes;
    WORD resource_id;
    WORD language;
} UsActivationInput;

DWORD us_activation_prepare(const UsActivationInput *inputs, size_t count);
/* Install after the drop, before payload execution, with thread creation
   quiescent. The hook, contexts and file pins intentionally last until process
   termination, including Python finalization and DLL detach callbacks. */
DWORD us_activation_install(void);
/* Shared matcher for the hook and native negative controls. NULL means that
   the original API must handle this request. A match transfers one reference. */
HANDLE us_activation_match(PCACTCTXW request);
LONG us_activation_substitutions(void);
#endif
