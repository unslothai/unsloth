/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#ifndef UNSLOTH_AUTHORITY_AUDIT_H
#define UNSLOTH_AUTHORITY_AUDIT_H
#include <windows.h>

#define US_AUTHORITY_MAX_HANDLES 2048u
#define US_AUTHORITY_MAX_TYPES 32u
#define US_AUTHORITY_TYPE_CHARS 64u

typedef struct {
    wchar_t name[US_AUTHORITY_TYPE_CHARS];
    DWORD count;
} UsAuthorityType;

typedef struct {
    ULONG_PTR handle;
    DWORD granted_access, type_index, owner_pid;
} UsAuthorityHandle;

typedef struct {
    DWORD version, error, native_status, handles, private_keys, foreign_keys;
    DWORD tokens, foreign_processes, foreign_threads, other_handles, type_count;
    BOOL stable_inventory;
    UsAuthorityType types[US_AUTHORITY_MAX_TYPES];
    UsAuthorityHandle inventory[US_AUTHORITY_MAX_HANDLES];
} UsAuthorityReport;

/* Metadata only. Caller supplies the already identity-validated APPKEY root
   and guarantees trusted prepayload handle/thread quiescence. Rejects retained
   foreign keys, tokens and foreign process/thread handles. Other object types
   are inventoried, NOT qualified. No discovered handle is closed or used for I/O.
   PSS enumeration is public; dynamic NtQueryObject/NtQueryKey metadata contracts
   still require OS-version qualification. Missing/failed queries fail closed. */
DWORD us_authority_audit(HKEY own_catalog, UsAuthorityReport *report);
#endif
