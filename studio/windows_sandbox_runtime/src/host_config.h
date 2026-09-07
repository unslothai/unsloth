/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#ifndef UNSLOTH_HOST_CONFIG_H
#define UNSLOTH_HOST_CONFIG_H
#include "gate.h"

#define US_CONFIG_BYTES 65536u
#define US_EXPANDED_CONFIG_BYTES (US_CONFIG_BYTES + 1048576u + 4u)
#define US_CONFIG_LIST 64u
enum UsPath {
    US_RUNTIME_DLL, US_STDLIB, US_NATIVE_DIR, US_RUNTIME_HOME, US_EXECUTABLE,
    US_PREFIX, US_BASE_PREFIX, US_POLICY, US_SHIM, US_WORKDIR, US_SCRIPT,
    US_TEMP, US_AAP, US_PACKAGE_SID, US_CONFIG_FIELDS
};

typedef struct {
    char magic[8];
    uint32_t version, bytes, major, minor, patch, packages, arguments, images;
    UsBinding binding;
} UsConfigHeader;
_Static_assert(sizeof(UsConfigHeader) == 136, "host configuration ABI changed");

typedef struct {
    UsConfigHeader header;
    wchar_t *values[US_CONFIG_FIELDS + 3 * US_CONFIG_LIST];
    BYTE *activation_plan;
    DWORD activation_bytes;
} UsConfig;

BOOL us_read_config(HANDLE input, UsConfig *config);
void us_free_config(UsConfig *config);
HANDLE us_handle_argument(const wchar_t *text);
#endif
