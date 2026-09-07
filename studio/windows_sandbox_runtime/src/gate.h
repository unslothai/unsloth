/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#ifndef UNSLOTH_WINDOWS_GATE_H
#define UNSLOTH_WINDOWS_GATE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>

#define US_PROTOCOL_VERSION 1u
#define US_REQUIRED_CHECKS 0x1ffu
#define US_READY 1u
#define US_FAILED 2u
#define US_READY_STAGE 16u

typedef struct {
    uint8_t nonce[32];
    uint8_t profile[32];
    uint8_t content[32];
} UsBinding;

typedef struct {
    char magic[8];
    uint32_t version, phase, pid, checks, error, stage;
    UsBinding binding;
} UsStatus;

typedef struct {
    char magic[8];
    uint32_t version, reserved;
    UsBinding binding;
} UsAcknowledgement;

_Static_assert(sizeof(UsStatus) == 128, "startup status ABI changed");
_Static_assert(sizeof(UsAcknowledgement) == 112, "startup acknowledgement ABI changed");

/* These functions never run payload code, resume threads or approve artifacts. */
BOOL us_validate_startup(PSID package_sid, const wchar_t *aap_path, UsStatus *status);
BOOL us_drop_and_validate(PSID package_sid, const wchar_t *aap_path, UsStatus *status);
BOOL us_send_status(HANDLE output, const UsStatus *status);
BOOL us_wait_acknowledgement(HANDLE input, const UsBinding *binding);
void us_status_init(UsStatus *status, const UsBinding *binding);
BOOL us_fail(UsStatus *status, DWORD stage, DWORD error);

#endif
