/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#define WIN32_LEAN_AND_MEAN
#include "authority_audit.h"
#include <stdio.h>
#include <wchar.h>

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) return 90;
    HKEY catalog = NULL, controlled = NULL;
    HANDLE token = NULL;
    HANDLE events[US_AUTHORITY_MAX_HANDLES] = {0};
    DWORD error = (DWORD)RegLoadAppKeyW(argv[2], &catalog, KEY_QUERY_VALUE, 0, 0);
    if (error) return 91;
    BOOL positive = FALSE;
    if (!wcscmp(argv[1], L"host-key")) {
        error = (DWORD)RegOpenKeyExW(HKEY_CURRENT_USER, L"Environment", 0, KEY_QUERY_VALUE, &controlled);
        if (!error) {
            DWORD keys = 0;
            positive = RegQueryInfoKeyW(controlled, NULL, NULL, NULL, &keys, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == ERROR_SUCCESS;
        }
    } else if (!wcscmp(argv[1], L"other-hive")) {
        if (argc != 4) error = ERROR_INVALID_PARAMETER;
        else error = (DWORD)RegLoadAppKeyW(argv[3], &controlled, KEY_QUERY_VALUE, 0, 0);
        positive = !error;
    } else if (!wcscmp(argv[1], L"token")) {
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) error = GetLastError();
        positive = !error;
    } else if (!wcscmp(argv[1], L"foreign-process")) {
        if (argc != 4) error = ERROR_INVALID_PARAMETER;
        else {
            token = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, wcstoul(argv[3], NULL, 10));
            if (!token) error = GetLastError();
            positive = !error;
        }
    } else if (!wcscmp(argv[1], L"handle-limit")) {
        for (DWORD i = 0; i < US_AUTHORITY_MAX_HANDLES; ++i) {
            events[i] = CreateEventW(NULL, TRUE, FALSE, NULL);
            if (!events[i]) { error = GetLastError(); break; }
        }
    } else if (wcscmp(argv[1], L"clean") && wcscmp(argv[1], L"invalid-root")) error = ERROR_INVALID_PARAMETER;
    if (error) return 92;
    if (RegDisablePredefinedCacheEx() != ERROR_SUCCESS) return 93;
    UsAuthorityReport report;
    error = us_authority_audit(!wcscmp(argv[1], L"invalid-root") ? (HKEY)(ULONG_PTR)0x12345678 : catalog, &report);
    /* Owned positive controls remain valid: the audit must never close them. */
    BOOL retained = TRUE;
    if (controlled) {
        DWORD keys = 0;
        retained = RegQueryInfoKeyW(controlled, NULL, NULL, NULL, &keys, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == ERROR_SUCCESS;
        RegCloseKey(controlled);
    }
    if (token) CloseHandle(token);
    for (DWORD i = 0; i < US_AUTHORITY_MAX_HANDLES; ++i) if (events[i]) CloseHandle(events[i]);
    RegCloseKey(catalog);
    printf("error=%lu native=%lu handles=%lu private=%lu foreign=%lu tokens=%lu other=%lu stable=%d positive=%d retained=%d foreign_processes=%lu\n",
        error, report.native_status, report.handles, report.private_keys, report.foreign_keys,
        report.tokens, report.other_handles, report.stable_inventory, positive, retained, report.foreign_processes);
    for (DWORD i = 0; i < report.type_count; ++i)
        printf("type=%ls count=%lu\n", report.types[i].name, report.types[i].count);
    return 0;
}
