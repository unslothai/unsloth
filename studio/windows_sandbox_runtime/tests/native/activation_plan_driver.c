/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#include "activation_plan.h"
#include "activation_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DWORD prepares, pin_denials;
static DWORD normalized_denials, dos_denials, opened_queries;

DWORD us_activation_test_path_query(HANDLE file, LPWSTR path, DWORD chars, DWORD flags) {
    if (!(flags & FILE_NAME_OPENED)) {
        ++normalized_denials;
        SetLastError(ERROR_ACCESS_DENIED);
        return 0;
    }
    if (!(flags & VOLUME_NAME_NONE)) {
        ++dos_denials;
        SetLastError(ERROR_ACCESS_DENIED);
        return 0;
    }
    ++opened_queries;
    return GetFinalPathNameByHandleW(file, path, chars, flags);
}

/* A test seam observes the parser/adapter handoff without installing a hook or
   creating contexts. Real adapter resource comparisons have separate controls. */
DWORD us_activation_prepare(const UsActivationInput *inputs, size_t count) {
    ++prepares;
    for (size_t i = 0; i < count; ++i) {
        HANDLE mutation = CreateFileW(inputs[i].path, GENERIC_WRITE, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (mutation != INVALID_HANDLE_VALUE) { CloseHandle(mutation); return ERROR_INVALID_STATE; }
        if (GetLastError() != ERROR_SHARING_VIOLATION) return ERROR_INVALID_STATE;
        ++pin_denials;
        HMODULE resources = LoadLibraryExW(inputs[i].path, NULL,
            LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
        if (!resources) return GetLastError();
        HRSRC entry = FindResourceExW(resources, RT_MANIFEST,
            MAKEINTRESOURCEW(inputs[i].resource_id), inputs[i].language);
        HGLOBAL loaded = entry ? LoadResource(resources, entry) : NULL;
        const void *data = loaded ? LockResource(loaded) : NULL;
        BOOL same = data && SizeofResource(resources, entry) == inputs[i].manifest_bytes
            && !memcmp(data, inputs[i].manifest, inputs[i].manifest_bytes);
        FreeLibrary(resources);
        if (!same) return ERROR_INVALID_DATA;
    }
    return ERROR_SUCCESS;
}

int wmain(int argc, wchar_t **argv) {
    if (argc != 3 && argc != 4) return 2;
    if (argc == 4) {
        if (wcscmp(argv[3], L"normalized-denied")) return 2;
        if (us_activation_test_path_query(INVALID_HANDLE_VALUE, NULL, 0, FILE_NAME_NORMALIZED)
            || GetLastError() != ERROR_ACCESS_DENIED) return 2;
        if (us_activation_test_path_query(INVALID_HANDLE_VALUE, NULL, 0, FILE_NAME_OPENED | VOLUME_NAME_DOS)
            || GetLastError() != ERROR_ACCESS_DENIED) return 2;
    }
    HANDLE file = CreateFileW(argv[1], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (file == INVALID_HANDLE_VALUE) return 3;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0 || size.QuadPart > 1048576) return 4;
    BYTE *data = (BYTE *)malloc((size_t)size.QuadPart + 1);
    DWORD read = 0;
    if (!data || !ReadFile(file, data, (DWORD)size.QuadPart, &read, NULL) || read != size.QuadPart) return 5;
    CloseHandle(file);
    UsBinding binding;
    memset(binding.nonce, 'n', 32);
    memset(binding.profile, 'p', 32);
    memset(binding.content, 'c', 32);
    DWORD count = 0;
    DWORD error = us_activation_plan_prepare(data, read, &binding, argv[2], &count);
    printf("error=%lu count=%lu prepares=%lu pin_denials=%lu diagnostic=%lu normalized_denials=%lu opened_queries=%lu dos_denials=%lu\n",
        error, count, prepares, pin_denials, us_activation_plan_diagnostic(), normalized_denials, opened_queries, dos_denials);
    free(data);
    return error ? 1 : 0;
}
