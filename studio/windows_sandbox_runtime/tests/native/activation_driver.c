/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#define WIN32_LEAN_AND_MEAN
#include "activation_context.h"
#include <stdio.h>
#include <stdlib.h>

int wmain(int argc, wchar_t **argv) {
    if (argc != 5) return 2;
    FILE *input = NULL;
    if (_wfopen_s(&input, argv[2], L"rb")) return 3;
    unsigned char data[US_ACTIVATION_BYTES + 1];
    size_t bytes = fread(data, 1, sizeof(data), input);
    fclose(input);
    if (!bytes || bytes > US_ACTIVATION_BYTES) return 4;
    UsActivationInput item = {argv[1], data, (DWORD)bytes, 2, (WORD)wcstoul(argv[3], NULL, 10)};
    DWORD error = us_activation_prepare(&item, 1);
    if (error) { printf("prepare=%lu\n", error); return 5; }
    if (GetModuleHandleW(argv[1])) return 6;
    if (wcscmp(argv[4], L"prepare-only") == 0) {
        puts("resource_only=1"); return 0;
    }
    error = us_activation_install();
    if (error) { printf("install=%lu\n", error); return 7; }
    if (us_activation_install() != ERROR_INVALID_STATE) return 8;
    if (wcscmp(argv[4], L"reload") == 0) {
        for (int i = 0; i < 32; ++i) {
            HMODULE loaded = LoadLibraryExW(argv[1], NULL,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!loaded) { printf("load=%lu\n", GetLastError()); return 9; }
            if (!FreeLibrary(loaded) || GetModuleHandleW(argv[1])) return 14;
        }
        printf("reloads=32 substitutions=%ld\n", us_activation_substitutions());
        return us_activation_substitutions() == 32 ? 0 : 15;
    }
    /* Resource-only mapping must never satisfy a request for an executable
       module; a loader mapping does, without executing a PyInit function. */
    HMODULE module = LoadLibraryExW(argv[1], NULL, DONT_RESOLVE_DLL_REFERENCES);
    if (!module) { printf("load=%lu\n", GetLastError()); return 9; }
    ACTCTXW request = {0};
    request.cbSize = sizeof(request);
    request.dwFlags = ACTCTX_FLAG_RESOURCE_NAME_VALID | ACTCTX_FLAG_HMODULE_VALID;
    request.lpResourceName = MAKEINTRESOURCEW(2);
    request.lpSource = argv[1];
    request.hModule = GetModuleHandleW(NULL);
    if (us_activation_match(&request)) return 10;
    request.hModule = module;
    request.dwFlags |= ACTCTX_FLAG_APPLICATION_NAME_VALID;
    if (us_activation_match(&request)) return 11;
    request.dwFlags &= ~ACTCTX_FLAG_APPLICATION_NAME_VALID;
    request.lpResourceName = MAKEINTRESOURCEW(1);
    if (us_activation_match(&request)) return 12;
    request.lpResourceName = MAKEINTRESOURCEW(2);
    request.lpSource = L"Z:\\incorrect-source-must-not-override-hmodule.dll";
    for (int i = 0; i < 128; ++i) {
        HANDLE context = us_activation_match(&request);
        if (!context) return 13;
        ReleaseActCtx(context);
    }
    FreeLibrary(module);
    puts("resource_only=1 wrong_module_denied=1 unsupported_request_denied=1 references=128");
    return 0;
}
