/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
/* Fixed native test workload, never selected by production. */
#include "gate.h"
#include <sddl.h>
#include <stdlib.h>
#include <wchar.h>

static BOOL hex_binding(const wchar_t *text, UsBinding *binding) {
    if (wcslen(text) != sizeof(*binding) * 2) return FALSE;
    BYTE *bytes = (BYTE *)binding;
    for (size_t i = 0; i < sizeof(*binding); ++i) {
        unsigned value = 0;
        for (size_t j = 0; j < 2; ++j) {
            wchar_t ch = text[i * 2 + j];
            if (ch >= L'0' && ch <= L'9') value = value * 16 + (unsigned)(ch - L'0');
            else if (ch >= L'a' && ch <= L'f') value = value * 16 + (unsigned)(ch - L'a' + 10);
            else return FALSE;
        }
        bytes[i] = (BYTE)value;
    }
    return TRUE;
}

static HANDLE handle_arg(const wchar_t *text) {
    wchar_t *end = NULL;
    unsigned long long value = _wcstoui64(text, &end, 10);
    if (!value || *end || value == UINT64_MAX) return NULL;
    return (HANDLE)(uintptr_t)value;
}

typedef struct {
    HANDLE ready, stop;
    BOOL retain;
    DWORD error;
} TestThread;

static DWORD WINAPI fixed_thread(void *argument) {
    TestThread *test = (TestThread *)argument;
    if (test->retain && !ImpersonateSelf(SecurityImpersonation)) test->error = GetLastError();
    if (!SetEvent(test->ready)) return 97;
    if (WaitForSingleObject(test->stop, 5000) != WAIT_OBJECT_0) return 98;
    if (test->retain && !RevertToSelf()) return 99;
    return 0;
}

int wmain(int argc, wchar_t **argv) {
    if (argc != 7 && argc != 8) return 90;
    if (argc == 8 && wcscmp(argv[7], L"retained-thread") && wcscmp(argv[7], L"clean-thread")) return 90;
    HANDLE output = handle_arg(argv[1]), input = handle_arg(argv[2]);
    UsBinding binding;
    PSID sid = NULL;
    if (!output || !input || !hex_binding(argv[3], &binding)
        || !ConvertStringSidToSidW(argv[4], &sid)) return 91;
    UsStatus status;
    us_status_init(&status, &binding);
    BOOL passed = us_validate_startup(sid, argv[5], &status);
    TestThread test = {0};
    HANDLE thread = NULL;
    if (passed && argc == 8) {
        test.retain = !wcscmp(argv[7], L"retained-thread");
        test.ready = CreateEventW(NULL, TRUE, FALSE, NULL);
        test.stop = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (test.ready && test.stop) thread = CreateThread(NULL, 0, fixed_thread, &test, 0, NULL);
        if (!thread || WaitForSingleObject(test.ready, 5000) != WAIT_OBJECT_0 || test.error)
            passed = us_fail(&status, 9, test.error ? test.error : GetLastError());
    }
    if (passed) passed = us_drop_and_validate(sid, argv[5], &status);
    if (thread) {
        if (!SetEvent(test.stop) || WaitForSingleObject(thread, 5000) != WAIT_OBJECT_0)
            passed = us_fail(&status, 9, ERROR_TIMEOUT);
        DWORD code = 1;
        if (!GetExitCodeThread(thread, &code) || code) passed = us_fail(&status, 9, code);
        CloseHandle(thread);
    }
    if (test.ready) CloseHandle(test.ready);
    if (test.stop) CloseHandle(test.stop);
    LocalFree(sid);
    if (!us_send_status(output, &status)) { CloseHandle(input); return 92; }
    if (!passed) { CloseHandle(input); return 93; }
    if (!us_wait_acknowledgement(input, &binding)) return 94;
    /* The externally observed sentinel is impossible before the complete gate
       and parent acknowledgement. No script or dynamic operation is accepted. */
    HANDLE file = CreateFileW(argv[6], GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
    if (file == INVALID_HANDLE_VALUE) return 95;
    DWORD written = 0;
    BOOL ok = WriteFile(file, "AFTER_GATE", 10, &written, NULL) && written == 10;
    CloseHandle(file);
    return ok ? 0 : 96;
}
