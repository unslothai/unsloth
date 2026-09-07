/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#define WIN32_LEAN_AND_MEAN
#include "authority_audit.h"
#include <processsnapshot.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

typedef DWORD (WINAPI *UsCapture)(HANDLE, PSS_CAPTURE_FLAGS, DWORD, HPSS *);
typedef DWORD (WINAPI *UsQuery)(HPSS, PSS_QUERY_INFORMATION_CLASS, void *, DWORD);
typedef DWORD (WINAPI *UsWalk)(HPSS, PSS_WALK_INFORMATION_CLASS, HPSSWALK, void *, DWORD);
typedef DWORD (WINAPI *UsMarkerCreate)(const PSS_ALLOCATOR *, HPSSWALK *);
typedef DWORD (WINAPI *UsMarkerFree)(HPSSWALK);
typedef DWORD (WINAPI *UsSnapshotFree)(HANDLE, HPSS);
typedef LONG (NTAPI *UsNtObject)(HANDLE, ULONG, void *, ULONG, ULONG *);
typedef LONG (NTAPI *UsNtKey)(HANDLE, ULONG, void *, ULONG, ULONG *);
typedef struct { USHORT length, maximum; wchar_t *buffer; } UsString;
typedef struct { ULONG attributes, access, handles, pointers, reserved[10]; } UsObjectBasic;
typedef struct {
    UsCapture capture; UsQuery query; UsWalk walk;
    UsMarkerCreate marker_create; UsMarkerFree marker_free; UsSnapshotFree snapshot_free;
    UsNtObject object; UsNtKey key;
} UsAuditApi;

static BOOL us_symbol(HMODULE module, const char *name, void *destination, size_t bytes) {
    FARPROC address = GetProcAddress(module, name);
    if (!address || bytes != sizeof(address)) return FALSE;
    memcpy(destination, &address, bytes);
    return TRUE;
}

static BOOL us_audit_api(UsAuditApi *api) {
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll"), nt = GetModuleHandleW(L"ntdll.dll");
    if (!kernel || !nt) return FALSE;
#define US_RESOLVE(module, field, name) if (!us_symbol(module, name, &api->field, sizeof(api->field))) return FALSE
    US_RESOLVE(kernel, capture, "PssCaptureSnapshot");
    US_RESOLVE(kernel, query, "PssQuerySnapshot");
    US_RESOLVE(kernel, walk, "PssWalkSnapshot");
    US_RESOLVE(kernel, marker_create, "PssWalkMarkerCreate");
    US_RESOLVE(kernel, marker_free, "PssWalkMarkerFree");
    US_RESOLVE(kernel, snapshot_free, "PssFreeSnapshot");
    US_RESOLVE(nt, object, "NtQueryObject");
    US_RESOLVE(nt, key, "NtQueryKey");
#undef US_RESOLVE
    return TRUE;
}

static int __cdecl us_handle_compare(const void *left, const void *right) {
    uintptr_t a = *(const uintptr_t *)left, b = *(const uintptr_t *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static DWORD us_capture_handles(const UsAuditApi *api, uintptr_t *handles, DWORD *count) {
    HPSS snapshot = NULL;
    HPSSWALK marker = NULL;
    *count = 0;
    DWORD live_count = 0;
    if (!GetProcessHandleCount(GetCurrentProcess(), &live_count)) return GetLastError();
    if (live_count > US_AUTHORITY_MAX_HANDLES) return ERROR_BUFFER_OVERFLOW;
    DWORD error = api->capture(GetCurrentProcess(), PSS_CAPTURE_HANDLES, 0, &snapshot);
    if (error) return error;
    PSS_HANDLE_INFORMATION information = {0};
    error = api->query(snapshot, PSS_QUERY_HANDLE_INFORMATION, &information, sizeof(information));
    if (!error && information.HandlesCaptured > US_AUTHORITY_MAX_HANDLES) error = ERROR_BUFFER_OVERFLOW;
    if (!error) error = api->marker_create(NULL, &marker);
    if (!error) {
        PSS_HANDLE_ENTRY entry;
        while ((error = api->walk(snapshot, PSS_WALK_HANDLES, marker, &entry, sizeof(entry))) == ERROR_SUCCESS) {
            if (*count >= US_AUTHORITY_MAX_HANDLES || !entry.Handle) { error = ERROR_INVALID_DATA; break; }
            handles[(*count)++] = (uintptr_t)entry.Handle;
        }
        if (error == ERROR_NO_MORE_ITEMS)
            error = *count == information.HandlesCaptured ? ERROR_SUCCESS : ERROR_INVALID_DATA;
    }
    if (marker) {
        DWORD closed = api->marker_free(marker);
        if (!error) error = closed;
    }
    DWORD closed = api->snapshot_free(GetCurrentProcess(), snapshot);
    if (!error) error = closed;
    if (!error) {
        qsort(handles, *count, sizeof(*handles), us_handle_compare);
        for (DWORD i = 1; i < *count; ++i)
            if (handles[i] == handles[i - 1]) return ERROR_INVALID_DATA;
    }
    return error;
}

static DWORD us_key_name(const UsAuditApi *api, HANDLE key, wchar_t *name, DWORD chars, UsAuthorityReport *report) {
    union { ULONG_PTR alignment; unsigned char bytes[8192]; } buffer;
    ULONG used = 0;
    LONG status = api->key(key, 3, buffer.bytes, sizeof(buffer.bytes), &used);
    if (status < 0) { report->native_status = (DWORD)status; return ERROR_NOT_SUPPORTED; }
    if (used < sizeof(ULONG) || used > sizeof(buffer.bytes)) return ERROR_INVALID_DATA;
    ULONG length;
    memcpy(&length, buffer.bytes, sizeof(length));
    if (length % sizeof(wchar_t) || length > used - sizeof(ULONG) || length / sizeof(wchar_t) >= chars)
        return ERROR_INVALID_DATA;
    memcpy(name, buffer.bytes + sizeof(ULONG), length);
    name[length / sizeof(wchar_t)] = 0;
    if (wcslen(name) != length / sizeof(wchar_t)) return ERROR_INVALID_DATA;
    return ERROR_SUCCESS;
}

static DWORD us_type_name(const UsAuditApi *api, HANDLE handle, wchar_t *name, UsAuthorityReport *report) {
    union { ULONG_PTR alignment; unsigned char bytes[4096]; } buffer;
    ULONG used = 0;
    LONG status = api->object(handle, 2, buffer.bytes, sizeof(buffer.bytes), &used);
    if (status < 0) { report->native_status = (DWORD)status; return ERROR_NOT_SUPPORTED; }
    UsString value;
    memcpy(&value, buffer.bytes, sizeof(value));
    uintptr_t start = (uintptr_t)buffer.bytes, address = (uintptr_t)value.buffer;
    if (used < sizeof(value) || used > sizeof(buffer.bytes) || !value.length
        || value.length % sizeof(wchar_t) || value.length / sizeof(wchar_t) >= US_AUTHORITY_TYPE_CHARS
        || address < start || address > start + used || value.length > start + used - address)
        return ERROR_INVALID_DATA;
    memcpy(name, value.buffer, value.length);
    name[value.length / sizeof(wchar_t)] = 0;
    return wcslen(name) == value.length / sizeof(wchar_t) ? ERROR_SUCCESS : ERROR_INVALID_DATA;
}

static DWORD us_count_type(UsAuthorityReport *report, const wchar_t *name, DWORD *index) {
    for (DWORD i = 0; i < report->type_count; ++i) {
        if (!wcscmp(report->types[i].name, name)) { ++report->types[i].count; *index = i; return ERROR_SUCCESS; }
    }
    if (report->type_count == US_AUTHORITY_MAX_TYPES) return ERROR_BUFFER_OVERFLOW;
    *index = report->type_count++;
    UsAuthorityType *entry = &report->types[*index];
    if (wcscpy_s(entry->name, US_AUTHORITY_TYPE_CHARS, name)) return ERROR_INVALID_DATA;
    entry->count = 1;
    return ERROR_SUCCESS;
}

DWORD us_authority_audit(HKEY own_catalog, UsAuthorityReport *report) {
    if (!report) return ERROR_INVALID_PARAMETER;
    memset(report, 0, sizeof(*report));
    report->version = 1;
    UsAuditApi api = {0};
    DWORD error = ERROR_SUCCESS;
    if (!own_catalog || !us_audit_api(&api)) { report->error = ERROR_NOT_SUPPORTED; return report->error; }
    wchar_t own_name[1024];
    error = us_key_name(&api, own_catalog, own_name, 1024, report);
    size_t root_length = error ? 0 : wcslen(own_name);
    const wchar_t prefix[] = L"\\REGISTRY\\A\\";
    size_t prefix_length = wcslen(prefix);
    if (!error && (_wcsnicmp(own_name, prefix, prefix_length) || root_length <= prefix_length
                   || wcschr(own_name + prefix_length, L'\\'))) error = ERROR_INVALID_DATA;
    uintptr_t *before = NULL, *after = NULL;
    if (!error) {
        before = HeapAlloc(GetProcessHeap(), 0, 2 * US_AUTHORITY_MAX_HANDLES * sizeof(uintptr_t));
        if (!before) error = ERROR_NOT_ENOUGH_MEMORY;
        else after = before + US_AUTHORITY_MAX_HANDLES;
    }
    DWORD after_count = 0;
    if (!error) error = us_capture_handles(&api, before, &report->handles);
    for (DWORD i = 0; !error && i < report->handles; ++i) {
        HANDLE handle = (HANDLE)before[i];
        wchar_t type[US_AUTHORITY_TYPE_CHARS];
        error = us_type_name(&api, handle, type, report);
        if (error) break;
        UsAuthorityHandle *item = &report->inventory[i];
        item->handle = (ULONG_PTR)handle;
        error = us_count_type(report, type, &item->type_index);
        if (error) break;
        UsObjectBasic basic;
        ULONG used = 0;
        LONG native = api.object(handle, 0, &basic, sizeof(basic), &used);
        if (native < 0 || used != sizeof(basic)) {
            report->native_status = (DWORD)native;
            error = ERROR_NOT_SUPPORTED;
            break;
        }
        item->granted_access = basic.access;
        if (!wcscmp(type, L"Key")) {
            wchar_t name[4096];
            error = us_key_name(&api, handle, name, 4096, report);
            if (error) break;
            if (!_wcsnicmp(name, own_name, root_length) && (name[root_length] == 0 || name[root_length] == L'\\'))
                ++report->private_keys;
            else ++report->foreign_keys;
        } else if (!wcscmp(type, L"Token")) ++report->tokens;
        else if (!wcscmp(type, L"Process")) {
            DWORD pid = GetProcessId(handle);
            item->owner_pid = pid;
            if (!pid) error = GetLastError();
            else if (pid != GetCurrentProcessId()) ++report->foreign_processes;
        } else if (!wcscmp(type, L"Thread")) {
            DWORD pid = GetProcessIdOfThread(handle);
            item->owner_pid = pid;
            if (!pid) error = GetLastError();
            else if (pid != GetCurrentProcessId()) ++report->foreign_threads;
        } else ++report->other_handles;
    }
    if (!error) error = us_capture_handles(&api, after, &after_count);
    if (!error) {
        report->stable_inventory = after_count == report->handles && !memcmp(before, after, after_count * sizeof(uintptr_t));
        if (!report->stable_inventory) error = ERROR_RETRY;
        else if (!report->private_keys) error = ERROR_INVALID_DATA;
        else if (report->foreign_keys || report->tokens || report->foreign_processes || report->foreign_threads)
            error = ERROR_ACCESS_DENIED;
    }
    if (before && !HeapFree(GetProcessHeap(), 0, before) && !error) error = GetLastError();
    report->error = error;
    return error;
}
