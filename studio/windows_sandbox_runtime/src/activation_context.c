/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#define WIN32_LEAN_AND_MEAN
#include "activation_context.h"
#include <psapi.h>
#include <tlhelp32.h>
#include <detours.h>
#include <string.h>

#define US_ACTIVATION_PATH 32768u
#define US_ACTIVATION_THREADS 256u
typedef HANDLE (WINAPI *CreateContext)(PCACTCTXW);
typedef struct {
    HANDLE pin;
    HMODULE resources;
    HANDLE context;
    wchar_t mapped_path[US_ACTIVATION_PATH];
    const void *manifest;
    DWORD manifest_bytes;
    WORD resource_id;
    WORD language;
} ContextEntry;
static ContextEntry entries[US_ACTIVATION_LIMIT];
static size_t entry_count;
static LONG prepared, installed;
static LONG substitutions;
static PVOID original_address;

static DWORD mapped_name(HMODULE module, wchar_t *name, BOOL executable) {
    MEMORY_BASIC_INFORMATION memory;
    void *base = (void *)((ULONG_PTR)module & ~(ULONG_PTR)3);
    if (!base || !VirtualQuery(base, &memory, sizeof(memory))
        || memory.AllocationBase != base
        || (executable && (memory.Type != MEM_IMAGE || base != (void *)module)))
        return ERROR_INVALID_HANDLE;
    DWORD length = GetMappedFileNameW(GetCurrentProcess(), base, name, US_ACTIVATION_PATH);
    if (!length) return GetLastError();
    if (length >= US_ACTIVATION_PATH) return ERROR_FILENAME_EXCED_RANGE;
    return ERROR_SUCCESS;
}

static DWORD manifest_bytes(HMODULE module, WORD id, WORD language,
                            const void **data, DWORD *bytes) {
    HRSRC resource = FindResourceExW(module, RT_MANIFEST, MAKEINTRESOURCEW(id), language);
    if (!resource) return GetLastError();
    *bytes = SizeofResource(module, resource);
    if (!*bytes || *bytes > US_ACTIVATION_BYTES) return ERROR_INVALID_DATA;
    HGLOBAL loaded = LoadResource(module, resource);
    if (!loaded) return GetLastError();
    *data = LockResource(loaded);
    return *data ? ERROR_SUCCESS : ERROR_INVALID_DATA;
}

DWORD us_activation_prepare(const UsActivationInput *inputs, size_t count) {
    if (!inputs || !count || count > US_ACTIVATION_LIMIT)
        return ERROR_INVALID_PARAMETER;
    if (InterlockedCompareExchange(&prepared, -1, 0)) return ERROR_INVALID_STATE;
    /* A failed preparation is terminal. Do not unload libraries or invoke
       callbacks to recover while the trusted startup token may still exist. */
    for (size_t i = 0; i < count; ++i) {
        const UsActivationInput *input = &inputs[i];
        ContextEntry *entry = &entries[i];
        if (!input->path || !input->manifest || !input->manifest_bytes
            || input->manifest_bytes > US_ACTIVATION_BYTES || input->resource_id != 2)
            return ERROR_INVALID_PARAMETER;
        entry->pin = CreateFileW(input->path, GENERIC_READ, FILE_SHARE_READ, NULL,
            OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
        if (entry->pin == INVALID_HANDLE_VALUE) return GetLastError();
        BY_HANDLE_FILE_INFORMATION info;
        if (!GetFileInformationByHandle(entry->pin, &info)) return GetLastError();
        if (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)
            || info.nNumberOfLinks != 1) return ERROR_INVALID_DATA;
        entry->resources = LoadLibraryExW(input->path, NULL,
            LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
        if (!entry->resources) return GetLastError();
        DWORD error = mapped_name(entry->resources, entry->mapped_path, FALSE);
        if (error) return error;
        for (size_t j = 0; j < i; ++j)
            if (!_wcsicmp(entry->mapped_path, entries[j].mapped_path))
                return ERROR_DUP_NAME;
        error = manifest_bytes(entry->resources, input->resource_id, input->language,
            &entry->manifest, &entry->manifest_bytes);
        if (error) return error;
        if (entry->manifest_bytes != input->manifest_bytes
            || memcmp(entry->manifest, input->manifest, input->manifest_bytes))
            return ERROR_INVALID_DATA;
        entry->resource_id = input->resource_id;
        entry->language = input->language;
        ACTCTXW request = {0};
        request.cbSize = sizeof(request);
        request.dwFlags = ACTCTX_FLAG_RESOURCE_NAME_VALID | ACTCTX_FLAG_LANGID_VALID;
        request.lpSource = input->path;
        request.lpResourceName = MAKEINTRESOURCEW(entry->resource_id);
        request.wLangId = entry->language;
        entry->context = CreateActCtxW(&request);
        if (entry->context == INVALID_HANDLE_VALUE) return GetLastError();
        ++entry_count;
    }
    InterlockedExchange(&prepared, 1);
    return ERROR_SUCCESS;
}

HANDLE us_activation_match(PCACTCTXW request) {
    if (InterlockedCompareExchange(&prepared, 1, 1) != 1 || !request
        || request->cbSize != sizeof(*request)
        || request->dwFlags != (ACTCTX_FLAG_RESOURCE_NAME_VALID | ACTCTX_FLAG_HMODULE_VALID)
        || request->lpResourceName != MAKEINTRESOURCEW(2) || !request->hModule)
        return NULL;
    /* HMODULE_VALID makes hModule authoritative. lpSource may be misleading or
       NULL, so never substitute a context on a source-string match alone.
       GetModuleFileName cannot identify images not yet in the loader list. */
    wchar_t name[US_ACTIVATION_PATH];
    if (mapped_name(request->hModule, name, TRUE)) return NULL;
    for (size_t i = 0; i < entry_count; ++i) {
        ContextEntry *entry = &entries[i];
        if (_wcsicmp(name, entry->mapped_path)) continue;
        const void *data = NULL;
        DWORD bytes = 0;
        if (manifest_bytes(request->hModule, entry->resource_id, entry->language,
                &data, &bytes) || bytes != entry->manifest_bytes
            || memcmp(data, entry->manifest, bytes)) return NULL;
        AddRefActCtx(entry->context);
        return entry->context;
    }
    return NULL;
}

static HANDLE WINAPI create_context(PCACTCTXW request) {
    DWORD saved = GetLastError();
    HANDLE context = us_activation_match(request);
    SetLastError(saved);
    if (context) { InterlockedIncrement(&substitutions); return context; }
    CreateContext original = NULL;
    memcpy(&original, &original_address, sizeof(original));
    return original(request);
}

LONG us_activation_substitutions(void) {
    return InterlockedCompareExchange(&substitutions, 0, 0);
}

DWORD us_activation_install(void) {
    if (InterlockedCompareExchange(&prepared, 1, 1) != 1
        || InterlockedCompareExchange(&installed, -1, 0)) return ERROR_INVALID_STATE;
    HMODULE kernel = GetModuleHandleW(L"KernelBase.dll");
    FARPROC address = kernel ? GetProcAddress(kernel, "CreateActCtxW") : NULL;
    if (!address) return ERROR_PROC_NOT_FOUND;
    memcpy(&original_address, &address, sizeof(address));
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return GetLastError();
    HANDLE threads[US_ACTIVATION_THREADS];
    size_t count = 0;
    LONG error = DetourTransactionBegin();
    if (error) { CloseHandle(snapshot); return (DWORD)error; }
    error = DetourUpdateThread(GetCurrentThread());
    THREADENTRY32 thread = {0};
    thread.dwSize = sizeof(thread);
    if (!error && !Thread32First(snapshot, &thread)) error = (LONG)GetLastError();
    while (!error) {
        if (thread.th32OwnerProcessID == GetCurrentProcessId()
            && thread.th32ThreadID != GetCurrentThreadId()) {
            if (count == US_ACTIVATION_THREADS) { error = ERROR_TOO_MANY_TCBS; break; }
            HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT
                | THREAD_SET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, thread.th32ThreadID);
            if (!handle) { error = (LONG)GetLastError(); break; }
            threads[count++] = handle;
            error = DetourUpdateThread(handle);
        }
        if (!Thread32Next(snapshot, &thread)) {
            DWORD next_error = GetLastError();
            if (next_error != ERROR_NO_MORE_FILES) error = (LONG)next_error;
            break;
        }
    }
    CloseHandle(snapshot);
    CreateContext hook = create_context;
    PVOID hook_address = NULL;
    memcpy(&hook_address, &hook, sizeof(hook));
    if (!error) error = DetourAttach(&original_address, hook_address);
    if (error) DetourTransactionAbort();
    else error = DetourTransactionCommit();
    for (size_t i = 0; i < count; ++i) CloseHandle(threads[i]);
    if (!error) InterlockedExchange(&installed, 1);
    return (DWORD)error;
}
