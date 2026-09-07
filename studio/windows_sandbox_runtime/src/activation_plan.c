/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#include "activation_plan.h"
#include "activation_context.h"
#include <bcrypt.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define PLAN_HEADER 116u
#define PLAN_ENTRY 112u
#define PLAN_BYTES 1048576u
#define IMAGE_BYTES 134217728u
#define PATH_CHARS 16383u
#ifdef US_ACTIVATION_TEST_PATH_QUERY
/* Compile-time native test seam; production always calls the Windows API. */
DWORD us_activation_test_path_query(HANDLE, LPWSTR, DWORD, DWORD);
#define query_opened_path us_activation_test_path_query
#else
#define query_opened_path GetFinalPathNameByHandleW
#endif
static DWORD diagnostic;
DWORD us_activation_plan_diagnostic(void) { return diagnostic; }

static DWORD word32(const BYTE *p) { DWORD value; memcpy(&value, p, 4); return value; }
static WORD word16(const BYTE *p) { WORD value; memcpy(&value, p, 2); return value; }
static ULONGLONG word64(const BYTE *p) { ULONGLONG value; memcpy(&value, p, 8); return value; }

static BOOL valid_path(const wchar_t *text, size_t length) {
    if (length < 4 || length > 1027 || text[1] != L':' || text[2] != L'\\'
        || !((text[0] >= L'A' && text[0] <= L'Z') || (text[0] >= L'a' && text[0] <= L'z')))
        return FALSE;
    size_t start = 3, components = 0;
    for (size_t i = 3; i <= length; ++i) {
        unsigned ch = text[i];
        if (i < length && !ch) return FALSE;
        if (ch >= 0xd800 && ch <= 0xdbff) {
            if (++i >= length || text[i] < 0xdc00 || text[i] > 0xdfff) return FALSE;
            continue;
        }
        if (ch >= 0xdc00 && ch <= 0xdfff) return FALSE;
        if (!ch || ch == L'\\') {
            size_t size = i - start;
            if (++components > 16 || !size || text[i-1] == L'.' || text[i-1] == L' ') return FALSE;
            wchar_t stem[5] = {0};
            size_t j = 0;
            while (j < size && j < 4 && text[start+j] != L'.') {
                wchar_t c = text[start+j];
                stem[j++] = (c >= L'A' && c <= L'Z') ? c + (L'a'-L'A') : c;
            }
            BOOL ended = j == size || text[start+j] == L'.';
            if (ended && (!_wcsicmp(stem, L"con") || !_wcsicmp(stem, L"prn")
                || !_wcsicmp(stem, L"aux") || !_wcsicmp(stem, L"nul"))) return FALSE;
            if (ended && j == 4 && (!_wcsnicmp(stem, L"com", 3) || !_wcsnicmp(stem, L"lpt", 3))
                && ((stem[3] >= L'1' && stem[3] <= L'9') || stem[3] == 0xb9
                    || stem[3] == 0xb2 || stem[3] == 0xb3)) return FALSE;
            start = i + 1;
        } else if (ch < 32 || wcschr(L"/:<>\"|?*", (wchar_t)ch)) return FALSE;
    }
    return TRUE;
}

static DWORD hash_file(BCRYPT_ALG_HANDLE algorithm, HANDLE file, ULONGLONG size,
                       const BYTE expected[32]) {
    BYTE buffer[65536], digest[32];
    BCRYPT_HASH_HANDLE hash = NULL;
    DWORD error = ERROR_CRC;
    NTSTATUS status = BCryptCreateHash(algorithm, &hash, NULL, 0, NULL, 0, 0);
    if (status < 0) return (DWORD)status;
    ULONGLONG remaining = size;
    while (remaining) {
        DWORD received = 0, chunk = remaining > sizeof(buffer) ? (DWORD)sizeof(buffer) : (DWORD)remaining;
        if (!ReadFile(file, buffer, chunk, &received, NULL)) { error = GetLastError(); goto done; }
        if (received != chunk) { error = ERROR_HANDLE_EOF; goto done; }
        status = BCryptHashData(hash, buffer, received, 0);
        if (status < 0) { error = (DWORD)status; goto done; }
        remaining -= received;
    }
    status = BCryptFinishHash(hash, digest, sizeof(digest), 0);
    if (status < 0) error = (DWORD)status;
    else if (!memcmp(digest, expected, 32)) error = ERROR_SUCCESS;
done:
    BCryptDestroyHash(hash);
    return error;
}

DWORD us_activation_plan_prepare(const BYTE *blob, DWORD bytes,
    const UsBinding *binding, const wchar_t *runtime_home, DWORD *count) {
    diagnostic = 1;
    if (!count) return ERROR_INVALID_PARAMETER;
    *count = 0;
    if (!blob || !binding || !runtime_home || bytes < PLAN_HEADER || bytes > PLAN_BYTES)
        return ERROR_INVALID_DATA;
    size_t home_length = wcsnlen(runtime_home, PATH_CHARS + 1);
    if (!valid_path(runtime_home, home_length)) return ERROR_INVALID_NAME;
    const wchar_t *last = wcsrchr(runtime_home, L'\\');
    size_t root_length = last ? (size_t)(last - runtime_home) : 0;
    if (root_length <= 3) return ERROR_INVALID_NAME;
    DWORD entries = word32(blob + 16);
    if (memcmp(blob, "USLPACT", 8) || word32(blob + 8) != 2 || word32(blob + 12) != bytes
        || entries > US_ACTIVATION_LIMIT || memcmp(blob + 20, binding, sizeof(*binding)))
        return ERROR_INVALID_DATA;
    UsActivationInput inputs[US_ACTIVATION_LIMIT] = {0};
    HANDLE pins[US_ACTIVATION_LIMIT] = {0};
    BCRYPT_ALG_HANDLE algorithm = NULL;
    DWORD error = ERROR_BAD_FORMAT;
    diagnostic = 2;
    NTSTATUS crypto_status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, NULL, 0);
    if (crypto_status < 0) return (DWORD)crypto_status;
    size_t cursor = PLAN_HEADER;
    for (DWORD i = 0; i < entries; ++i) {
        diagnostic = 3;
        if (cursor + PLAN_ENTRY > bytes) goto done;
        const BYTE *record = blob + cursor;
        DWORD path_bytes = word32(record), manifest_bytes = word32(record + 4);
        ULONGLONG image_bytes = word64(record + 16);
        if (!path_bytes || path_bytes > PATH_CHARS * 2 || path_bytes % 2
            || !manifest_bytes || manifest_bytes > US_ACTIVATION_BYTES
            || !image_bytes || image_bytes > IMAGE_BYTES || word16(record + 8) != 2)
            goto done;
        cursor += PLAN_ENTRY;
        if (cursor + path_bytes + manifest_bytes > bytes) goto done;
        wchar_t *path = (wchar_t *)calloc((size_t)path_bytes / 2 + 1, sizeof(wchar_t));
        if (!path) { error = ERROR_NOT_ENOUGH_MEMORY; goto done; }
        inputs[i].path = path;
        memcpy(path, blob + cursor, path_bytes);
        cursor += path_bytes;
        diagnostic = 4;
        if (!valid_path(path, path_bytes / 2) || path_bytes / 2 <= root_length
            || _wcsnicmp(path, runtime_home, root_length) || path[root_length] != L'\\') {
            error = ERROR_BAD_PATHNAME; goto done;
        }
        for (DWORD j = 0; j < i; ++j)
            if (!_wcsicmp(path, inputs[j].path)) { error = ERROR_DUP_NAME; goto done; }
        inputs[i].manifest = blob + cursor;
        inputs[i].manifest_bytes = manifest_bytes;
        inputs[i].resource_id = word16(record + 8);
        inputs[i].language = word16(record + 10);
        cursor += manifest_bytes;
        BYTE digest[32];
        diagnostic = 5;
        crypto_status = BCryptHash(algorithm, NULL, 0, (PUCHAR)inputs[i].manifest, manifest_bytes,
                digest, sizeof(digest));
        if (crypto_status < 0) { error = (DWORD)crypto_status; goto done; }
        if (memcmp(digest, record + 56, 32)) {
            error = ERROR_CRC; goto done;
        }
        diagnostic = 6;
        pins[i] = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (pins[i] == INVALID_HANDLE_VALUE) { error = GetLastError(); goto done; }
        BY_HANDLE_FILE_INFORMATION info;
        LARGE_INTEGER actual_size;
        diagnostic = 7;
        if (!GetFileInformationByHandle(pins[i], &info) || !GetFileSizeEx(pins[i], &actual_size)) {
            error = GetLastError(); goto done;
        }
        if (GetFileType(pins[i]) != FILE_TYPE_DISK || actual_size.QuadPart < 0
            || (ULONGLONG)actual_size.QuadPart != image_bytes || info.nNumberOfLinks != 1
            || (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) {
            error = ERROR_FILE_INVALID; goto done;
        }
        FILE_ID_INFO actual_identity;
        _Static_assert(sizeof(FILE_ID_INFO) == 24, "Activation file identity ABI changed");
        if (!GetFileInformationByHandleEx(pins[i], FileIdInfo, &actual_identity, sizeof(actual_identity))) {
            error = GetLastError(); goto done;
        }
        if (memcmp(&actual_identity, record + 88, sizeof(actual_identity))) {
            error = ERROR_FILE_INVALID; goto done;
        }
        wchar_t final_path[PATH_CHARS + 5];
        diagnostic = 8;
        /* The broker supplied a canonical final path and retains its verified
           directory leases. Query the opened handle without normalizing ancestor
           names or resolving DOS drive mappings: both can fail inside LPAC.
           The volume/file ID above binds the exact object independently of its
           drive letter. This remains a handle path comparison plus metadata, the complete
           image hash below and the read pin held through context preparation.
           No ancestor ACL is widened and lexical path equality alone admits
           nothing. Reparse aliases that resolve elsewhere still fail equality. */
        DWORD length = query_opened_path(pins[i], final_path, PATH_CHARS + 5, FILE_NAME_OPENED | VOLUME_NAME_NONE);
        if (!length) { error = GetLastError(); goto done; }
        if (length >= PATH_CHARS + 5 || final_path[0] != L'\\'
            || _wcsicmp(final_path, path + 2)) { error = ERROR_BAD_PATHNAME; goto done; }
        diagnostic = 9;
        DWORD hash_error = hash_file(algorithm, pins[i], image_bytes, record + 24);
        if (hash_error) { error = hash_error; goto done; }
    }
    if (cursor != bytes) goto done;
    /* Every image remains pinned while the adapter takes its own read pin and
       compares the mapped resource's exact bytes before CreateActCtxW. */
    diagnostic = 10;
    error = entries ? us_activation_prepare(inputs, entries) : ERROR_SUCCESS;
    if (!error) { *count = entries; diagnostic = 11; }
done:
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    for (DWORD i = 0; i < US_ACTIVATION_LIMIT; ++i) {
        if (pins[i] && pins[i] != INVALID_HANDLE_VALUE) CloseHandle(pins[i]);
        free((void *)inputs[i].path);
    }
    return error;
}
