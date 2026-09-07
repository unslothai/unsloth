/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#include "host_config.h"
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

HANDLE us_handle_argument(const wchar_t *text) {
    uint64_t value = 0;
    size_t length = wcslen(text);
    if (!length || length > 20 || text[0] == L'0') return NULL;
    for (size_t i = 0; i < length; ++i) {
        if (text[i] < L'0' || text[i] > L'9') return NULL;
        uint64_t digit = (uint64_t)(text[i] - L'0');
        if (value > (UINT64_MAX - digit) / 10) return NULL;
        value = value * 10 + digit;
    }
    if (!value || value == UINT64_MAX) return NULL;
    return (HANDLE)(uintptr_t)value;
}

static BOOL valid_unicode(const wchar_t *text, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        unsigned ch = text[i];
        if (!ch) return FALSE;
        if (ch >= 0xd800 && ch <= 0xdbff) {
            if (++i >= length || text[i] < 0xdc00 || text[i] > 0xdfff) return FALSE;
        } else if (ch >= 0xdc00 && ch <= 0xdfff) return FALSE;
    }
    return TRUE;
}

static BOOL valid_path(const wchar_t *text) {
    size_t length = wcslen(text);
    if (length < 4 || text[1] != L':' || text[2] != L'\\'
        || !((text[0] >= L'A' && text[0] <= L'Z') || (text[0] >= L'a' && text[0] <= L'z')))
        return FALSE;
    size_t start = 3;
    for (size_t i = 3; i <= length; ++i) {
        wchar_t ch = text[i];
        if (!ch || ch == L'\\') {
            size_t size = i - start;
            if (!size || text[i-1] == L'.' || text[i-1] == L' ') return FALSE;
            wchar_t stem[5] = {0};
            size_t j = 0;
            while (j < size && j < 4 && text[start+j] != L'.') {
                wchar_t part = text[start+j];
                stem[j++] = (part >= L'A' && part <= L'Z') ? part + (L'a'-L'A') : part;
            }
            BOOL ended = j == size || text[start+j] == L'.';
            if (ended && (!wcscmp(stem, L"con") || !wcscmp(stem, L"prn")
                || !wcscmp(stem, L"aux") || !wcscmp(stem, L"nul"))) return FALSE;
            if (ended && j == 4 && (!wcsncmp(stem, L"com", 3) || !wcsncmp(stem, L"lpt", 3))
                && ((stem[3] >= L'1' && stem[3] <= L'9') || stem[3] == 0xb9
                    || stem[3] == 0xb2 || stem[3] == 0xb3)) return FALSE;
            start = i + 1;
        } else if (ch < 32 || wcschr(L"/:<>\"|?*", ch)) return FALSE;
    }
    return TRUE;
}

void us_free_config(UsConfig *config) {
    for (size_t i = 0; i < US_CONFIG_FIELDS + 3 * US_CONFIG_LIST; ++i) {
        free(config->values[i]);
        config->values[i] = NULL;
    }
}

BOOL us_read_config(HANDLE input, UsConfig *config) {
    BYTE *data = NULL;
    BOOL ok = FALSE;
    LARGE_INTEGER size, zero = {0};
    memset(config, 0, sizeof(*config));
    if (GetFileType(input) != FILE_TYPE_DISK || !GetFileSizeEx(input, &size)
        || size.QuadPart < sizeof(UsConfigHeader) || size.QuadPart > US_CONFIG_BYTES
        || !SetFilePointerEx(input, zero, NULL, FILE_BEGIN)) goto done;
    data = (BYTE *)malloc((size_t)size.QuadPart);
    if (!data) goto done;
    DWORD received = 0, count = 0, total = (DWORD)size.QuadPart;
    while (received < total) {
        if (!ReadFile(input, data + received, total - received, &count, NULL) || !count) goto done;
        received += count;
    }
    memcpy(&config->header, data, sizeof(config->header));
    UsConfigHeader *header = &config->header;
    if (memcmp(header->magic, "USLPCF1", 8) || header->version != 1 || header->bytes != total
        || header->images > US_CONFIG_LIST || header->packages > US_CONFIG_LIST || header->arguments > US_CONFIG_LIST)
        goto done;
    size_t position = sizeof(config->header), image_start = US_CONFIG_FIELDS + header->packages + header->arguments;
    size_t fields = image_start + header->images;
    for (size_t i = 0; i < fields; ++i) {
        DWORD bytes = 0;
        if (position + sizeof(bytes) > total) goto done;
        memcpy(&bytes, data + position, sizeof(bytes));
        position += sizeof(bytes);
        if (bytes > 32766 || bytes % 2 || position + bytes > total) goto done;
        wchar_t *text = (wchar_t *)calloc((size_t)bytes / 2 + 1, sizeof(wchar_t));
        if (!text) goto done;
        config->values[i] = text;
        memcpy(text, data + position, bytes);
        position += bytes;
        if (!valid_unicode(text, bytes / 2)) goto done;
        if ((i < US_PACKAGE_SID || (i >= US_CONFIG_FIELDS && i < US_CONFIG_FIELDS + header->packages) || i >= image_start)
            && !valid_path(text)) goto done;
        if (i >= image_start) {
            size_t home_length = wcslen(config->values[US_RUNTIME_HOME]);
            if (wcslen(text) <= home_length || _wcsnicmp(text, config->values[US_RUNTIME_HOME], home_length)
                || text[home_length] != L'\\') goto done;
        }
    }
    ok = position == total;
done:
    free(data);
    if (!CloseHandle(input)) ok = FALSE;
    if (!ok) { us_free_config(config); SetLastError(ERROR_INVALID_DATA); }
    return ok;
}
