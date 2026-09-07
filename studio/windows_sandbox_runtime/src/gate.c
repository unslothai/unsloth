/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#include "gate.h"
#include <tlhelp32.h>
#include <string.h>

void us_status_init(UsStatus *status, const UsBinding *binding) {
    memset(status, 0, sizeof(*status));
    memcpy(status->magic, "USLPAC1", 8);
    status->version = US_PROTOCOL_VERSION;
    status->pid = GetCurrentProcessId();
    status->phase = US_FAILED;
    status->stage = 1;
    status->binding = *binding;
}

BOOL us_fail(UsStatus *status, DWORD stage, DWORD error) {
    status->phase = US_FAILED;
    status->stage = stage;
    status->error = error ? error : ERROR_ACCESS_DENIED;
    return FALSE;
}

static BOOL token_info(HANDLE token, TOKEN_INFORMATION_CLASS kind, void *data, DWORD size) {
    DWORD needed = 0;
    return GetTokenInformation(token, kind, data, size, &needed) && needed <= size;
}

static BOOL package_matches(HANDLE token, PSID expected) {
    __declspec(align(8)) BYTE data[4096];
    TOKEN_APPCONTAINER_INFORMATION *info = (TOKEN_APPCONTAINER_INFORMATION *)data;
    if (!token_info(token, TokenAppContainerSid, data, sizeof(data))) return FALSE;
    if (!info->TokenAppContainer || !EqualSid(info->TokenAppContainer, expected)) {
        SetLastError(ERROR_ACCESS_DENIED);
        return FALSE;
    }
    return TRUE;
}

static BOOL startup_capability(HANDLE token) {
    __declspec(align(8)) BYTE data[4096];
    TOKEN_GROUPS *groups = (TOKEN_GROUPS *)data;
    PSID *group_sids = NULL, *cap_sids = NULL;
    DWORD group_count = 0, cap_count = 0, error = ERROR_ACCESS_DENIED;
    BOOL matches = FALSE;
    if (!token_info(token, TokenCapabilities, data, sizeof(data))) return FALSE;
    if (groups->GroupCount != 1 || groups->Groups[0].Attributes != SE_GROUP_ENABLED) {
        SetLastError(ERROR_ACCESS_DENIED);
        return FALSE;
    }
    if (DeriveCapabilitySidsFromName(L"registryRead", &group_sids, &group_count, &cap_sids, &cap_count)) {
        matches = cap_count == 1 && EqualSid(groups->Groups[0].Sid, cap_sids[0]);
    } else {
        error = GetLastError();
    }
    for (DWORD i = 0; i < group_count; ++i) LocalFree(group_sids[i]);
    for (DWORD i = 0; i < cap_count; ++i) LocalFree(cap_sids[i]);
    LocalFree(group_sids);
    LocalFree(cap_sids);
    if (!matches) SetLastError(error);
    return matches;
}

BOOL us_validate_startup(PSID package_sid, const wchar_t *aap_path, UsStatus *status) {
    HANDLE token = NULL;
    DWORD is_app = 0;
    if (!IsValidSid(package_sid) || !OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, FALSE, &token))
        return us_fail(status, 2, GetLastError());
    BOOL valid = token_info(token, TokenIsAppContainer, &is_app, sizeof(is_app)) && is_app == 1
        && package_matches(token, package_sid) && startup_capability(token);
    DWORD error = GetLastError();
    if (!CloseHandle(token)) return us_fail(status, 2, GetLastError());
    if (!valid) return us_fail(status, 2, error);
    /* A real positive control: a missing/inaccessible file is not LPAC evidence. */
    HANDLE aap = CreateFileW(aap_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (aap == INVALID_HANDLE_VALUE) return us_fail(status, 3, GetLastError());
    if (!CloseHandle(aap)) return us_fail(status, 3, GetLastError());
    status->checks |= 1u;
    return TRUE;
}

static BOOL all_thread_tokens_absent(void) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return FALSE;
    THREADENTRY32 entry = {0};
    entry.dwSize = sizeof(entry);
    DWORD count = 0, scanned = 0, error = 0;
    BOOL more = Thread32First(snapshot, &entry);
    if (!more) error = GetLastError();
    while (more && !error) {
        if (++scanned > 65536) { error = ERROR_NOT_ENOUGH_QUOTA; break; }
        if (entry.dwSize < FIELD_OFFSET(THREADENTRY32, th32OwnerProcessID) + sizeof(DWORD)) {
            error = ERROR_INVALID_DATA; break;
        }
        if (entry.th32OwnerProcessID == GetCurrentProcessId()) {
            if (++count > 128) { error = ERROR_NOT_ENOUGH_QUOTA; break; }
            HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
            if (!thread) { error = GetLastError(); break; }
            HANDLE token = NULL;
            BOOL opened = OpenThreadToken(thread, TOKEN_QUERY, TRUE, &token);
            DWORD token_error = GetLastError();
            if (opened) CloseHandle(token);
            if (!CloseHandle(thread)) { error = GetLastError(); break; }
            if (opened || token_error != ERROR_NO_TOKEN) {
                error = opened ? ERROR_ACCESS_DENIED : token_error;
                break;
            }
        }
        entry.dwSize = sizeof(entry);
        more = Thread32Next(snapshot, &entry);
        if (!more && GetLastError() != ERROR_NO_MORE_FILES) error = GetLastError();
    }
    if (!CloseHandle(snapshot) && !error) error = GetLastError();
    if (error || !count) { SetLastError(error ? error : ERROR_INVALID_DATA); return FALSE; }
    return TRUE;
}

BOOL us_drop_and_validate(PSID package_sid, const wchar_t *aap_path, UsStatus *status) {
    HANDLE token = NULL;
    DWORD value = 0;
    __declspec(align(8)) BYTE data[4096];
    if (!(status->checks & 1u)) return us_fail(status, 4, ERROR_INVALID_STATE);
    if (!RevertToSelf()) return us_fail(status, 4, GetLastError());
    if (OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &token)) {
        CloseHandle(token);
        return us_fail(status, 4, ERROR_ACCESS_DENIED);
    }
    if (GetLastError() != ERROR_NO_TOKEN) return us_fail(status, 4, GetLastError());
    status->checks |= 2u;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return us_fail(status, 5, GetLastError());
    BOOL valid = token_info(token, TokenIsAppContainer, &value, sizeof(value)) && value == 1;
    if (valid) status->checks |= 4u;
    if (valid) {
        valid = token_info(token, TokenCapabilities, data, sizeof(data)) && ((TOKEN_GROUPS *)data)->GroupCount == 0;
        if (valid) status->checks |= 8u;
    }
    if (valid) {
        valid = package_matches(token, package_sid);
        if (valid) status->checks |= 16u;
    }
    if (valid) {
        valid = token_info(token, TokenIntegrityLevel, data, sizeof(data));
        if (valid) {
            PSID sid = ((TOKEN_MANDATORY_LABEL *)data)->Label.Sid;
            SID_IDENTIFIER_AUTHORITY mandatory = SECURITY_MANDATORY_LABEL_AUTHORITY;
            valid = IsValidSid(sid) && *GetSidSubAuthorityCount(sid) == 1
                && !memcmp(GetSidIdentifierAuthority(sid), &mandatory, sizeof(mandatory))
                && *GetSidSubAuthority(sid, 0) == SECURITY_MANDATORY_LOW_RID;
        }
        if (valid) status->checks |= 32u;
    }
    DWORD error = GetLastError();
    if (!CloseHandle(token)) return us_fail(status, 5, GetLastError());
    if (!valid) return us_fail(status, 5, error);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job;
    memset(&job, 0, sizeof(job));
    if (!QueryInformationJobObject(NULL, JobObjectExtendedLimitInformation, &job, sizeof(job), NULL))
        return us_fail(status, 6, GetLastError());
    DWORD flags = job.BasicLimitInformation.LimitFlags;
    if (job.BasicLimitInformation.ActiveProcessLimit != 1 || !(flags & JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
        || !(flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        || (flags & (JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)))
        return us_fail(status, 6, ERROR_ACCESS_DENIED);
    status->checks |= 64u;
    HANDLE aap = CreateFileW(aap_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    error = GetLastError();
    if (aap != INVALID_HANDLE_VALUE) { CloseHandle(aap); return us_fail(status, 7, ERROR_ACCESS_DENIED); }
    if (error != ERROR_ACCESS_DENIED) return us_fail(status, 7, error);
    status->checks |= 128u;
    /* This is an own-thread authority audit, never descendant cleanup polling.
       Reviewed startup initializers must not race it with new token-bearing
       threads; retained handles/state still require workload qualification. */
    if (!all_thread_tokens_absent()) return us_fail(status, 8, GetLastError());
    status->checks |= 256u;
    status->phase = US_READY;
    status->stage = US_READY_STAGE;
    status->error = 0;
    return TRUE;
}

BOOL us_send_status(HANDLE output, const UsStatus *status) {
    DWORD written = 0;
    BOOL ok = WriteFile(output, status, sizeof(*status), &written, NULL) && written == sizeof(*status);
    if (!CloseHandle(output)) ok = FALSE;
    return ok;
}

BOOL us_wait_acknowledgement(HANDLE input, const UsBinding *binding) {
    UsAcknowledgement ack;
    DWORD count = 0, received = 0;
    BOOL valid = FALSE;
    while (received < sizeof(ack)) {
        if (!ReadFile(input, (BYTE *)&ack + received, (DWORD)sizeof(ack) - received, &count, NULL) || !count)
            goto done;
        received += count;
    }
    if (memcmp(ack.magic, "USLACK1", 8) || ack.version != US_PROTOCOL_VERSION || ack.reserved
        || memcmp(&ack.binding, binding, sizeof(*binding))) goto done;
    /* EOF is required: payload code must not inherit a live startup channel. */
    BYTE extra;
    if (ReadFile(input, &extra, 1, &count, NULL) || GetLastError() != ERROR_BROKEN_PIPE) goto done;
    valid = TRUE;
done:
    if (!CloseHandle(input)) valid = FALSE;
    return valid;
}
