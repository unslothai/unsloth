#include <windows.h>
#include <wchar.h>
static void marker(const wchar_t *name) {
    wchar_t path[32768];
    DWORD count = GetEnvironmentVariableW(L"UNSLOTH_TEST_DETACH_DIR", path, 32700);
    if (!count || count >= 32700) return;
    if (wcscat_s(path, 32768, name)) return;
    HANDLE file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
}
BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)module; (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) marker(L"\\attached");
    if (reason == DLL_PROCESS_DETACH) marker(L"\\detached");
    return TRUE;
}
