#include <intrin.h>
/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. */
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include "host_config.h"
#include <sddl.h>
#include <string.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_MAJOR_VERSION != US_PY_MAJOR || PY_MINOR_VERSION != US_PY_MINOR
#error CPython headers do not match the selected ABI adapter
#endif
#if defined(Py_DEBUG) || defined(Py_GIL_DISABLED) || !defined(_WIN64)
#error This profile requires release GIL-enabled x64 CPython
#endif

#define PY_API(X) \
    X(void, PyPreConfig_InitIsolatedConfig, (PyPreConfig *)) \
    X(PyStatus, Py_PreInitialize, (const PyPreConfig *)) \
    X(void, PyConfig_InitIsolatedConfig, (PyConfig *)) \
    X(PyStatus, PyConfig_SetString, (PyConfig *, wchar_t **, const wchar_t *)) \
    X(PyStatus, PyWideStringList_Append, (PyWideStringList *, const wchar_t *)) \
    X(PyStatus, Py_InitializeFromConfig, (const PyConfig *)) \
    X(void, PyConfig_Clear, (PyConfig *)) \
    X(int, PyStatus_Exception, (PyStatus)) \
    X(const char *, Py_GetVersion, (void)) \
    X(PyObject *, PyImport_ImportModule, (const char *)) \
    X(PyObject *, PyUnicode_FromWideChar, (const wchar_t *, Py_ssize_t)) \
    X(PyObject *, PyList_New, (Py_ssize_t)) \
    X(int, PyList_SetItem, (PyObject *, Py_ssize_t, PyObject *)) \
    X(int, PySys_SetObject, (const char *, PyObject *)) \
    X(void, Py_DecRef, (PyObject *)) \
    X(void, PyErr_Print, (void)) \
    X(int, PyRun_SimpleStringFlags, (const char *, PyCompilerFlags *)) \
    X(int, Py_FinalizeEx, (void))
#define DECLARE(result, name, args) result (*name) args;
typedef struct { PY_API(DECLARE) } PythonApi;
#undef DECLARE

static BOOL load_api(HMODULE module, PythonApi *api) {
#define RESOLVE(result, name, args) { \
    FARPROC address = GetProcAddress(module, #name); \
    if (!address) return FALSE; \
    _Static_assert(sizeof(api->name) == sizeof(address), "function pointer ABI mismatch"); \
    memcpy(&api->name, &address, sizeof(address)); \
}
    PY_API(RESOLVE)
#undef RESOLVE
    return TRUE;
}

static HMODULE load_runtime_image(const wchar_t *path) {
    /* Input paths have already passed canonical local-drive validation. Do not
       accept extended/UNC paths from configuration; add the Win32 length prefix
       only here, without changing DLL search directories or resolving aliases. */
    size_t length = wcslen(path);
    wchar_t *extended = (wchar_t *)malloc((length + 5) * sizeof(wchar_t));
    if (!extended) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); return NULL; }
    memcpy(extended, L"\\\\?\\", 4 * sizeof(wchar_t));
    memcpy(extended + 4, path, (length + 1) * sizeof(wchar_t));
    HMODULE module = LoadLibraryExW(extended, NULL,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    DWORD error = GetLastError();
    free(extended);
    SetLastError(error);
    return module;
}
static BOOL version_matches(PythonApi *api, const UsConfigHeader *header) {
    const char *version = api->Py_GetVersion();
    const uint32_t expected[] = {header->major, header->minor, header->patch};
    if (!version || header->major != PY_MAJOR_VERSION || header->minor != PY_MINOR_VERSION)
        return FALSE;
    for (size_t i = 0; i < 3; ++i) {
        unsigned value = 0, digits = 0;
        while (*version >= '0' && *version <= '9') {
            if (++digits > 5) return FALSE;
            value = value * 10 + (unsigned)(*version++ - '0');
        }
        if (!digits || value != expected[i] || *version++ != (i == 2 ? ' ' : '.')) return FALSE;
    }
    return TRUE;
}

static BOOL initialize_python(PythonApi *api, const UsConfig *launch) {
    PyPreConfig pre;
    api->PyPreConfig_InitIsolatedConfig(&pre);
    pre.utf8_mode = 1;
    if (api->PyStatus_Exception(api->Py_PreInitialize(&pre))) return FALSE;
    PyConfig config;
    api->PyConfig_InitIsolatedConfig(&config);
    config.use_environment = 0;
    config.site_import = 0;
    config.user_site_directory = 0;
    config.parse_argv = 0;
    config.write_bytecode = 0;
    config.install_signal_handlers = 0;
    config.buffered_stdio = 0;
    config.safe_path = 1;
    config.module_search_paths_set = 1;
    BOOL ok = FALSE;
#define SET(field, value) if (api->PyStatus_Exception(api->PyConfig_SetString(&config, &config.field, value))) goto done
    SET(stdio_encoding, L"utf-8");
    SET(stdio_errors, L"backslashreplace");
    SET(program_name, launch->values[US_EXECUTABLE]);
    SET(executable, launch->values[US_EXECUTABLE]);
    SET(base_executable, launch->values[US_EXECUTABLE]);
    SET(home, launch->values[US_RUNTIME_HOME]);
    /* Prefixes used for startup remain inside protected content. User-visible
       selected/venv prefixes are restored only after the native drop gate. */
    SET(prefix, launch->values[US_RUNTIME_HOME]);
    SET(exec_prefix, launch->values[US_RUNTIME_HOME]);
    SET(base_prefix, launch->values[US_RUNTIME_HOME]);
    SET(base_exec_prefix, launch->values[US_RUNTIME_HOME]);
    SET(stdlib_dir, launch->values[US_RUNTIME_HOME]);
#undef SET
    if (api->PyStatus_Exception(api->PyWideStringList_Append(&config.module_search_paths, launch->values[US_STDLIB]))
        || api->PyStatus_Exception(api->PyWideStringList_Append(&config.module_search_paths, launch->values[US_NATIVE_DIR]))
        || api->PyStatus_Exception(api->PyWideStringList_Append(&config.argv, L""))) goto done;
    ok = !api->PyStatus_Exception(api->Py_InitializeFromConfig(&config));
done:
    api->PyConfig_Clear(&config);
    return ok;
}

static BOOL warm_providers(void) {
    for (int family = 0; family < 2; ++family) {
        for (int kind = 0; kind < 2; ++kind) {
            SOCKET socket_handle = socket(family ? AF_INET6 : AF_INET, kind ? SOCK_DGRAM : SOCK_STREAM, 0);
            if (socket_handle == INVALID_SOCKET) return FALSE;
            if (closesocket(socket_handle)) return FALSE;
        }
    }
    return TRUE;
}

static BOOL publish_post_drop_config(PythonApi *api, const UsConfig *launch) {
    size_t count = US_CONFIG_FIELDS + launch->header.packages + launch->header.arguments;
    PyObject *values = api->PyList_New((Py_ssize_t)count);
    if (!values) return FALSE;
    BOOL ok = FALSE;
    for (size_t i = 0; i < count; ++i) {
        PyObject *text = api->PyUnicode_FromWideChar(launch->values[i], -1);
        if (!text || api->PyList_SetItem(values, (Py_ssize_t)i, text)) goto done;
    }
    if (api->PySys_SetObject("_unsloth_host_paths", values)) goto done;
    /* Counts are encoded as a second list to keep all Python data conversions
       explicit. Neither list is visible before the restricted phase. */
    wchar_t counts[32];
    if (swprintf_s(counts, 32, L"%u", launch->header.packages) < 0) goto done;
    PyObject *packages = api->PyUnicode_FromWideChar(counts, -1);
    if (!packages) goto done;
    ok = api->PySys_SetObject("_unsloth_host_package_count", packages) == 0;
    api->Py_DecRef(packages);
done:
    api->Py_DecRef(values);
    return ok;
}

/* These fixed programs run only after the native gate. Paths are Python
   objects, never string-interpolated into executable configuration. */
static const char *setup =
    "import sys, types, os\n"
    "def _unsloth_load(path, name):\n"
    "    module = types.ModuleType(name)\n"
    "    module.__file__ = path\n"
    "    with open(path, 'rb') as source:\n"
    "        code = compile(source.read(), path, 'exec', dont_inherit=True)\n"
    "    exec(code, module.__dict__)\n"
    "    return module\n"
    "_cfg = sys._unsloth_host_paths\n"
    "_unsloth_policy = _unsloth_load(_cfg[7], '_unsloth_process_policy')\n"
    "_unsloth_policy.install_single_process_policy()\n"
    "_unsloth_shim = _unsloth_load(_cfg[8], '_unsloth_path_shim')\n"
    "sys.prefix = sys.exec_prefix = _cfg[5]\n"
    "sys.base_prefix = sys.base_exec_prefix = _cfg[6]\n";

static const char *payload =
    "def _unsloth_run():\n"
    "    cfg = sys._unsloth_host_paths\n"
    "    count = int(sys._unsloth_host_package_count)\n"
    "    del sys._unsloth_host_paths, sys._unsloth_host_package_count\n"
    "    sys.argv = [cfg[10], *cfg[14+count:]]\n"
    "    sys.path[:0] = [cfg[9], *cfg[14:14+count]]\n"
    "    main = types.ModuleType('__main__')\n"
    "    main.__file__ = cfg[10]\n"
    "    main.__package__ = None\n"
    "    main.__spec__ = None\n"
    "    sys.modules['__main__'] = main\n"
    "    with open(cfg[10], 'rb') as source:\n"
    "        code = compile(source.read(), cfg[10], 'exec', dont_inherit=True)\n"
    "    exec(code, main.__dict__)\n"
    "_unsloth_run()\n";

static int run_host(int argc, wchar_t **argv) {
    if (argc != 4) return 90;
    HANDLE input = us_handle_argument(argv[1]), output = us_handle_argument(argv[2]), ack = us_handle_argument(argv[3]);
    if (!input || !output || !ack || input == output || input == ack || output == ack) return 90;
    UsConfig launch;
    if (!us_read_config(input, &launch)) { CloseHandle(output); CloseHandle(ack); return 91; }
    UsStatus status;
    us_status_init(&status, &launch.header.binding);
    PSID sid = NULL;
    PythonApi api = {0};
    HMODULE python = NULL;
    BOOL winsock = FALSE;
    HMODULE native_images[US_CONFIG_LIST] = {0};
    int result = 92;
    if (!ConvertStringSidToSidW(launch.values[US_PACKAGE_SID], &sid)
        || !us_validate_startup(sid, launch.values[US_AAP], &status)) goto failed;
    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)) { us_fail(&status, 10, GetLastError()); goto failed; }
    python = load_runtime_image(launch.values[US_RUNTIME_DLL]);
    if (!python) { us_fail(&status, 10, GetLastError()); goto failed; }
    if (!load_api(python, &api)) { us_fail(&status, 10, ERROR_PROC_NOT_FOUND); goto failed; }
    if (!version_matches(&api, &launch.header)) {
        us_fail(&status, 10, ERROR_REVISION_MISMATCH); goto failed;
    }
    if (!initialize_python(&api, &launch)) { us_fail(&status, 11, ERROR_DLL_INIT_FAILED); goto failed; }
    /* Load only the broker-admitted CPython feature graph. Windows activation
       context creation can require startup access even when image read/execute
       succeeds under LPAC. PyInit functions still run in the payload phase,
       except for the separately approved _overlapped initialization below. */
    size_t image_start = US_CONFIG_FIELDS + launch.header.packages + launch.header.arguments;
    for (size_t i = 0; i < launch.header.images; ++i) {
        native_images[i] = load_runtime_image(launch.values[image_start+i]);
        if (!native_images[i]) { us_fail(&status, 11, GetLastError()); goto failed; }
    }
    WSADATA wsa;
    int error = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (error) { us_fail(&status, 11, (DWORD)error); goto failed; }
    winsock = TRUE;
    if (!warm_providers()) { us_fail(&status, 11, (DWORD)WSAGetLastError()); goto failed; }
    PyObject *overlapped = api.PyImport_ImportModule("_overlapped");
    if (!overlapped) { api.PyErr_Print(); us_fail(&status, 11, ERROR_DLL_INIT_FAILED); goto failed; }
    api.Py_DecRef(overlapped);
    if (!us_drop_and_validate(sid, launch.values[US_AAP], &status)) goto failed;
    if (!SetCurrentDirectoryW(launch.values[US_WORKDIR]) || !publish_post_drop_config(&api, &launch)
        || api.PyRun_SimpleStringFlags(setup, NULL)) { us_fail(&status, 12, ERROR_DLL_INIT_FAILED); goto failed; }
    us_free_config(&launch);
    LocalFree(sid);
    sid = NULL;
    if (!us_send_status(output, &status)) { CloseHandle(ack); return 93; }
    if (!us_wait_acknowledgement(ack, &status.binding)) return 94;
    result = api.PyRun_SimpleStringFlags(payload, NULL) ? 1 : 0;
    if (api.Py_FinalizeEx()) result = 120;
    if (winsock) WSACleanup();
    for (size_t i = US_CONFIG_LIST; i; --i) if (native_images[i-1]) FreeLibrary(native_images[i-1]);
    FreeLibrary(python);
    return result;
failed:
    /* Startup failure must never run finalizers under a retained startup token.
       The outer entry point terminates without loader callbacks; the broker reaps the Job before
       releasing the generation. No retry or payload callback is possible. */
    us_free_config(&launch);
    if (sid) LocalFree(sid);
    us_send_status(output, &status);
    CloseHandle(ack);
    return result;
}

/* Failure must not return into CRT teardown while startup authority is retained.
   TerminateProcess skips DLL detach and atexit callbacks, including adversarial
   runtime callbacks registered before the irreversible drop. */
int wmain(int argc, wchar_t **argv) {
    int result = run_host(argc, argv);
    if (result != 0) {
        TerminateProcess(GetCurrentProcess(), (UINT)result);
        __fastfail(7);
    }
    return result;
}
