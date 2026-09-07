#include <windows.h>
int wmain(int argc, wchar_t **argv) {
    if (argc != 2) return 2;
    HMODULE library = LoadLibraryExW(argv[1], NULL, LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
    if (!library) return 3;
    return FreeLibrary(library) ? 0 : 4;
}
