/*
 * SPDX-License-Identifier: AGPL-3.0-only
 * Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
 *
 * Unsloth Studio Setup — C port of studio/setup.sh
 *
 * Build:
 *   Linux   : make
 *   Windows : make windows   (requires mingw-w64)
 */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  include <direct.h>
#  define PATH_SEP      "\\"
#  define popen         _popen
#  define pclose        _pclose
#  define isatty        _isatty
#  define fileno        _fileno
#  define DEVNULL       "NUL"
#  define SYS_MKDIR(p)  _mkdir(p)
#  ifndef PATH_MAX
#    define PATH_MAX    4096
#  endif
#else
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <dirent.h>
#  include <limits.h>
#  define PATH_SEP      "/"
#  define DEVNULL       "/dev/null"
#  define SYS_MKDIR(p)  mkdir((p), 0755)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <time.h>

/* ── All path buffers use this size (== PATH_MAX, at least 4096) ── */
#define PATHSZ  PATH_MAX

/* ── Command buffers: path * 3 + fixed overhead ── */
#define CMDSZ   (PATH_MAX * 3 + 512)

/* ── Safe snprintf — aborts on truncation ── */
static void safe_snprintf(char *buf, size_t bufsz, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, bufsz, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= bufsz) {
        fprintf(stderr, "FATAL: buffer too small (%zu) for formatted string (%d chars)\n",
                bufsz, n);
        abort();
    }
}
#define SNPRINTF(buf, ...) safe_snprintf(buf, sizeof(buf), __VA_ARGS__)

/* ── Fire-and-forget system() / chdir() — discard return value cleanly ── */
static inline int _nochk(int r) { return r; }
#define SYS(x) _nochk(x)

/* ══════════════════════════════════════════════════════════════════════
 *  Platform helpers
 * ══════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
static int path_exists(const char *p)
{
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
}
static int is_dir(const char *p)
{
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
static int is_exec(const char *p) { return path_exists(p); }
#else
static int path_exists(const char *p) { struct stat s; return stat(p,&s)==0; }
static int is_dir(const char *p)      { struct stat s; return stat(p,&s)==0 && S_ISDIR(s.st_mode); }
static int is_exec(const char *p)     { return access(p, X_OK) == 0; }
#endif

/* mkdir -p equivalent */
static void mkdirp(const char *path)
{
    char cmd[CMDSZ];
#ifdef _WIN32
    SNPRINTF(cmd, "mkdir \"%s\" 2>NUL", path);
#else
    SNPRINTF(cmd, "mkdir -p '%s'", path);
#endif
    SYS(system(cmd));
}

/* rm -rf equivalent */
static void rm_rf(const char *path)
{
    if (!path_exists(path)) return;
    char cmd[CMDSZ];
#ifdef _WIN32
    SNPRINTF(cmd, "rmdir /S /Q \"%s\" 2>NUL", path);
#else
    SNPRINTF(cmd, "rm -rf '%s'", path);
#endif
    SYS(system(cmd));
}

/* Portable dirname — strips last component in-place */
static void dirname_inplace(char *path)
{
    char *last = NULL;
    for (char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') last = p;
    if (last && last != path) *last = '\0';
    else if (!last) strcpy(path, ".");
}

/* ══════════════════════════════════════════════════════════════════════
 *  Color
 * ══════════════════════════════════════════════════════════════════════ */

static int g_color = 0;

#define C_TITLE  "\033[38;5;150m"
#define C_DIM    "\033[38;5;245m"
#define C_OK     "\033[38;5;108m"
#define C_WARN   "\033[38;5;136m"
#define C_ERR    "\033[91m"
#define C_RST    "\033[0m"

static const char *col(const char *c) { return g_color ? c : ""; }

#define RULE \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200\342\224\200" \
    "\342\224\200\342\224\200\342\224\200\342\224\200"

/* ══════════════════════════════════════════════════════════════════════
 *  Output helpers
 * ══════════════════════════════════════════════════════════════════════ */

static void step(const char *label, const char *msg, const char *color)
{
    if (!color) color = col(C_OK);
    printf("  %s%-15.15s%s%s%s%s\n",
           col(C_DIM), label, col(C_RST), color, msg, col(C_RST));
}

static void substep(const char *msg)
{
    printf("  %-15s%s\n", "", msg);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Subprocess helpers
 * ══════════════════════════════════════════════════════════════════════ */

/* Read first line of command output; returns 0 on success */
static int cmd_output(const char *cmd, char *buf, size_t bufsz)
{
    FILE *fp = popen(cmd, "r");
    if (!fp) { buf[0] = '\0'; return -1; }
    buf[0] = '\0';
    if (fgets(buf, (int)bufsz, fp)) {
        size_t n = strlen(buf);
        while (n && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
    }
    int rc = pclose(fp);
#ifndef _WIN32
    return WIFEXITED(rc) && WEXITSTATUS(rc) == 0 ? 0 : -1;
#else
    return rc == 0 ? 0 : -1;
#endif
}

/* Returns 1 if command is found in PATH */
static int has_cmd(const char *prog)
{
    char cmd[256];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "where \"%s\" >NUL 2>&1", prog);
#else
    snprintf(cmd, sizeof(cmd), "command -v '%s' >/dev/null 2>&1", prog);
#endif
    return system(cmd) == 0;
}

/*
 * run_quiet / run_quiet_no_exit
 *
 * Runs `cmd` in a shell. On success: silent, returns 0.
 * On failure: prints captured output to stderr, then either exits or returns
 * the non-zero exit code.
 */
static int _run_quiet(int fatal, const char *label, const char *cmd)
{
    /* Build temp-file path */
    char tmpfile[PATHSZ];
#ifdef _WIN32
    char tdir[PATHSZ];
    if (!GetTempPathA((DWORD)sizeof(tdir), tdir)) strcpy(tdir, ".");
    char tname[MAX_PATH];
    GetTempFileNameA(tdir, "usl", 0, tname);
    snprintf(tmpfile, sizeof(tmpfile), "%s", tname);
#else
    snprintf(tmpfile, sizeof(tmpfile), "/tmp/unsloth_setup_%d_%ld",
             (int)getpid(), (long)time(NULL));
#endif

    /* Redirect both stdout and stderr to tempfile */
    char full_cmd[CMDSZ];
    snprintf(full_cmd, sizeof(full_cmd), "%s >\"%s\" 2>&1", cmd, tmpfile);
    if ((size_t)strlen(full_cmd) >= sizeof(full_cmd) - 1) {
        fprintf(stderr, "FATAL: command too long for buffer\n");
        abort();
    }

    int raw = system(full_cmd);
    int exit_code = 0;
#ifndef _WIN32
    if (WIFEXITED(raw)) exit_code = WEXITSTATUS(raw); else exit_code = 1;
#else
    exit_code = raw;
#endif

    if (exit_code == 0) { remove(tmpfile); return 0; }

    /* Print failure log */
    char errmsg[512];
    snprintf(errmsg, sizeof(errmsg), "%s failed (exit code %d)", label, exit_code);
    step("error", errmsg, col(C_ERR));

    FILE *fp = fopen(tmpfile, "r");
    if (fp) {
        char line[1024];
        while (fgets(line, sizeof(line), fp)) fputs(line, stderr);
        fclose(fp);
    }
    remove(tmpfile);

    if (fatal) exit(exit_code);
    return exit_code;
}

#define run_quiet(label, cmd)         _run_quiet(1, label, cmd)
#define run_quiet_no_exit(label, cmd) _run_quiet(0, label, cmd)

/* ══════════════════════════════════════════════════════════════════════
 *  Colab detection
 * ══════════════════════════════════════════════════════════════════════ */

static int detect_colab(void)
{
#ifdef _WIN32
    return 0;
#else
    return system("printenv | cut -d= -f1 | grep -q '^COLAB_'") == 0;
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 *  Frontend
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * node_bin_dir()
 *
 * Finds the directory containing the `node` binary, respecting nvm if
 * installed.  Returns 1 and fills `out` on success, 0 if node is not found.
 *
 * We must ask the shell to source nvm and then print $NVM_BIN (or `which
 * node`) because nvm is a shell *function*, not an executable — it cannot be
 * invoked from C's system() without sourcing nvm.sh first.
 */
static int node_bin_dir(char *out, size_t outsz)
{
    /* Try plain PATH first (node already installed system-wide or by distro) */
    char plain[PATHSZ]={0};
    if (cmd_output("command -v node 2>/dev/null", plain, sizeof(plain)) == 0 && plain[0]) {
        /* strip "/node" suffix to get the bin dir */
        char *slash = strrchr(plain, '/');
        if (slash) { *slash = '\0'; snprintf(out, outsz, "%s", plain); return 1; }
    }

    /* Try nvm — source nvm.sh and ask where node landed */
    char nvm_check[PATHSZ]={0};
    cmd_output(
        "export NVM_DIR=\"$HOME/.nvm\" && "
        "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && "
        "command -v node 2>/dev/null",
        nvm_check, sizeof(nvm_check));
    if (nvm_check[0]) {
        char *slash = strrchr(nvm_check, '/');
        if (slash) { *slash = '\0'; snprintf(out, outsz, "%s", nvm_check); return 1; }
    }
    return 0;
}

/*
 * prepend_path()
 *
 * Prepends `dir` to the C process's PATH so subsequent system()/popen()
 * calls (which inherit PATH) can find node/npm/bun without sourcing nvm.
 */
static void prepend_path(const char *dir)
{
#ifdef _WIN32
    const char *cur = getenv("PATH");
    char newpath[CMDSZ];
    /* Windows uses ; as path separator */
    snprintf(newpath, sizeof(newpath), "%s;%s", dir, cur ? cur : "");
    SetEnvironmentVariableA("PATH", newpath);
#else
    const char *cur = getenv("PATH");
    char newpath[CMDSZ];
    snprintf(newpath, sizeof(newpath), "%s:%s", dir, cur ? cur : "");
    setenv("PATH", newpath, 1);
#endif
}

static void setup_node(int is_colab)
{
    /* ── 1. Check whether a good node+npm is already reachable ── */
    int need_node = 1;

    /* Source nvm.sh for this check too, in case user already has nvm+node.
     * The || true ensures we don't fail when nvm isn't installed yet. */
    char maj_s[32]={0}, min_s[32]={0}, npm_s[32]={0};
    cmd_output(
        "{ export NVM_DIR=\"$HOME/.nvm\" && "
        "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"; } 2>/dev/null || true; "
        "node -v 2>/dev/null | sed 's/v//' | cut -d. -f1",
        maj_s, sizeof(maj_s));
    cmd_output(
        "{ export NVM_DIR=\"$HOME/.nvm\" && "
        "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"; } 2>/dev/null || true; "
        "node -v 2>/dev/null | sed 's/v//' | cut -d. -f2",
        min_s, sizeof(min_s));
    cmd_output(
        "{ export NVM_DIR=\"$HOME/.nvm\" && "
        "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"; } 2>/dev/null || true; "
        "npm -v 2>/dev/null | cut -d. -f1",
        npm_s, sizeof(npm_s));

    int nmaj = atoi(maj_s), nmin = atoi(min_s), npmmaj = atoi(npm_s);
    int node_ok = (nmaj == 20 && nmin >= 19) ||
                  (nmaj == 22 && nmin >= 12) ||
                  (nmaj >= 23);

    if (node_ok && npmmaj >= 11) {
        need_node = 0;
    } else if (is_colab && node_ok) {
        if (npmmaj < 11) {
            substep("upgrading npm...");
            SYS(system(
                "export NVM_DIR=\"$HOME/.nvm\" && "
                "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"; "
                "npm install -g npm@latest >" DEVNULL " 2>&1"));
        }
        need_node = 0;
    }

#ifndef _WIN32
    /* ── 2. Install nvm + Node LTS if needed ── */
    if (need_node) {
        substep("installing nvm...");

        /*
         * Do everything in ONE shell invocation so nvm.sh is sourced once
         * and node ends up on PATH for the install + version-check steps.
         * We write a small script to a temp file to keep the quoting sane.
         */
        char script_tmp[PATHSZ];
        snprintf(script_tmp, sizeof(script_tmp),
                 "/tmp/unsloth_nvm_install_%d.sh", (int)getpid());

        FILE *sf = fopen(script_tmp, "w");
        if (!sf) { step("node", "cannot create temp script", col(C_ERR)); exit(1); }
        fprintf(sf,
            "#!/bin/sh\n"
            "set -e\n"
            /* remove broken prefix/globalconfig lines that break nvm */
            "if [ -f \"$HOME/.npmrc\" ]; then\n"
            "  sed -i.bak '/^\\s*\\(prefix\\|globalconfig\\)\\s*=/d' \"$HOME/.npmrc\" 2>/dev/null || true\n"
            "fi\n"
            /* Source nvm if already installed — non-fatal if absent */
            "export NVM_DIR=\"$HOME/.nvm\"\n"
            "NVM_SOURCED=0\n"
            "if [ -s \"$NVM_DIR/nvm.sh\" ]; then\n"
            "  . \"$NVM_DIR/nvm.sh\"\n"
            "  NVM_SOURCED=1\n"
            "fi\n"
            /* Check if a good enough node is already present */
            "NODE_MAJ=0\n"
            "if command -v node >/dev/null 2>&1; then\n"
            "  NODE_MAJ=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)\n"
            "fi\n"
            "NODE_GOOD=0\n"
            "if [ \"${NODE_MAJ:-0}\" -ge 20 ]; then NODE_GOOD=1; fi\n"
            /* Only install nvm+node if we don't already have a good version */
            "if [ \"$NODE_GOOD\" = \"0\" ]; then\n"
            "  export NODE_OPTIONS=--dns-result-order=ipv4first\n"
            "  curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash >/dev/null 2>&1\n"
            "  [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"\n"
            "  nvm install --lts >/dev/null 2>&1\n"
            "  nvm use --lts    >/dev/null 2>&1\n"
            "fi\n"
            /* upgrade npm if needed */
            "NPM_MAJ=$(npm -v 2>/dev/null | cut -d. -f1)\n"
            "if [ \"${NPM_MAJ:-0}\" -lt 11 ]; then\n"
            "  npm install -g npm@latest >/dev/null 2>&1 || true\n"
            "fi\n"
            /* print the node binary path — C reads this to update PATH */
            "command -v node\n"
        );
        fclose(sf);
        chmod(script_tmp, 0700);

        char run_cmd[CMDSZ];
        snprintf(run_cmd, sizeof(run_cmd), "sh '%s'", script_tmp);

        /* Run and capture last line = path to node binary */
        char node_path[PATHSZ]={0};
        FILE *fp = popen(run_cmd, "r");
        if (fp) {
            char line[PATHSZ];
            while (fgets(line, sizeof(line), fp)) {
                size_t n = strlen(line);
                while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
                if (line[0]) snprintf(node_path, sizeof(node_path), "%s", line);
            }
            int rc = pclose(fp);
            if (!WIFEXITED(rc) || WEXITSTATUS(rc) != 0) {
                remove(script_tmp);
                step("node", "nvm/node install failed", col(C_ERR));
                exit(1);
            }
        }
        remove(script_tmp);

        if (!node_path[0]) {
            step("node", "node binary not found after nvm install", col(C_ERR));
            exit(1);
        }

        /* node_path is e.g. /home/user/.nvm/versions/node/v22.x.x/bin/node
         * Strip "/node" to get the bin dir and prepend it to our PATH       */
        char *slash = strrchr(node_path, '/');
        if (slash) {
            *slash = '\0';
            prepend_path(node_path);   /* now node/npm/npx are on C's PATH */
        }
    }
#else  /* _WIN32 */
    if (need_node) {
        step("node", "Please install Node >=20.19.0 from https://nodejs.org", col(C_ERR));
        exit(1);
    }
#endif

    /* ── 3. If nvm node is already installed but not on PATH, fix that ── */
    if (!has_cmd("node")) {
        char bin[PATHSZ]={0};
        if (node_bin_dir(bin, sizeof(bin))) prepend_path(bin);
    }

    /* ── 4. Final version check ── */
    {
        char mj[32]={0};
        cmd_output("node -v 2>/dev/null | sed 's/v//' | cut -d. -f1", mj, sizeof(mj));
        if (atoi(mj) < 20) {
            step("node", "FAILED -- version must be >= 20", col(C_ERR));
            exit(1);
        }
    }

    char nv[64]={0}, npmv[64]={0};
    cmd_output("node -v 2>/dev/null", nv,   sizeof(nv));
    cmd_output("npm -v  2>/dev/null", npmv, sizeof(npmv));
    char msg[256]; SNPRINTF(msg, "%s | npm %s", nv, npmv);
    step("node", msg, NULL);
}

static void install_bun(void)
{
    if (!has_cmd("bun")) {
        substep("installing bun...");
        if (system("npm install -g bun >" DEVNULL " 2>&1") == 0 && has_cmd("bun")) {
            char bv[64]={0}; cmd_output("bun --version 2>/dev/null", bv, sizeof(bv));
            char msg[128]; SNPRINTF(msg, "bun installed (%s)", bv); substep(msg);
        } else {
            substep("bun install skipped (npm will be used instead)");
        }
    } else {
        char bv[64]={0}; cmd_output("bun --version 2>/dev/null", bv, sizeof(bv));
        char msg[128]; SNPRINTF(msg, "bun already installed (%s)", bv); substep(msg);
    }
}

static void build_frontend(const char *script_dir)
{
    char dist_dir[PATHSZ];
    SNPRINTF(dist_dir, "%s/frontend/dist", script_dir);

    /* Check if build is up to date */
    if (is_dir(dist_dir)) {
        char check[CMDSZ];
        SNPRINTF(check,
            "find '%s/frontend' -maxdepth 1 -type f ! -name 'bun.lock'"
            " -newer '%s' -print -quit 2>/dev/null | grep -q .",
            script_dir, dist_dir);
        int changed = (system(check) == 0);
        if (!changed) {
            SNPRINTF(check,
                "find '%s/frontend/src' '%s/frontend/public'"
                " -type f -newer '%s' -print -quit 2>/dev/null | grep -q .",
                script_dir, script_dir, dist_dir);
            changed = (system(check) == 0);
        }
        if (!changed) { step("frontend", "up to date", NULL); return; }
    }

    setup_node(detect_colab());
    install_bun();
    substep("building frontend...");

    char fe_dir[PATHSZ];
    SNPRINTF(fe_dir, "%s/frontend", script_dir);
    if (chdir(fe_dir) != 0) {
        char errmsg[PATHSZ + 64];
        snprintf(errmsg, sizeof(errmsg), "cannot chdir to %s: %s", fe_dir, strerror(errno));
        step("frontend", errmsg, col(C_ERR)); exit(1);
    }

    /* bun install with fallback to npm */
    int bun_ok = 0;
    if (has_cmd("bun")) {
        printf("   Using bun for package install (faster)\n");
        int rc = system("bun install >" DEVNULL " 2>&1");
        /* Prefer bun's own exit code; also accept npm-style .bin links when present */
        int bins_ok = is_exec("node_modules/.bin/tsc") || is_exec("node_modules/typescript/bin/tsc");
        if (rc == 0) {
            bun_ok = 1;
            if (!bins_ok)
                printf("   bun install succeeded (binaries may use bun-native paths)\n");
        } else {
            printf("   bun install failed -- clearing cache and retrying...\n");
            SYS(system("bun pm cache rm >" DEVNULL " 2>&1"));
            rm_rf("node_modules");
            rc = system("bun install >" DEVNULL " 2>&1");
            if (rc == 0) {
                bun_ok = 1;
            } else {
                printf("   bun install failed after retry -- falling back to npm\n");
            }
        }
    }
    if (!bun_ok) run_quiet("npm install", "npm install");
    /* Use bun run build when bun succeeded (faster); npm run build otherwise */
    if (bun_ok) run_quiet("npm run build", "bun run build");
    else        run_quiet("npm run build", "npm run build");

    /* CSS output size check */
    char css_cmd[CMDSZ];
    SNPRINTF(css_cmd,
        "find '%s/frontend/dist/assets' -name '*.css' -exec wc -c {} + 2>/dev/null"
        " | sort -n | tail -1 | awk '{print $1}'",
        script_dir);
    char max_css_s[32]={0}; cmd_output(css_cmd, max_css_s, sizeof(max_css_s));
    int max_css = atoi(max_css_s);
    if      (max_css == 0)       step("frontend", "built (warning: no CSS emitted)",       col(C_WARN));
    else if (max_css < 100000)   step("frontend", "built (warning: CSS may be truncated)", col(C_WARN));
    else                         step("frontend", "built", NULL);

    SYS(chdir(script_dir));
}

/* ══════════════════════════════════════════════════════════════════════
 *  oxc-validator
 * ══════════════════════════════════════════════════════════════════════ */

static void setup_oxc_validator(const char *script_dir)
{
    char oxc_dir[PATHSZ];
    SNPRINTF(oxc_dir, "%s/backend/core/data_recipe/oxc-validator", script_dir);
    if (is_dir(oxc_dir) && has_cmd("npm")) {
        SYS(chdir(oxc_dir));
        run_quiet("npm install (oxc validator runtime)", "npm install");
        SYS(chdir(script_dir));
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  Python
 * ══════════════════════════════════════════════════════════════════════ */

static void install_python_stack(const char *script_dir)
{
    char cmd[CMDSZ];
    SNPRINTF(cmd, "python '%s/install_python_stack.py'", script_dir);
    SYS(system(cmd));
}

static int ensure_uv(void)
{
    if (has_cmd("uv")) return 1;
    int rc = system("curl -LsSf https://astral.sh/uv/install.sh | sh >" DEVNULL " 2>&1");
    if (rc == 0) {
#ifndef _WIN32
        const char *home = getenv("HOME");
        const char *cur  = getenv("PATH");
        if (home && cur) {
            char newpath[CMDSZ];
            snprintf(newpath, sizeof(newpath), "%s/.local/bin:%s", home, cur);
            setenv("PATH", newpath, 1);
        }
#else
        (void)0;
#endif
        return has_cmd("uv");
    }
    return 0;
}

static void fast_install(int use_uv, const char *args)
{
    char cmd[CMDSZ];
    if (use_uv) {
        char py[PATHSZ]={0};
        cmd_output("command -v python 2>/dev/null", py, sizeof(py));
        snprintf(cmd, sizeof(cmd), "uv pip install --python '%s' %s", py, args);
    } else {
        snprintf(cmd, sizeof(cmd), "python -m pip install %s", args);
    }
    SYS(system(cmd));
}

static void setup_python(const char *script_dir, int is_colab)
{
    const char *home = getenv("HOME");
    if (!home) home = ".";

    char studio_home[PATHSZ], venv_dir[PATHSZ], venv_t5_dir[PATHSZ];
    SNPRINTF(studio_home, "%s/.unsloth/studio",  home);
    SNPRINTF(venv_dir,    "%s/unsloth_studio",   studio_home);
    SNPRINTF(venv_t5_dir, "%s/.venv_t5",         studio_home);

    /* Remove stale in-repo venvs */
    char tmp[PATHSZ];
    SNPRINTF(tmp, "%s/../.venv",         script_dir); rm_rf(tmp);
    SNPRINTF(tmp, "%s/../.venv_overlay", script_dir); rm_rf(tmp);
    SNPRINTF(tmp, "%s/../.venv_t5",      script_dir); rm_rf(tmp);

    char venv_python[PATHSZ];
    SNPRINTF(venv_python, "%s/bin/python", venv_dir);

    int colab_no_venv = 0;
    if (!is_exec(venv_python)) {
        if (is_colab) {
            substep("Colab detected, installing Studio backend dependencies...");
            char req_file[PATHSZ];
            SNPRINTF(req_file, "%s/backend/requirements/studio.txt", script_dir);
            char install_cmd[CMDSZ];
            SNPRINTF(install_cmd,
                "sed 's/[><=!~;].*//' '%s' | grep -v '^#' | grep -v '^$'"
                " | pip install -q -r /dev/stdin >/dev/null 2>&1",
                req_file);
            SYS(system(install_cmd));
            colab_no_venv = 1;
        } else {
            /* Auto-create the venv rather than asking the user to run install.sh */
            substep("creating Python environment (first run)...");
            mkdirp(studio_home);

            /* Prefer uv for venv creation (faster); fall back to python3 -m venv */
            int uv_available = ensure_uv();
            int venv_created = 0;
            if (uv_available) {
                char uv_cmd[CMDSZ];
                SNPRINTF(uv_cmd, "uv venv '%s' >" DEVNULL " 2>&1", venv_dir);
                venv_created = (system(uv_cmd) == 0 && is_exec(venv_python));
            }
            if (!venv_created) {
                char py_cmd[CMDSZ];
                SNPRINTF(py_cmd, "python3 -m venv '%s' >" DEVNULL " 2>&1", venv_dir);
                venv_created = (system(py_cmd) == 0 && is_exec(venv_python));
            }
            if (!venv_created) {
                char msg[PATHSZ + 64];
                snprintf(msg, sizeof(msg),
                    "could not create venv at %s -- ensure python3 or uv is installed", venv_dir);
                step("python", msg, col(C_ERR));
                exit(1);
            }

            /* Activate the freshly created venv for subsequent child processes */
#ifndef _WIN32
            {
                const char *cur = getenv("PATH");
                char bin_dir[PATHSZ], newpath[CMDSZ];
                SNPRINTF(bin_dir, "%s/bin", venv_dir);
                snprintf(newpath, sizeof(newpath), "%s:%s", bin_dir, cur ? cur : "");
                setenv("PATH",        newpath, 1);
                setenv("VIRTUAL_ENV", venv_dir, 1);
                SNPRINTF(venv_python, "%s/bin/python", venv_dir);
            }
#endif
            substep("environment created, installing packages...");
        }
    } else {
#ifndef _WIN32
        const char *cur = getenv("PATH");
        char bin_dir[PATHSZ], newpath[CMDSZ];
        SNPRINTF(bin_dir, "%s/bin", venv_dir);
        snprintf(newpath, sizeof(newpath), "%s:%s", bin_dir, cur ? cur : "");
        setenv("PATH",        newpath, 1);
        setenv("VIRTUAL_ENV", venv_dir, 1);
#endif
    }

    int use_uv = ensure_uv();

    /* Version check */
    int skip_deps = 0;
    const char *pkg = getenv("STUDIO_PACKAGE_NAME");
    if (!pkg) pkg = "unsloth";

    if (!colab_no_venv &&
        strcmp(getenv("SKIP_STUDIO_BASE")    ? getenv("SKIP_STUDIO_BASE")    : "0", "1") != 0 &&
        strcmp(getenv("STUDIO_LOCAL_INSTALL")? getenv("STUDIO_LOCAL_INSTALL"): "0", "1") != 0)
    {
        char ver_cmd[CMDSZ], installed[64]={0};
        snprintf(ver_cmd, sizeof(ver_cmd),
            "'%s' -c \"from importlib.metadata import version; print(version('%s'))\" 2>/dev/null",
            venv_python, pkg);
        cmd_output(ver_cmd, installed, sizeof(installed));

        char pypi_cmd[CMDSZ], latest[64]={0};
        snprintf(pypi_cmd, sizeof(pypi_cmd),
            "curl -fsSL --max-time 5 'https://pypi.org/pypi/%s/json' 2>/dev/null"
            " | '%s' -c \"import sys,json; print(json.load(sys.stdin)['info']['version'])\" 2>/dev/null",
            pkg, venv_python);
        cmd_output(pypi_cmd, latest, sizeof(latest));

        if (installed[0] && latest[0]) {
            if (strcmp(installed, latest) == 0) {
                char msg[256]; SNPRINTF(msg, "%s %s is up to date", pkg, installed);
                step("python", msg, NULL);
                skip_deps = 1;
            } else {
                char msg[256]; SNPRINTF(msg, "%s %s -> %s available, updating...", pkg, installed, latest);
                substep(msg);
            }
        } else if (!latest[0]) {
            substep("could not reach PyPI, updating to be safe...");
        }
    }

    if (!skip_deps) {
        install_python_stack(script_dir);
        mkdirp(venv_t5_dir);

        char arg[CMDSZ];
        SNPRINTF(arg, "--target '%s' --no-deps 'transformers==5.3.0'",    venv_t5_dir); fast_install(use_uv, arg);
        SNPRINTF(arg, "--target '%s' --no-deps 'huggingface_hub==1.7.1'", venv_t5_dir); fast_install(use_uv, arg);
        SNPRINTF(arg, "--target '%s' --no-deps 'hf_xet==1.4.2'",          venv_t5_dir); fast_install(use_uv, arg);
        SNPRINTF(arg, "--target '%s' 'tiktoken'",                          venv_t5_dir); fast_install(use_uv, arg);
        step("transformers", "5.x pre-installed", NULL);
    } else {
        step("python", "dependencies up to date", NULL);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  llama.cpp
 * ══════════════════════════════════════════════════════════════════════ */

static void resolve_llama_tag(const char *script_dir,
                               const char *requested_tag,
                               const char *helper_repo,
                               char *resolved_tag, size_t tsz,
                               int *need_source_build)
{
    *need_source_build = 0;
    resolved_tag[0] = '\0';

    char log_tmp[PATHSZ];
#ifdef _WIN32
    SNPRINTF(log_tmp, "%s\\..\\unsloth_llama_resolve.tmp", script_dir);
#else
    snprintf(log_tmp, tsz, "/tmp/unsloth_llama_resolve_%d", (int)getpid());
#endif

    char resolve_cmd[CMDSZ];
    SNPRINTF(resolve_cmd,
        "python '%s/install_llama_prebuilt.py'"
        " --resolve-install-tag '%s' --published-repo '%s' >'%s' 2>&1",
        script_dir, requested_tag, helper_repo, log_tmp);

    int rc = system(resolve_cmd);
    int status = 0;
#ifndef _WIN32
    if (WIFEXITED(rc)) status = WEXITSTATUS(rc); else status = 1;
#else
    status = rc;
#endif

    if (status == 0) {
        FILE *fp = fopen(log_tmp, "r");
        if (fp) {
            char line[256]; resolved_tag[0] = '\0';
            while (fgets(line, sizeof(line), fp)) {
                size_t n = strlen(line);
                while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
                if (line[0]) snprintf(resolved_tag, tsz, "%s", line);
            }
            fclose(fp);
        }
    }
    remove(log_tmp);

    if (!resolved_tag[0]) {
        char warn_msg[PATHSZ];
        snprintf(warn_msg, sizeof(warn_msg), "failed to resolve prebuilt tag via %s", helper_repo);
        step("llama.cpp", warn_msg, col(C_WARN));

        /* Fallback 1: resolve-llama-tag */
        char fallback_cmd[CMDSZ];
        SNPRINTF(fallback_cmd,
            "python '%s/install_llama_prebuilt.py'"
            " --resolve-llama-tag '%s' --published-repo '%s' 2>/dev/null",
            script_dir, requested_tag, helper_repo);
        cmd_output(fallback_cmd, resolved_tag, tsz);

        /* Fallback 2: GitHub API */
        if (!resolved_tag[0] && strcmp(requested_tag, "latest") == 0) {
            char api_cmd[CMDSZ];
            snprintf(api_cmd, sizeof(api_cmd),
                "curl -fsSL 'https://api.github.com/repos/%s/releases/latest' 2>/dev/null"
                " | python -c \"import sys,json; print(json.load(sys.stdin)['tag_name'])\" 2>/dev/null",
                helper_repo);
            cmd_output(api_cmd, resolved_tag, tsz);
        }
        if (!resolved_tag[0] && strcmp(requested_tag, "latest") == 0) {
            cmd_output(
                "curl -fsSL 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' 2>/dev/null"
                " | python -c \"import sys,json; print(json.load(sys.stdin)['tag_name'])\" 2>/dev/null",
                resolved_tag, tsz);
        }
        if (!resolved_tag[0]) snprintf(resolved_tag, tsz, "%s", requested_tag);
        *need_source_build = 1;
    }
}

#ifndef _WIN32
static void install_wsl_build_deps(void)
{
    FILE *fp = fopen("/proc/version", "r");
    if (!fp) return;
    char line[512]; int is_wsl = 0;
    if (fgets(line, sizeof(line), fp)) {
        /* case-insensitive search for "microsoft" */
        char lower[512]; int i;
        for (i = 0; line[i] && i < (int)sizeof(lower)-1; i++)
            lower[i] = (line[i] >= 'A' && line[i] <= 'Z') ? line[i]+32 : line[i];
        lower[i] = '\0';
        if (strstr(lower, "microsoft")) is_wsl = 1;
    }
    fclose(fp);
    if (!is_wsl) return;

    const char *deps = "pciutils build-essential cmake curl git libcurl4-openssl-dev";
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "apt-get update -y >/dev/null 2>&1 && apt-get install -y %s >/dev/null 2>&1", deps);
    SYS(system(cmd));
    step("gguf deps", "installed", NULL);
}
#endif

static void build_llama_source(const char *llama_dir, const char *resolved_tag)
{
    if (!has_cmd("cmake")) { step("llama.cpp", "skipped (cmake not found)", col(C_WARN)); return; }
    if (!has_cmd("git"))   { step("llama.cpp", "skipped (git not found)",   col(C_WARN)); return; }

    char build_tmp[PATHSZ + 32];
#ifdef _WIN32
    snprintf(build_tmp, sizeof(build_tmp), "%s.build.%lu", llama_dir, (unsigned long)GetCurrentProcessId());
#else
    snprintf(build_tmp, sizeof(build_tmp), "%s.build.%d", llama_dir, (int)getpid());
#endif
    rm_rf(build_tmp);

    char clone_cmd[CMDSZ];
    if (resolved_tag[0] && strcmp(resolved_tag, "latest") != 0)
        SNPRINTF(clone_cmd,
            "git clone --depth 1 --branch '%s' https://github.com/ggml-org/llama.cpp.git '%s'",
            resolved_tag, build_tmp);
    else
        SNPRINTF(clone_cmd,
            "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git '%s'",
            build_tmp);

    int build_ok = (run_quiet_no_exit("clone llama.cpp", clone_cmd) == 0);

    char cmake_args[4096];
    snprintf(cmake_args, sizeof(cmake_args), "%s",
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON");

    if (has_cmd("ccache")) {
        strncat(cmake_args,
            " -DCMAKE_C_COMPILER_LAUNCHER=ccache"
            " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
            " -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            sizeof(cmake_args) - strlen(cmake_args) - 1);
    }

    const char *build_desc = "building (CPU)";
    char build_desc_buf[256];

    if (build_ok) {
        /* CUDA detection */
        int has_cuda = has_cmd("nvcc") ||
                       is_exec("/usr/local/cuda/bin/nvcc") ||
                       (system("ls /usr/local/cuda-*/bin/nvcc >/dev/null 2>&1") == 0);
        /* ROCm detection */
        int has_rocm = !has_cuda && (has_cmd("hipcc") ||
                       is_exec("/opt/rocm/bin/hipcc") ||
                       (system("ls /opt/rocm-*/bin/hipcc >/dev/null 2>&1") == 0));

        if (has_cuda) {
            strncat(cmake_args, " -DGGML_CUDA=ON -DCMAKE_CUDA_FLAGS=--threads=0",
                    sizeof(cmake_args) - strlen(cmake_args) - 1);

            /* Detect GPU architectures */
            char archs[256]={0};
            FILE *fp2 = popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null", "r");
            if (fp2) {
                char cap[32];
                while (fgets(cap, sizeof(cap), fp2)) {
                    int mj=0, mn=0;
                    if (sscanf(cap, "%d.%d", &mj, &mn) == 2) {
                        char arch[16]; snprintf(arch, sizeof(arch), "%d%d", mj, mn);
                        if (!strstr(archs, arch)) {
                            if (archs[0]) strncat(archs, ";", sizeof(archs)-strlen(archs)-1);
                            strncat(archs, arch, sizeof(archs)-strlen(archs)-1);
                        }
                    }
                }
                pclose(fp2);
            }
            if (archs[0]) {
                char tmp2[256]; snprintf(tmp2, sizeof(tmp2), " -DCMAKE_CUDA_ARCHITECTURES=%s", archs);
                strncat(cmake_args, tmp2, sizeof(cmake_args)-strlen(cmake_args)-1);
                snprintf(build_desc_buf, sizeof(build_desc_buf), "building (CUDA, sm_%s)", archs);
                build_desc = build_desc_buf;
            } else {
                build_desc = "building (CUDA)";
            }
        } else if (has_rocm) {
            strncat(cmake_args, " -DGGML_HIP=ON", sizeof(cmake_args)-strlen(cmake_args)-1);
            build_desc = "building (ROCm)";
        }

        substep(build_desc);

        /* Number of parallel jobs */
        int ncpu = 4;
#ifndef _WIN32
        char ncpu_s[16]={0};
        cmd_output("nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4", ncpu_s, sizeof(ncpu_s));
        ncpu = atoi(ncpu_s); if (ncpu <= 0) ncpu = 4;
#else
        { SYSTEM_INFO si; GetSystemInfo(&si); ncpu = (int)si.dwNumberOfProcessors; }
#endif

        char gen_flag[64] = "";
        if (has_cmd("ninja")) snprintf(gen_flag, sizeof(gen_flag), "-G Ninja");

        char build_dir[PATHSZ]; SNPRINTF(build_dir, "%s/build", build_tmp);

        char cmake_cfg[CMDSZ];
        SNPRINTF(cmake_cfg, "cmake %s -S '%s' -B '%s' %s",
                 gen_flag, build_tmp, build_dir, cmake_args);
        if (run_quiet_no_exit("cmake llama.cpp", cmake_cfg) != 0) build_ok = 0;

        if (build_ok) {
            char cmd2[CMDSZ];
            SNPRINTF(cmd2,
                "cmake --build '%s' --config Release --target llama-server -j%d",
                build_dir, ncpu);
            if (run_quiet_no_exit("build llama-server", cmd2) != 0) build_ok = 0;
        }
        if (build_ok) {
            char cmd2[CMDSZ];
            SNPRINTF(cmd2,
                "cmake --build '%s' --config Release --target llama-quantize -j%d",
                build_dir, ncpu);
            SYS(run_quiet_no_exit("build llama-quantize", cmd2));
        }
    }

    if (build_ok) {
        rm_rf(llama_dir);
        char mv_cmd[CMDSZ];
        SNPRINTF(mv_cmd, "mv '%s' '%s'", build_tmp, llama_dir);
        SYS(system(mv_cmd));

        char quantize_bin[PATHSZ], symlink_dst[PATHSZ];
        SNPRINTF(quantize_bin, "%s/build/bin/llama-quantize", llama_dir);
        SNPRINTF(symlink_dst,  "%s/llama-quantize", llama_dir);
        if (path_exists(quantize_bin)) {
#ifndef _WIN32
            SYS(symlink("build/bin/llama-quantize", symlink_dst));
#else
            char cp_cmd[CMDSZ];
            SNPRINTF(cp_cmd, "copy \"%s\" \"%s\" >NUL 2>&1", quantize_bin, symlink_dst);
            SYS(system(cp_cmd));
#endif
        }

        char server_bin[PATHSZ]; SNPRINTF(server_bin, "%s/build/bin/llama-server", llama_dir);
        if (path_exists(server_bin)) {
            step("llama.cpp",    "built", NULL);
            if (path_exists(symlink_dst)) step("llama-quantize", "built", NULL);
        } else {
            step("llama.cpp", "binary not found after build", col(C_WARN));
        }
    } else {
        rm_rf(build_tmp);
        step("llama.cpp", "build failed", col(C_ERR));
    }
}

static void setup_llama(const char *script_dir)
{
    const char *home = getenv("HOME");
    if (!home) home = ".";

    char unsloth_home[PATHSZ], llama_dir[PATHSZ];
    SNPRINTF(unsloth_home, "%s/.unsloth", home);
    SNPRINTF(llama_dir,    "%s/llama.cpp", unsloth_home);
    mkdirp(unsloth_home);

    const char *requested_tag = getenv("UNSLOTH_LLAMA_TAG");
    if (!requested_tag) requested_tag = "latest";
    const char *helper_repo = getenv("UNSLOTH_LLAMA_RELEASE_REPO");
    if (!helper_repo) helper_repo = "unslothai/llama.cpp";
    const char *force_compile = getenv("UNSLOTH_LLAMA_FORCE_COMPILE");
    if (!force_compile) force_compile = "0";

    char resolved_tag[256] = {0};
    int need_source_build = 0;
    resolve_llama_tag(script_dir, requested_tag, helper_repo,
                      resolved_tag, sizeof(resolved_tag), &need_source_build);

    char msg[512]; snprintf(msg, sizeof(msg), "resolved llama.cpp tag: %s", resolved_tag);
    substep(msg);

    if (strcmp(force_compile, "1") == 0) {
        step("llama.cpp", "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt", col(C_WARN));
        need_source_build = 1;
    } else if (!need_source_build) {
        substep("installing prebuilt llama.cpp...");
        if (is_dir(llama_dir)) substep("existing install detected -- validating update");

        char prebuilt_cmd[CMDSZ];
        SNPRINTF(prebuilt_cmd,
            "python '%s/install_llama_prebuilt.py'"
            " --install-dir '%s' --llama-tag '%s' --published-repo '%s'",
            script_dir, llama_dir, resolved_tag, helper_repo);

        const char *rel_tag = getenv("UNSLOTH_LLAMA_RELEASE_TAG");
        if (rel_tag) {
            strncat(prebuilt_cmd, " --published-release-tag '", sizeof(prebuilt_cmd)-strlen(prebuilt_cmd)-1);
            strncat(prebuilt_cmd, rel_tag,                      sizeof(prebuilt_cmd)-strlen(prebuilt_cmd)-1);
            strncat(prebuilt_cmd, "'",                          sizeof(prebuilt_cmd)-strlen(prebuilt_cmd)-1);
        }

        if (system(prebuilt_cmd) == 0) {
            step("llama.cpp", "prebuilt installed and validated", NULL);
        } else {
            if (is_dir(llama_dir)) substep("prebuilt update failed; existing install restored");
            substep("falling back to source build");
            need_source_build = 1;
        }
    }

    if (need_source_build) {
#ifndef _WIN32
        install_wsl_build_deps();
#endif
        build_llama_source(llama_dir, resolved_tag);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  main
 * ══════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    /* Collect any extra args to forward to the studio launcher (e.g. -H 0.0.0.0 -p 8888) */
    /* argv[0] = this binary; argv[1..] = forwarded args                                   */

    /* Color detection */
    if (!getenv("NO_COLOR") && (isatty(fileno(stdout)) || getenv("FORCE_COLOR")))
        g_color = 1;

    /* Resolve script dir from argv[0] */
    char script_dir[PATHSZ];
#ifndef _WIN32
    /* Use /proc/self/exe for a reliable absolute path */
    ssize_t len = readlink("/proc/self/exe", script_dir, sizeof(script_dir) - 1);
    if (len > 0) {
        script_dir[len] = '\0';
    } else {
        /* Fallback: resolve argv[0] */
        if (!realpath(argv[0], script_dir)) {
            snprintf(script_dir, sizeof(script_dir), "%s", argv[0]);
        }
    }
    dirname_inplace(script_dir);   /* strip binary name → get its directory */
#else
    GetModuleFileNameA(NULL, script_dir, (DWORD)sizeof(script_dir));
    dirname_inplace(script_dir);
#endif

    /* repo_root = one level up from script_dir */
    char repo_root[PATHSZ];
    snprintf(repo_root, sizeof(repo_root), "%s", script_dir);
    dirname_inplace(repo_root);

    /* ── Banner ── */
    printf("\n");
    printf("  %s%s%s\n", col(C_TITLE), "🦥 Unsloth Studio Setup", col(C_RST));
    printf("  %s%s%s\n", col(C_DIM),   RULE,                       col(C_RST));

    /* ── Clean stale caches ── */
    {
        char p[PATHSZ];
        SNPRINTF(p, "%s/unsloth_compiled_cache",         repo_root);   rm_rf(p);
        SNPRINTF(p, "%s/backend/unsloth_compiled_cache", script_dir);  rm_rf(p);
        SNPRINTF(p, "%s/tmp/unsloth_compiled_cache",     script_dir);  rm_rf(p);
    }

    int is_colab = detect_colab();

    build_frontend(script_dir);
    setup_oxc_validator(script_dir);
    setup_python(script_dir, is_colab);
    setup_llama(script_dir);

    /* ── Footer + auto-launch ── */
    printf("  %s%s%s\n", col(C_DIM), RULE, col(C_RST));
    if (is_colab) {
        printf("  %s%s%s\n", col(C_TITLE), "Unsloth Studio Setup Complete", col(C_RST));
        printf("  %s%s%s\n", col(C_DIM),   RULE,                            col(C_RST));
        substep("from colab import start");
        substep("start()");
    } else {
        printf("  %s%s%s\n", col(C_TITLE), "Unsloth Studio", col(C_RST));
        printf("  %s%s%s\n", col(C_DIM),   RULE,             col(C_RST));

        /* ── Parse -H / --host and -p / --port from forwarded args ── */
        const char *host = "127.0.0.1";
        const char *port = "7860";
        for (int i = 1; i < argc - 1; i++) {
            if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--host") == 0)
                host = argv[i + 1];
            if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0)
                port = argv[i + 1];
        }
        /* For the browser URL always use localhost even if server binds 0.0.0.0 */
        const char *browser_host =
            (strcmp(host, "0.0.0.0") == 0 || strcmp(host, "::") == 0)
            ? "127.0.0.1" : host;

        char studio_url[256];
        snprintf(studio_url, sizeof(studio_url), "http://%s:%s", browser_host, port);

        /* ── Build the launch command: "unsloth studio [forwarded args]" ── */
        char launch_cmd[CMDSZ];
        int  pos = snprintf(launch_cmd, sizeof(launch_cmd), "unsloth studio");
        for (int i = 1; i < argc && pos < (int)sizeof(launch_cmd) - 2; i++)
            pos += snprintf(launch_cmd + pos, sizeof(launch_cmd) - (size_t)pos,
                            " %s", argv[i]);
        if (argc < 2)
            pos += snprintf(launch_cmd + pos, sizeof(launch_cmd) - (size_t)pos,
                            " -H 0.0.0.0 -p 7860");

        printf("  %slaunching%s   %s%s%s\n",
               col(C_DIM), col(C_RST), col(C_OK), launch_cmd, col(C_RST));
        printf("  %sbrowser%s     %s%s%s\n",
               col(C_DIM), col(C_RST), col(C_OK), studio_url, col(C_RST));
        printf("\n");

        /* ── Fork a background process that polls /api/health then opens browser ── */
#ifndef _WIN32
        pid_t browser_pid = fork();
        if (browser_pid == 0) {
            /* Child: poll /api/health until 200 (or 60 s timeout), then open browser */
            char health_url[512];
            snprintf(health_url, sizeof(health_url), "%s/api/health", studio_url);

            char poll_cmd[CMDSZ];
            snprintf(poll_cmd, sizeof(poll_cmd),
                     "curl -fsSL --max-time 2 --silent --output " DEVNULL
                     " --write-out '%%{http_code}' '%s' 2>/dev/null",
                     health_url);

            int ready = 0;
            for (int attempt = 0; attempt < 60 && !ready; attempt++) {
                sleep(1);
                FILE *fp = popen(poll_cmd, "r");
                if (fp) {
                    char code[8] = {0};
                    if (fgets(code, sizeof(code), fp))
                        ready = (strncmp(code, "200", 3) == 0);
                    pclose(fp);
                }
            }

            char open_cmd[CMDSZ];
            int  is_wsl = 0;

            /* Detect WSL */
            FILE *pv = fopen("/proc/version", "r");
            if (pv) {
                char pvline[512]; char pvlow[512]; int k;
                if (fgets(pvline, sizeof(pvline), pv)) {
                    for (k = 0; pvline[k] && k < (int)sizeof(pvlow)-1; k++)
                        pvlow[k] = (pvline[k] >= 'A' && pvline[k] <= 'Z')
                                   ? pvline[k] + 32 : pvline[k];
                    pvlow[k] = '\0';
                    if (strstr(pvlow, "microsoft")) is_wsl = 1;
                }
                fclose(pv);
            }

#  if defined(__APPLE__)
            snprintf(open_cmd, sizeof(open_cmd), "open '%s'", studio_url);
#  else
            if (is_wsl)
                snprintf(open_cmd, sizeof(open_cmd),
                         "cmd.exe /c start '' '%s' >" DEVNULL " 2>&1", studio_url);
            else
                snprintf(open_cmd, sizeof(open_cmd),
                         "xdg-open '%s' >" DEVNULL " 2>&1"
                         " || sensible-browser '%s' >" DEVNULL " 2>&1"
                         " || x-www-browser '%s' >" DEVNULL " 2>&1",
                         studio_url, studio_url, studio_url);
#  endif
            int _unused_rc = system(open_cmd); (void)_unused_rc;
            _exit(0);
        }
        /* Parent continues to start the server (blocks until server exits) */
#else
        /* Windows: poll health endpoint then open browser via ShellExecute */
        {
            char health_url[512];
            snprintf(health_url, sizeof(health_url), "%s/api/health", studio_url);
            char poll_cmd[CMDSZ];
            snprintf(poll_cmd, sizeof(poll_cmd),
                     "curl -fsSL --max-time 2 --silent --output NUL"
                     " --write-out \"%%{http_code}\" \"%s\" 2>NUL",
                     health_url);
            int ready = 0;
            for (int attempt = 0; attempt < 60 && !ready; attempt++) {
                Sleep(1000);
                FILE *fp = _popen(poll_cmd, "r");
                if (fp) {
                    char code[8] = {0};
                    if (fgets(code, sizeof(code), fp))
                        ready = (strncmp(code, "200", 3) == 0);
                    _pclose(fp);
                }
            }
            ShellExecuteA(NULL, "open", studio_url, NULL, NULL, SW_SHOWNORMAL);
        }
#endif

        int rc = system(launch_cmd);
        if (rc != 0) {
            /* 'unsloth' not on PATH — try via the venv python directly */
            char fallback[CMDSZ];
            const char *venv = getenv("VIRTUAL_ENV");
            if (venv) {
                char py[PATHSZ];
                snprintf(py, sizeof(py), "%s/bin/python", venv);
                int fpos = snprintf(fallback, sizeof(fallback),
                                    "'%s' -m unsloth.studio", py);
                for (int i = 1; i < argc && fpos < (int)sizeof(fallback) - 2; i++)
                    fpos += snprintf(fallback + fpos, sizeof(fallback) - (size_t)fpos,
                                     " %s", argv[i]);
                if (argc < 2)
                    fpos += snprintf(fallback + fpos, sizeof(fallback) - (size_t)fpos,
                                     " -H 0.0.0.0 -p 7860");
                rc = system(fallback);
            }
        }
        if (rc != 0) {
            step("launch", "studio could not be started -- run: unsloth studio", col(C_ERR));
            return 1;
        }
    }
    printf("\n");
    return 0;
}