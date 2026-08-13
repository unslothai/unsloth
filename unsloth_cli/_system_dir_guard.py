# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows system-folder guard for the `unsloth` console script.

C:\\Windows\\System32 is unwritable for a normal user, and cwd-relative paths
(`./models`, `unsloth_compiled_cache`) would resolve inside the Windows tree.

Two ways in. "Run as administrator" opens a terminal there, a mistake that still
stops with an actionable error. And "Run Unsloth at login" starts the desktop
from an HKCU Run value, which carries no working directory, so it and every CLI
child inherit System32 (issue #8510). That one is not the user's mistake: the
desktop's own commands take no paths from the user, so they move to ~/.unsloth
rather than leaving a tray icon and no server.

Imports stay at `os`: this runs before the command modules, which resolve
STUDIO_HOME against the working directory.
"""

import os as _os

# Set by Unsloth Desktop on every CLI child it owns (process.rs). Forging it
# grants nothing: anyone who can set a child's environment can set its working
# directory, and the move lands inside the caller's own account.
DESKTOP_MANAGED_ENV = "UNSLOTH_DESKTOP_MANAGED"

# The directory process.rs pins, so an older desktop lands in the same place.
WORK_DIR_NAME = ".unsloth"


def windows_root(
    environ,
    pathmod = _os.path,
    isdir = None,
):
    """Where Windows is installed, for messages."""
    return windows_roots(environ, pathmod, isdir)[0]


def windows_roots(
    environ,
    pathmod = _os.path,
    isdir = None,
):
    """Every real Windows directory.

    Candidates are checked, not trusted: a WINDIR aimed at the user's profile
    would make ordinary folders look like system ones, and one aimed elsewhere
    would disarm the guard. So a directory counts only if it holds System32.
    """
    if isdir is None:
        isdir = pathmod.isdir
    system_root = environ.get("SystemRoot")
    candidates = [system_root, environ.get("WINDIR"), r"C:\Windows"]

    roots = []
    for value in candidates:
        if value and value not in roots and isdir(pathmod.join(value, "System32")):
            roots.append(value)
    if roots:
        return roots
    # No Windows installation found: keep the guard alive on SystemRoot or the
    # default, never on a user-settable value.
    return [system_root or r"C:\Windows"]


def _strip_extended_prefix(path):
    r"""Drop the \\?\ (and \\?\UNC\) form so it compares like an ordinary path.

    Matched case-insensitively: the object manager accepts \\?\unc\server\share,
    and reading that as relative would reject a profile Windows itself resolves.
    """
    lowered = path.lower()
    if lowered.startswith("\\\\?\\unc\\"):
        return "\\\\" + path[8:]
    if lowered.startswith("\\\\?\\"):
        return path[4:]
    return path


def _normalize(path, pathmod):
    return pathmod.normcase(pathmod.normpath(_strip_extended_prefix(path)))


def system_dirs(windir, pathmod = _os.path):
    """The Windows folders Unsloth refuses to run from."""
    # SysWOW64 too: a 32-bit elevated shell opens there, same unwritable folder.
    return [_normalize(pathmod.join(windir, name), pathmod) for name in ("System32", "SysWOW64")]


def is_system_dir(
    cwd,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
):
    """True for a system folder itself or anything under it.

    `windir` may be a single directory or several candidates. The separator keeps
    the match on a path boundary, so C:\\Windows2\\System32x is an ordinary folder.
    """
    if not cwd:
        return False
    roots = [windir] if isinstance(windir, str) else list(windir)
    normalized = _normalize(cwd, pathmod)
    return any(
        normalized == directory or normalized.startswith(directory + sep)
        for root in roots
        for directory in system_dirs(root, pathmod)
    )


def _is_rooted(path, pathmod):
    """Absolute, or at least rooted at a drive.

    "." and "C:sub" name no directory on their own, so they are no escape
    (pin_relative_overrides resolves the drive-relative form separately). A
    leading separator is not absolute, but can never resolve back into System32.
    """
    stripped = _strip_extended_prefix(path)
    return pathmod.isabs(stripped) or stripped.startswith(("\\", "/"))


def _is_fully_qualified(path, pathmod):
    r"""Whether the value names one directory whatever the process does next.

    Narrower than _is_rooted: "\cache" is rooted only to the drive of the current
    directory, so a profile on another drive silently moves it too. Spelled out
    rather than deferred to isabs(), which answered True for a leading separator
    until Python 3.13 and False after: the folder a value names cannot depend on
    the interpreter running the guard.
    """
    stripped = _strip_extended_prefix(path)
    if stripped.startswith(("\\\\", "//")):
        # A UNC share names its own root.
        return True
    drive, rest = pathmod.splitdrive(stripped)
    return bool(drive) and rest.startswith(("\\", "/"))


def _outside_windows(candidate, windirs, pathmod, sep):
    if not candidate or not _is_rooted(candidate, pathmod):
        return False
    norm = _normalize(candidate, pathmod)
    for windir in windirs:
        windir_norm = _normalize(windir, pathmod)
        # A root-relative candidate carries no drive, so it equals no
        # drive-qualified root: "\Windows\System32\config\systemprofile" is
        # SYSTEM's profile on whichever drive, so compare that spelling too.
        for form in (windir_norm, pathmod.splitdrive(windir_norm)[1]):
            if not form:
                continue
            if norm == form or norm.startswith(form + sep):
                return False
    return True


def safe_user_dir(
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    allow_public = False,
):
    """First home outside the Windows tree, or None.

    SYSTEM's USERPROFILE is C:\\Windows\\System32\\config\\systemprofile, so a naive
    pick lands back in the rejected folder. %PUBLIC% is only ever a suggestion a
    human can type: moving there would put one account's caches and outputs in a
    folder every other account can read and write.
    """
    if expanduser is None:
        expanduser = pathmod.expanduser
    windirs = [windir] if isinstance(windir, str) else list(windir)
    public = (environ.get("PUBLIC") or "").strip()
    candidates = [environ.get("USERPROFILE")]
    if allow_public:
        candidates.append(public)
    candidates.append(expanduser("~"))
    for candidate in candidates:
        if not _outside_windows(candidate, windirs, pathmod, sep):
            continue
        # USERPROFILE and ~ can name the public profile themselves, so the check
        # is on the folder, not on which variable it came from.
        if (
            not allow_public
            and public
            and _normalize(candidate, pathmod) == _normalize(public, pathmod)
        ):
            continue
        return candidate
    return None


# Commands the desktop runs that take no path from a user: Studio resolves its
# venv, llama.cpp, auth, pid files and logs from the Studio home. `update` is here
# so an older desktop can still upgrade from the tray. Everything else keeps the
# hard error. Matched whole, not on the first word, since `studio update --local
# <path>` resolves that path against the working directory.
_STUDIO_COMMANDS = (
    ("provision-desktop-auth",),
    ("desktop-capabilities",),
    ("desktop-capabilities", "--json"),
    ("update",),
)
_HELP_FLAGS = ("-h", "--help", "--version", "-V")
_API_ONLY_FLAGS = ("--api-only", "-H", "--host", "-p", "--port")


def _is_desktop_backend_launch(rest):
    """`studio --api-only -H 127.0.0.1 -p 8888` and nothing else: matching
    --api-only anywhere would also match `studio run --model ./m.gguf --api-only`,
    a user command with user paths.
    """
    if "--api-only" not in rest:
        return False
    expects_value = False
    for arg in rest:
        if expects_value:
            expects_value = False
            continue
        if arg not in _API_ONLY_FLAGS:
            return False
        expects_value = arg != "--api-only"
    return True


# Subcommands that take a path from the caller, by argument or by environment.
# `run` takes --model and a raw llama-server tail; `update --local` installs
# from a checkout the caller names.
_PATH_TAKING_STUDIO_COMMANDS = ("run", "update")


def _takes_a_path(rest):
    """Whether this `studio` invocation can carry a caller's path. Blunt on
    purpose: the bare forms the desktop runs carry none, so anything else might.
    """
    if not rest:
        return False
    if rest[0] in _PATH_TAKING_STUDIO_COMMANDS:
        return tuple(rest) not in _STUDIO_COMMANDS
    # rest[0] is the subcommand name, skipped unless it is a flag (then there is
    # no subcommand). An attached value hides inside its own token, so a leading
    # dash does not clear it: `studio --frontend=.\dist` carries a path.
    tail = rest if rest[0].startswith("-") else rest[1:]
    return any(not arg.startswith("-") or "=" in arg for arg in tail)


def is_relocatable_invocation(argv, environ):
    """True when this invocation is desktop-managed or provably cwd-independent.

    The argv arm matters on its own: it fixes users whose desktop build predates
    the Rust-side fix and so sets no marker.
    """
    args = [arg for arg in argv if arg]
    if not args:
        return False
    # Click handles top-level -h/--help/--version eagerly, before the callback
    # this runs from, so only `studio --help` actually arrives here.
    if all(arg in _HELP_FLAGS for arg in args):
        return True
    if args[0] != "studio":
        # The marker is inherited by everything the backend spawns, so it
        # authorises the desktop's studio commands and nothing else: rebasing
        # `train --dataset .\data.json` under a stray one would be worse than the
        # refusal it replaced.
        return False
    rest = args[1:]
    if environ.get(DESKTOP_MANAGED_ENV) == "1" and not _takes_a_path(rest):
        # The marker covers a desktop build whose command shape this CLI does not
        # know yet, but must not widen the set to path-carrying commands: `studio
        # run --model .\local.gguf` from a marked shell would be rebased.
        return True
    if rest and all(arg in _HELP_FLAGS for arg in rest):
        return True
    if _is_desktop_backend_launch(rest):
        return True
    return tuple(rest) in _STUDIO_COMMANDS


# Path overrides the caller may have written relative to the folder being left.
# Studio resolves them with Path.resolve(), which anchors a relative value to the
# working directory, so moving first would silently retarget them.
_RELATIVE_PATH_ENV = (
    # Studio roots: storage_roots.py.
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_STUDIO_DOCUMENTS_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "UNSLOTH_STUDIO_SANDBOX_HOME",
    # `studio update` reads it, and that command relocates.
    "STUDIO_LOCAL_REPO",
    # Engine and tool locations the user may point somewhere of their own.
    "UNSLOTH_LLAMA_CPP_PATH",
    "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR",
    "UNSLOTH_SD_CPP_PATH",
    "UNSLOTH_WHISPER_CPP_PATH",
    "LLAMA_SERVER_PATH",
    "WHISPER_SERVER_PATH",
    "SD_CLI_PATH",
    "SD_SERVER_PATH",
    # Model files llama-server reads straight from the environment, and Studio
    # reads back when it sizes a launch (llama_cpp.py). The URL and HF-repo
    # spellings are deliberately absent: they name no local file.
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_MMPROJ",
    "LLAMA_ARG_MODEL_DRAFT",
    "LLAMA_ARG_SPEC_DRAFT_MODEL",
    # GPU SDK roots, joined with bin/ for DLL discovery.
    "CUDA_PATH",
    "HIP_PATH",
    "HIP_PATH_57",
    "ROCM_PATH",
    "MLX_HOSTFILE",
    # Read exactly like MLX_HOSTFILE: either inline JSON or a filename.
    "MLX_IBV_DEVICES",
    "OLLAMA_MODELS",
    "DG_VISUAL_BIN",
    "UNSLOTH_DG_SHIM",
    # Caches.
    "UNSLOTH_COMPILE_LOCATION",
    "TORCHINDUCTOR_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COND_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    # The credential file: a relative value would follow the child and lose
    # access to gated repos.
    "HF_TOKEN_PATH",
    # Read as written, and authoritative when non-blank (storage_roots.py), so
    # `unsloth studio update` would install from a different cache after a move.
    "UV_CACHE_DIR",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "UNSLOTH_STUDIO_CHILD_RECORD",
    "UNSLOTH_LLAMA_INSTALLER",
    "CUDA_HOME",
    "CUDA_ROOT",
)

# The separator is Windows', not the host's: this guard only ever runs on
# Windows, and os.pathsep would split "D:\\shared" apart anywhere else.
_PATH_LIST_SEPARATOR = ";"

# Values holding several separated directories, anchored entry by entry. A
# relative PYTHONPATH entry is resolved at import time, so a move would let
# whatever sits in the new directory shadow a managed import. PATH is left out on
# purpose: refusing the whole move over one unresolvable entry in a list that long
# would cost more than it protects.
_PATH_LIST_ENV = (
    "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",
    "CUDA_RUNTIME_DLL_DIR",
    "PYTHONPATH",
)


def pin_relative_overrides(
    environ,
    cwd,
    pathmod = _os.path,
    abspath = None,
    expandvars = None,
    expanduser = None,
):
    """Rewrite relative path overrides so they keep naming the same folder.

    Returns the names pinned. A `~` value is written out, since only some readers
    expand it themselves.
    """
    pinned = []
    for name in _RELATIVE_PATH_ENV:
        value = (environ.get(name) or "").strip()
        anchored = _anchor(name, value, cwd, pathmod, abspath, expandvars, expanduser)
        if anchored is not None:
            environ[name] = anchored
            pinned.append(name)
    for name in _PATH_LIST_ENV:
        raw = environ.get(name) or ""
        if not raw.strip():
            continue
        # Each entry is anchored on its own: one relative entry changes what the
        # whole list means.
        entries = raw.split(_PATH_LIST_SEPARATOR)
        anchored_entries = [
            _anchor_list_entry(name, e, cwd, pathmod, abspath, expandvars, expanduser)
            for e in entries
        ]
        if anchored_entries != entries:
            environ[name] = _PATH_LIST_SEPARATOR.join(anchored_entries)
            pinned.append(name)
    return pinned


# Values a consumer deliberately does not read as a plain path: anchoring one
# changes what it means rather than moving a folder. Each exemption names the
# variables whose reader proves it, since the syntax is only special there and a
# directory really called "[llama]" or "%data%" is legal on Windows.

# MLX_HOSTFILE holds either a filename or the host list itself, as JSON
# (`unsloth_cli/_inference.py`, `_json_rank_count_from_env`).
_INLINE_JSON_ENV = frozenset(("MLX_HOSTFILE", "MLX_IBV_DEVICES"))

# Names whose readers disagree about %VAR% and $VAR: huggingface_hub calls
# expandvars on HF_HOME (and on the XDG_CACHE_HOME it defaults from), HF_HUB_CACHE
# and HF_ASSETS_CACHE, and Studio calls it on SENTENCE_TRANSFORMERS_HOME, but
# Studio's own hf_cache_settings._canonical() does not, so it would read
# %LOCALAPPDATA%\hf as a relative folder. Expanding here before deciding settles
# it: both readers then see one absolute path. Scoped to these names because a
# directory really called "%data%" is legal, and every other name is read as one.
_EXPANDED_ENV = frozenset(
    (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_TOKEN_PATH",
        "XDG_CACHE_HOME",
        "SENTENCE_TRANSFORMERS_HOME",
    )
)

# The pre-quant allowlist skips a bare on/off token precisely so there is no
# "allow all" mode (`diffusion_prequant.py`); anchoring one would turn it into a
# real allowlisted directory.
_TOGGLE_ENV = frozenset(("UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",))
_TOGGLE_TOKENS = frozenset(("1", "true", "yes", "on", "0", "false", "no", "off"))


def _names_a_path(name, value):
    """Whether the working directory is what resolves this variable's value."""
    if name in _INLINE_JSON_ENV and value.startswith(("[", "{")):
        return False
    if name in _TOGGLE_ENV and value.lower() in _TOGGLE_TOKENS:
        return False
    return True


def pin_relative_sys_path(
    cwd,
    pathmod = _os.path,
    syspath = None,
    abspath = None,
    exists = None,
    expanduser = None,
):
    """Anchor the relative import roots this interpreter already carries.

    sys.path holds PYTHONPATH entries as written, including the two spellings that
    follow the process rather than the caller: an empty entry means the working
    directory, and a leading `~` is never expanded there.

    Only an entry naming something on disk is touched, folder or archive. The rest
    are other people's strings rather than paths, such as the relative sentinel
    setuptools registers for an editable install and accepts back by exact
    equality; rewriting one breaks the import it was meant to protect, and leaving
    a real archive breaks the next import from it.
    """
    if syspath is None:
        import sys as _sys
        syspath = _sys.path
    if exists is None:
        exists = _os.path.exists
    pinned = []
    for index, entry in enumerate(syspath):
        if not isinstance(entry, str):
            continue
        try:
            # The empty entry is the working directory by definition; anything
            # else has to name something that is really there.
            if entry.strip() and not exists(entry):
                continue
            anchored = _anchor_list_entry(
                "PYTHONPATH", entry, cwd, pathmod, abspath, None, expanduser
            )
        except Exception:
            # Best effort, unlike the environment: an import root this process
            # already holds is not worth refusing the move over.
            continue
        if anchored != entry:
            syspath[index] = anchored
            pinned.append(anchored)
    return pinned


def _anchor_list_entry(
    name,
    entry,
    cwd,
    pathmod,
    abspath,
    expandvars,
    expanduser = None,
):
    """One entry of a path list, anchored, or left as written. An empty PYTHONPATH
    component means the working directory itself, the one spelling _anchor cannot
    see.
    """
    entry = entry.strip()
    if name == "PYTHONPATH" and not entry:
        return cwd
    return _anchor(name, entry, cwd, pathmod, abspath, expandvars, expanduser) or entry


def _expand_settled(
    value,
    expandvars,
    passes = 8,
):
    """The value expanded until it stops changing, or None if it never does.

    One pass is what each reader does, and one pass is enough for the values
    Windows itself builds. A nested reference (LOCALAPPDATA holding %USERPROFILE%)
    needs more, and a self-reference needs infinitely many.
    """
    for _ in range(passes):
        expanded = expandvars(value)
        if expanded == value:
            return value
        value = expanded
    return None


def _anchor(
    name,
    value,
    cwd,
    pathmod,
    abspath = None,
    expandvars = None,
    expanduser = None,
):
    """The value rewritten to name the same folder from anywhere, or None.

    None means no rewriting is needed: empty, or already fully qualified.
    """
    original = value = (value or "").strip()
    if value.startswith("~"):
        # Written out rather than skipped: only some readers call expanduser
        # first, and llama_cpp.py hands UNSLOTH_LLAMA_CPP_PATH straight to Path(),
        # so a move would leave it naming a folder called "~" under the new
        # directory. It is what the caller meant either way.
        value = (expanduser or pathmod.expanduser)(value)
    if name in _EXPANDED_ENV and value:
        # Written out, so the reader that expands and the one that does not land
        # in the same folder. An unset variable is left exactly as written.
        value = _expand_settled(value, expandvars or _os.path.expandvars)
        if value is None:
            # A value whose expansion never settles (%HF_HOME% inside HF_HOME) is
            # left exactly as it was: no reader can resolve it either, and
            # anchoring a half-expanded string would build a folder name with a
            # second drive in the middle of it.
            return None
    if not value:
        return None
    if _is_fully_qualified(value, pathmod):
        # Already names one folder, but still worth writing back if expanding is
        # what made it name one: the reader that does not expand cannot see that.
        return value if value != original else None
    if not _names_a_path(name, value):
        return None
    if pathmod.splitdrive(value)[0] or value.startswith(("\\", "/")):
        # "D:cache" is drive D's own current directory and "\cache" the root of
        # the current drive, neither of which join() knows, so ask the OS. A
        # failure reaches the caller, which then declines to move at all.
        return (abspath or pathmod.abspath)(value)
    return pathmod.join(cwd, value)


def relocation_target(
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    makedirs = _os.makedirs,
    home_isdir = None,
):
    """Where a desktop-managed command should run instead, or None."""
    home = safe_user_dir(environ, windir, pathmod, sep, expanduser)
    if not home:
        return None
    if home_isdir is None:
        home_isdir = pathmod.isdir
    # A profile that has not mounted yet still has a writable parent, so makedirs
    # would build an empty second one that shadows the real one when it arrives.
    if not home_isdir(home):
        return None
    work_dir = pathmod.join(home, WORK_DIR_NAME)
    try:
        makedirs(work_dir, exist_ok = True)
    except OSError:
        # An unwritable home is a broken profile and Studio must write there
        # anyway, so stop now rather than failing later.
        return None
    return work_dir


def blocked_message(
    cwd,
    argv,
    environ,
    windir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
):
    """The error shown to someone who ran Unsloth from a system folder by hand."""
    # allow_public here only: a person can sensibly `cd C:\Users\Public`, but
    # relocating there would share one account's state with every other account.
    home = safe_user_dir(environ, windir, pathmod, sep, expanduser, allow_public = True)
    if home:
        # Quote it, or C:\Users\Jane Doe reaches Set-Location as two arguments.
        # PowerShell single quotes are verbatim ('' escapes an apostrophe); cmd
        # needs double quotes once extensions are off.
        home_ps = "'" + home.replace("'", "''") + "'"
        home_cmd = '"' + home + '"'
        cd_lines = (
            f"    cd {home_ps}          (PowerShell)\n" f"    cd /d {home_cmd}       (cmd.exe)\n"
        )
    else:
        cd_lines = f"    (any folder outside {windir if isinstance(windir, str) else windir[0]})\n"
    rendered_argv = " ".join((f'"{arg}"' if " " in arg else arg) for arg in argv)
    retry = ("unsloth " + rendered_argv).rstrip()
    return (
        f"Unsloth cannot run from {cwd}\n"
        "\n"
        "That is a Windows system folder. Windows blocks writes here, and any\n"
        "relative path you pass would resolve inside the Windows folder.\n"
        "Opening a terminal with 'Run as administrator' starts you in a folder like\n"
        "this one, which is how most people end up here.\n"
        "\n"
        "Change to a normal folder and run the command again:\n"
        f"{cd_lines}"
        f"    {retry}"
    )


def check_working_directory(
    argv,
    environ,
    platform,
    getcwd = _os.getcwd,
    chdir = _os.chdir,
    pathmod = _os.path,
    sep = _os.sep,
    expanduser = None,
    makedirs = _os.makedirs,
    isdir = None,
    abspath = None,
    home_isdir = None,
    exists = None,
    syspath = None,
    expandvars = None,
    relocate = True,
):
    """Decide what to do about the current working directory.

    Returns (message, colour, fatal). `fatal` is the caller's cue to exit 1;
    a message with fatal False is a warning printed after a successful move.
    """
    if platform != "win32":
        return None, None, False

    windirs = windows_roots(environ, pathmod, isdir)
    windir = windirs[0]
    try:
        cwd = getcwd()
    except OSError:
        # The launch directory is gone: say so rather than name a folder they
        # were never in.
        return (
            (
                "Unsloth cannot determine its current folder. It may have been deleted,\n"
                "or it may be on a drive that is no longer available.\n"
                "Change to a folder that exists and run the command again."
            ),
            "red",
            True,
        )

    if not is_system_dir(cwd, windirs, pathmod, sep):
        return None, None, False

    if not relocate or not is_relocatable_invocation(argv, environ):
        # `relocate = False` is the imported-as-a-library case: the command
        # modules are already loaded, so a move now is too late for the roots they
        # resolved at import time.
        return blocked_message(cwd, argv, environ, windirs, pathmod, sep, expanduser), "red", True

    target = relocation_target(environ, windirs, pathmod, sep, expanduser, makedirs, home_isdir)
    unpinnable = None
    # The caller's own process state when the guard runs inside a host, so nothing
    # stays rewritten unless the move actually happens.
    environ_before = dict(environ)
    if syspath is None:
        # Resolved here rather than inside the pinning, or the console script,
        # which passes nothing, would rewrite the real sys.path with no snapshot
        # to put back when the move then fails.
        import sys as _sys
        syspath = _sys.path
    syspath_before = list(syspath)
    if target is not None:
        try:
            # Before moving, or a relative override would end up naming a folder
            # under the new directory instead of theirs.
            pin_relative_overrides(environ, cwd, pathmod, abspath, expandvars, expanduser)
            # This interpreter read PYTHONPATH before the guard ran and keeps the
            # entries as written, resolving a relative one on every import, so it
            # needs anchoring here as well as in the environment.
            pin_relative_sys_path(cwd, pathmod, syspath, abspath, exists, expanduser)
        except Exception as error:
            # An environment we cannot pin is one we must not move underneath.
            # Both reasons are real: a drive with no current directory, and a list
            # that no longer fits in a Windows variable once fully qualified.
            unpinnable = error
            target = None
    if target is not None:
        try:
            chdir(target)
        except OSError:
            target = None
        else:
            # Confirm where it landed rather than trusting chdir not to raise.
            try:
                if is_system_dir(getcwd(), windirs, pathmod, sep):
                    target = None
            except OSError:
                target = None
    if target is None:
        # Nothing moved, so nothing stays rewritten: put back what pinning wrote.
        if environ_before != environ:
            environ.clear()
            environ.update(environ_before)
        if syspath_before != syspath:
            syspath[:] = syspath_before
    if unpinnable is not None:
        # Named separately from the profile case below: blaming the user folder
        # for an unpinnable override sends them looking in the wrong place.
        return (
            (
                f"Unsloth cannot run from {cwd}, and could not move out of it\n"
                "without changing where one of its path settings points\n"
                f"({type(unpinnable).__name__}: {unpinnable}).\n"
                "Set that value to a full path, or start Unsloth from a normal folder."
            ),
            "red",
            True,
        )
    if target is None:
        # Fail closed: nowhere usable outside the Windows tree. This text lands in
        # the desktop's logs, so it describes that case, not a shell.
        return (
            (
                f"Unsloth cannot run from {cwd}, and no folder outside {windir} was\n"
                "available to run from instead. Check that the user profile for this\n"
                "account exists and is writable."
            ),
            "red",
            True,
        )

    return (
        (
            f"Unsloth was started from {cwd}, which is a Windows system folder,\n"
            f"so it switched to {target} instead.\n"
            "This happens when Unsloth Desktop is started by 'Run Unsloth at login'."
        ),
        "yellow",
        False,
    )
