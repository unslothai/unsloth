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

# Set by Unsloth Desktop on every CLI child (process.rs); forging it grants nothing.
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
    # Never fall back on a user-settable value.
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
    # SysWOW64 too: a 32-bit elevated shell opens there.
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
        return True
    drive, rest = pathmod.splitdrive(stripped)
    return bool(drive) and rest.startswith(("\\", "/"))


def _outside_windows(candidate, windirs, pathmod, sep):
    if not candidate or not _is_rooted(candidate, pathmod):
        return False
    norm = _normalize(candidate, pathmod)
    for windir in windirs:
        windir_norm = _normalize(windir, pathmod)
        # A root-relative candidate carries no drive, so compare that spelling too.
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
        # USERPROFILE and ~ can name the public profile, so check the folder itself.
        if (
            not allow_public
            and public
            and _normalize(candidate, pathmod) == _normalize(public, pathmod)
        ):
            continue
        return candidate
    return None


# Desktop commands that take no user path (`update` lets an older desktop upgrade). Matched
# whole, since `studio update --local <path>` resolves against the working directory.
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


_PATH_TAKING_STUDIO_COMMANDS = ("run", "update")


def _carries_a_value(arg):
    """Whether this token can hold a value, attached or not."""
    if not arg.startswith("-"):
        return True
    if arg.startswith("--"):
        return "=" in arg
    # `-f.\dist` is Click's spelling of `--frontend .\dist`.
    return len(arg) > 2


def _takes_a_path(rest):
    """Whether this `studio` invocation can carry a caller's path. Blunt on
    purpose: the bare forms the desktop runs carry none, so anything else might.
    """
    if not rest:
        return False
    if rest[0] in _PATH_TAKING_STUDIO_COMMANDS:
        return tuple(rest) not in _STUDIO_COMMANDS
    # Click reads `--frontend=.\dist` and `-f.\dist` as values with a path.
    tail = rest if rest[0].startswith("-") else rest[1:]
    return any(_carries_a_value(arg) for arg in tail)


def is_relocatable_invocation(argv, environ):
    """True when this invocation is desktop-managed or provably cwd-independent.

    The argv arm matters on its own: it fixes users whose desktop build predates
    the Rust-side fix and so sets no marker.
    """
    args = [arg for arg in argv if arg]
    if not args:
        return False
    # Click handles top-level -h/--help/--version eagerly, so only `studio --help` arrives here.
    if all(arg in _HELP_FLAGS for arg in args):
        return True
    if args[0] != "studio":
        # The marker is inherited by everything the backend spawns, so it authorises studio commands only.
        return False
    rest = args[1:]
    if environ.get(DESKTOP_MANAGED_ENV) == "1" and not _takes_a_path(rest):
        # Unknown desktop command shapes are allowed, but never path-carrying ones.
        return True
    if rest and all(arg in _HELP_FLAGS for arg in rest):
        return True
    if _is_desktop_backend_launch(rest):
        return True
    return tuple(rest) in _STUDIO_COMMANDS


# Unsloth resolves these with Path.resolve(), so moving first would silently retarget them.
_RELATIVE_PATH_ENV = (
    "UNSLOTH_HOME",
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_STUDIO_DOCUMENTS_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "UNSLOTH_STUDIO_SANDBOX_HOME",
    "STUDIO_LOCAL_REPO",
    "UNSLOTH_LLAMA_CPP_PATH",
    "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR",
    "UNSLOTH_SD_CPP_PATH",
    "UNSLOTH_WHISPER_CPP_PATH",
    "UNSLOTH_AUDIO_CPP_PATH",
    "LLAMA_SERVER_PATH",
    "WHISPER_SERVER_PATH",
    "AUDIOCPP_SERVER_PATH",
    "SD_CLI_PATH",
    "SD_SERVER_PATH",
    # URL and HF-repo spellings are absent: they name no local file.
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_MMPROJ",
    "LLAMA_ARG_MODEL_DRAFT",
    "LLAMA_ARG_SPEC_DRAFT_MODEL",
    "AMDGPU_ASIC_ID_TABLE_PATH",
    "VLLM_CACHE_ROOT",
    "GGML_BACKEND_PATH",
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
    "UNSLOTH_COMPILE_LOCATION",
    "TORCHINDUCTOR_CACHE_DIR",
    # storage_roots.py fills these only when blank, so a user's relative value is kept as written.
    "TORCH_EXTENSIONS_DIR",
    "TORCH_HOME",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "CUDA_CACHE_PATH",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "DATA_DESIGNER_HOME",
    "DATA_DESIGNER_MANAGED_ASSETS_PATH",
    "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COND_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    # transformers appends this to sys.path.
    "HF_MODULES_CACHE",
    "HF_TOKEN_PATH",
    # Authoritative when non-blank (storage_roots.py).
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

# Best effort: only `studio update --local` reads it and it keeps the hard error.
_BEST_EFFORT_ENV = frozenset(("STUDIO_LOCAL_REPO",))

# The most a Windows environment variable holds, terminator included.
_WINDOWS_ENV_VALUE_LIMIT = 32767

# Windows' separator, not os.pathsep, which would split "D:\shared" elsewhere.
_PATH_LIST_SEPARATOR = ";"

# Anchored entry by entry: a relative PYTHONPATH entry would let the new directory shadow a
# managed import. PATH is left out: one unresolvable entry is not worth refusing the move.
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
        try:
            anchored = _anchor(name, value, cwd, pathmod, abspath, expandvars, expanduser)
        except Exception:
            if name not in _BEST_EFFORT_ENV:
                raise
            continue
        if anchored is not None:
            if len(anchored) >= _WINDOWS_ENV_VALUE_LIMIT:
                raise ValueError(
                    f"{name} does not fit in an environment variable once it "
                    "names its folder in full"
                )
            environ[name] = anchored
            pinned.append(name)
    for name in _PATH_LIST_ENV:
        raw = environ.get(name) or ""
        if not raw.strip():
            continue
        entries = raw.split(_PATH_LIST_SEPARATOR)
        anchored_entries = [
            _anchor_list_entry(name, e, cwd, pathmod, abspath, expandvars, expanduser)
            for e in entries
        ]
        if anchored_entries != entries:
            joined = _PATH_LIST_SEPARATOR.join(anchored_entries)
            if len(joined) >= _WINDOWS_ENV_VALUE_LIMIT:
                raise ValueError(
                    f"{name} does not fit in an environment variable once each "
                    "entry names its folder in full"
                )
            environ[name] = joined
            pinned.append(name)
    return pinned


# Anchoring these changes their meaning. A directory really called "[llama]" is legal on Windows.

# MLX_HOSTFILE holds either a filename or the host list itself as JSON.
_INLINE_JSON_ENV = frozenset(("MLX_HOSTFILE", "MLX_IBV_DEVICES"))

# huggingface_hub expandvars these but hf_cache_settings._canonical() does not: expand here.
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

# A bare on/off token is not a directory (diffusion_prequant.py).
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
            # The empty entry is the working directory by definition.
            if entry.strip() and not exists(entry):
                continue
            anchored = _anchor_list_entry(
                "PYTHONPATH", entry, cwd, pathmod, abspath, None, expanduser
            )
        except Exception:
            # Best effort: an import root this process already holds is not worth refusing the move over.
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
    r"""One entry of a path list, anchored, or left as written.

    PYTHONPATH has two spellings that follow the process rather than the caller:
    an empty component is the working directory itself, and `~` is never expanded
    there, so Python reads `~\plugins` as an ordinary relative folder and so does
    this.
    """
    entry = entry.strip()
    if name == "PYTHONPATH":
        if not entry:
            return cwd
        expanduser = lambda value: value
    return _anchor(name, entry, cwd, pathmod, abspath, expandvars, expanduser) or entry


def _expand_settled(value, expandvars):
    """The value expanded exactly once, or None if one pass does not settle it.

    One pass is what every reader does, so one pass is what the guard does. The
    result is only usable if expanding it again would change nothing, because the
    reader expands whatever gets written back: a value that still holds a
    reference (a nested %LOCALAPPDATA% that itself holds %USERPROFILE%, an escaped
    %%NAME%%, a self-reference) would be expanded a second time by the reader and
    read as a folder with another drive in the middle of it. Those are left
    exactly as written instead.
    """
    expanded = expandvars(value)
    return expanded if expandvars(expanded) == expanded else None


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
        # llama_cpp.py hands UNSLOTH_LLAMA_CPP_PATH straight to Path(), so expand ~ now.
        value = (expanduser or pathmod.expanduser)(value)
    if name in _EXPANDED_ENV and value:
        # Written out so readers that expand and that do not agree. Unset variables stay as written.
        expandvars = expandvars or _os.path.expandvars
        settled = _expand_settled(value, expandvars)
        if settled is None:
            # If one pass does not settle it, the reader would expand twice: refuse the move.
            once = expandvars(value)
            if _is_fully_qualified(once, pathmod):
                return None
            raise ValueError(f"{name} does not expand to one folder")
        value = settled
    if not value:
        return None
    if _is_fully_qualified(value, pathmod):
        return value if value != original else None
    if not _names_a_path(name, value):
        return None
    if pathmod.splitdrive(value)[0] or value.startswith(("\\", "/")):
        # Ask the OS: "D:cache" and "\cache" depend on per-drive state join() does not know.
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
    # An unmounted profile still has a writable parent; makedirs would shadow the real one.
    if not home_isdir(home):
        return None
    work_dir = pathmod.join(home, WORK_DIR_NAME)
    try:
        makedirs(work_dir, exist_ok = True)
    except OSError:
        # An unwritable home is a broken profile and Unsloth must write there anyway.
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
    # allow_public here only: relocating to C:\Users\Public would share state across accounts.
    home = safe_user_dir(environ, windir, pathmod, sep, expanduser, allow_public = True)
    if home:
        # Quote it (spaces); PowerShell single quotes are verbatim, cmd needs double quotes.
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
        # `relocate = False` is the library case: command modules already resolved their roots.
        return blocked_message(cwd, argv, environ, windirs, pathmod, sep, expanduser), "red", True

    target = relocation_target(environ, windirs, pathmod, sep, expanduser, makedirs, home_isdir)
    unpinnable = None
    # Snapshot so nothing stays rewritten unless the move happens.
    environ_before = dict(environ)
    if syspath is None:
        # Resolved here, or the console script would rewrite sys.path with no snapshot.
        import sys as _sys
        syspath = _sys.path
    syspath_before = list(syspath)
    if target is not None:
        try:
            # Before moving, or a relative override would resolve under the new directory.
            pin_relative_overrides(environ, cwd, pathmod, abspath, expandvars, expanduser)
            # This interpreter resolves relative PYTHONPATH entries on every import.
            pin_relative_sys_path(cwd, pathmod, syspath, abspath, exists, expanduser)
        except Exception as error:
            # An environment we cannot pin is one we must not move underneath.
            unpinnable = error
            target = None
    moved = False
    if target is not None:
        try:
            chdir(target)
        except OSError:
            target = None
        else:
            moved = True
            # Confirm where it landed rather than trusting chdir not to raise.
            try:
                if is_system_dir(getcwd(), windirs, pathmod, sep):
                    target = None
            except OSError:
                target = None
            if target is None:
                # The pinned values only mean the same folder from where they were written: go back.
                try:
                    chdir(cwd)
                except OSError:
                    pass
                else:
                    moved = False
    if target is None and not moved:
        if environ_before != environ:
            environ.clear()
            environ.update(environ_before)
        if syspath_before != syspath:
            syspath[:] = syspath_before
    if unpinnable is not None:
        # Named separately: blaming the user folder would mislead.
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
        # Fail closed. This text lands in the desktop's logs.
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
