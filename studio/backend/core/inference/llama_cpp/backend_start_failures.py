# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, start failure classification."""

import os
import re
import sys

from typing import (
    Optional,
    Sequence,
)
from utils.code_integrity import (
    code_integrity_block_reason,
    code_integrity_user_message,
)
from utils.gguf_archs import (
    is_audio_cpp_gguf_architecture,
    is_speech_gguf_architecture,
)

from .binary_runtime import (
    _is_installer_entrypoint,
    _llama_lib_dir,
    _resolve_llama_binary,
)
from .tool_loop_text import (
    _FAILURE_QUOTE_CHARS,
    _FAILURE_SCAN_TAIL_CHARS,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _StartFailureMixin:
    @staticmethod
    def _is_bundled_llama_library(lib: str) -> bool:
        # The loader prints a bare soname when the file is absent but a full
        # path when it found and rejected it, so compare on the basename.
        return os.path.basename((lib or "").strip()).startswith(
            LlamaCppBackend._BUNDLED_LIB_PREFIXES
        )

    @staticmethod
    def _is_unsloth_managed_binary(binary: Optional[str]) -> bool:
        """Would `unsloth studio update` actually replace ``binary``?

        A pinned LLAMA_SERVER_PATH, or a llama-server found on PATH, is
        explicitly unmanaged (update_flow.managed_install_root returns None),
        so telling the user to update Unsloth's runtime cannot repair it.
        Unknown binary (callers that pass nothing) keeps the managed default.
        """
        if not binary:
            return True
        try:
            from utils.llama_cpp_path_settings import custom_llama_cpp_path_source
            from utils.llama_cpp_update import (
                _active_install_is_local_link,
                _llama_install_root,
            )

            # A Settings-selected checkout is user-owned even when its ordinary
            # `llama.cpp` directory name or prebuilt marker resembles our tree.
            if custom_llama_cpp_path_source() == "studio":
                return False
            # A --with-llama-cpp-dir tree is the user's own checkout reached
            # through a symlinked llama.cpp dir. The updater refuses to write
            # through that link, so it cannot repair the binary either, and its
            # entrypoint is theirs to keep.
            if _active_install_is_local_link(binary):
                return False
            return _llama_install_root(binary) is not None
        except Exception:
            return True

    @staticmethod
    def _is_llama_install_tree(binary: Optional[str]) -> bool:
        """Does ``binary`` sit in a llama.cpp tree Unsloth resolved for itself?

        The pre-existing half of the question above, kept separate because the
        two are not the same: a --with-llama-cpp-dir checkout IS the active
        install (so anything that only needs to read or copy from it still
        applies) while not being something the updater can replace.
        """
        if not binary:
            return True
        try:
            from utils.llama_cpp_update import _llama_install_root
            return _llama_install_root(binary) is not None
        except Exception:
            return True

    @staticmethod
    def _is_library_path(lib: str) -> bool:
        # glibc's own discriminator is `strchr (name, '/') == NULL`
        # (elf/dl-load.c): a slash anywhere means the name is a pathname and no
        # search happens, so a relative DT_NEEDED is as exact as an absolute
        # one. os.sep would be platform-bound and this string comes from
        # whatever host ran llama-server, so match both separators.
        lib = (lib or "").strip()
        return "/" in lib or "\\" in lib

    @staticmethod
    def _missing_library_message(lib: str, binary: Optional[str] = None) -> str:
        """The loader could not FIND ``lib`` ("cannot open shared object file")."""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib} is part of Unsloth's own "
                    "llama.cpp runtime and is missing from the install. Run "
                    "`unsloth studio update` to reinstall it, then load the model "
                    "again."
                )
            return (
                f"llama-server could not start: {lib} is part of the llama.cpp "
                "runtime that the llama-server binary in use was built with. That "
                "binary is a custom install Unsloth does not manage, so reinstall "
                "or rebuild that llama.cpp, then load the model again."
            )
        # A bare soname means the loader searched the standard directories,
        # which is what a package populates. Any name carrying a separator is a
        # pathname the loader opened directly (e.g. a vendor .so under /opt), so
        # one exact file is absent and no package will put it there.
        if LlamaCppBackend._is_library_path(lib):
            _remedy = (
                "run `unsloth studio update`"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                f"llama-server could not start: {lib} is missing from that exact "
                f"location. Restore it, or {_remedy}, then load the model again."
            )
        return (
            f"llama-server could not start: the system library "
            f"{lib or 'it needs'} is missing. Install it with your package "
            f"manager{LlamaCppBackend._SHARED_LIB_PACKAGES.get(lib, '')}, "
            "then load the model again."
        )

    @staticmethod
    def _unloadable_library_message(
        lib: str,
        reason: str,
        binary: Optional[str] = None,
    ) -> str:
        """The loader FOUND ``lib`` and refused it: the same "error while
        loading shared libraries" line also carries "file too short", "invalid
        ELF header", "wrong ELF class", "cannot open shared object file:
        Permission denied", ... Installing a package is the wrong advice there
        -- the file is already present."""
        detail = f" ({reason})" if reason else ""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib}, part of Unsloth's own "
                    f"llama.cpp runtime, could not be loaded{detail}. The install "
                    "is incomplete or mismatched. Run `unsloth studio update` to "
                    "reinstall it, then load the model again."
                )
            return (
                f"llama-server could not start: {lib}, part of the llama.cpp "
                "runtime that the llama-server binary in use was built with, "
                f"could not be loaded{detail}. That binary is a custom install "
                "Unsloth does not manage, so reinstall or rebuild that "
                "llama.cpp, then load the model again."
            )
        return (
            f"llama-server could not start: the library {lib or 'it needs'} "
            f"could not be loaded{detail}. Reinstall or update whatever "
            "provides it, then load the model again."
        )

    @staticmethod
    def _exec_path_for_launch(binary: Optional[str]) -> Optional[str]:
        """The path to actually execute, resolving a managed macOS entrypoint.

        SIP purges DYLD_* while starting the protected /bin/sh a shell
        entrypoint runs under, so the loader path we build for the child would
        not survive the wrapper's own exec (#8566). Only OUR entrypoint is
        resolved: someone's own wrapper may export backend variables or do
        other setup before its exec line, and skipping past it would silently
        drop that.

        The test is the wrapper's shape, not where it came from. Provenance is
        not enough (_llama_install_root treats the directory named by
        UNSLOTH_LLAMA_CPP_PATH as the active install with no marker file
        needed, so somebody's own checkout reads as managed), and neither is an
        explicit LLAMA_SERVER_PATH pin: pointing that at the installer's own
        entrypoint is a supported configuration, and refusing to resolve it
        just because it was named put the launch back through /bin/sh and
        reproduced the very bug this fixes. An installer-template wrapper does
        nothing but exec its target, so resolving past one is never a loss,
        however it was reached.
        """
        if not binary or sys.platform != "darwin":
            return binary
        if not LlamaCppBackend._is_unsloth_managed_binary(binary):
            try:
                from utils.llama_cpp_path_settings import custom_llama_cpp_path_source
                if custom_llama_cpp_path_source() != "studio":
                    return binary
            except Exception:
                return binary
        if not _is_installer_entrypoint(binary):
            return binary
        # template_only: the outer entrypoint being ours says nothing about
        # what it points at. Resolve only through links that are also ours.
        return str(_resolve_llama_binary(binary, template_only = True))

    @staticmethod
    def _runtime_remedy(binary: Optional[str]) -> str:
        """The repair advice for ``binary``, by provenance.

        `unsloth studio update` cannot touch a pinned LLAMA_SERVER_PATH or a
        llama-server found on PATH, so those owners get told to rebuild theirs.
        """
        return (
            "run `unsloth studio update` to reinstall the llama.cpp runtime"
            if LlamaCppBackend._is_unsloth_managed_binary(binary)
            else "reinstall or rebuild that custom llama.cpp"
        )

    # A "tried:" list names every searched path, so the reason can run to
    # kilobytes. The glibc branch is capped to one line; cap this one too.
    _MACOS_REASON_CHARS = 300

    # A dyld install name is a path, and macOS PATH_MAX is 1024. Double it and
    # anything longer is not something the loader produced.
    _DYLD_LIB_NAME_MAX_CHARS = 2048

    @staticmethod
    def _plain_dyld_lib_name(lib: str) -> str:
        """Drop dyld's search directives from an install name.

        ``@rpath/libllama.dylib`` is a lookup token, not a location, so passing
        it through as a path makes _missing_library_message tell the user to
        restore a file at a directory that does not exist. The soname is what
        they can act on, and _is_bundled_llama_library already matches on it.
        """
        # Basename, not just the directive: "@loader_path/../Frameworks/x.dylib"
        # still reads as an exact location to _is_library_path.
        if lib.startswith(("@rpath/", "@loader_path/", "@executable_path/")):
            lib = os.path.basename(lib)
        # The name is quoted from untrusted output and lands in an HTTP error.
        # A real install name is well under this; without the cap, a wrapper
        # printing a 100000-character one turned a 200-character diagnosis into
        # a 100000-character API response. PATH_MAX on macOS is 1024, so this
        # only ever truncates something dyld could not have produced.
        if len(lib) > LlamaCppBackend._DYLD_LIB_NAME_MAX_CHARS:
            lib = lib[: LlamaCppBackend._DYLD_LIB_NAME_MAX_CHARS] + "..."
        return lib

    # Ten search paths at a long path each, with room to spare. Only ever cuts
    # text a real dyld reason does not produce.
    _DYLD_REASON_MAX_CHARS = 4000
    # How far past "Library not loaded:" to look for its Reason at all. Four
    # times the cap above, so a real reason preceded by a little noise is still
    # found whole.
    _DYLD_REASON_SCAN_CHARS = 16000
    # Candidates read out of one "tried:" list. dyld searches a bounded set of
    # paths; a list longer than this is not one it produced.
    _DYLD_MAX_CANDIDATES = 128

    # dyld announces itself at the start of a line: "dyld[1234]: ", the older
    # "dyld: ", or a crash report's "Termination Reason: ... DYLD". Structural
    # on purpose: "Referenced from:" and friends are words a wrapper or a GGUF
    # metadata string can carry, and were accepted as framing before.
    _DYLD_FRAMING_RE = re.compile(
        r"(?im)^[ \t]*(?:dyld(?:\[\d+\])?:|termination reason:.*\bdyld\b)"
    )

    @staticmethod
    def _dyld_reason_after(output: str, start: int) -> str:
        """The ``Reason:`` belonging to a dyld error, flattened to one line.

        Bounded to that line plus its indented continuations, and searched only
        after the "Library not loaded:" it explains: an unrelated earlier
        "Reason:" in llama.cpp's own output would otherwise be read as dyld's,
        and an unbounded read would swallow everything printed afterwards.

        Length-capped as well. dyld lists one candidate per search path, so a
        real reason is a couple of KB at most, but the text is whatever the
        child printed: a pathological one drives the candidate scan below into
        quadratic backtracking (100KB of "'a' (" measured at 6.3s, against
        0.0s before this change), and that scan runs on the event loop.
        """
        # Windowed, not just capped afterwards: dyld prints the reason directly
        # under the "Library not loaded:" line, so nothing useful lives further
        # out, and matching against the whole remaining output would materialise
        # a group as large as whatever the child printed (a 25MB reason was
        # measured at ~366MB peak RSS through the flatten below).
        window = output[start : start + LlamaCppBackend._DYLD_REASON_SCAN_CHARS]
        match = re.compile(
            r"reason:[ \t]*([^\r\n]*(?:\r?\n[ \t]+[^\r\n]*)*)",
            re.IGNORECASE,
        ).search(window)
        if not match:
            return ""
        return " ".join(match.group(1)[: LlamaCppBackend._DYLD_REASON_MAX_CHARS].split())

    @staticmethod
    def _dyld_tried_verdict(reason: str, binary: Optional[str]) -> str:
        """The verdict for the candidate that is ours, out of a "tried:" list.

        dyld reports ``tried: '<path>' (<verdict>), '<path>' (<verdict>), ...``
        over every search path, so the list routinely mixes "no such file" for
        our own bundle with "incompatible architecture" for a leftover Intel
        Homebrew copy. Prefer the candidate in the runtime's own lib dir, else
        the first one; with no list at all the reason stands as its own verdict.
        """
        lib_dir = ""
        if binary:
            try:
                lib_dir = str(_llama_lib_dir(binary))
            except (OSError, ValueError):
                lib_dir = ""
        # finditer with a count, not findall: the reason is already capped, but
        # a list built before the first useful match is work done for nothing.
        first: Optional[str] = None
        for index, match in enumerate(re.finditer(r"'([^']+)'\s*\(([^)]*)\)", reason)):
            if index >= LlamaCppBackend._DYLD_MAX_CANDIDATES:
                break
            path, verdict = match.group(1), match.group(2)
            if first is None:
                first = verdict
            if lib_dir and os.path.dirname(path) == lib_dir:
                return verdict
        return first if first is not None else reason

    @staticmethod
    def _classify_macos_loader_failure(output: str, binary: Optional[str]) -> Optional[str]:
        """dyld / Mach-O startup failures, or None when the output has none.

        Pure text matching on captured llama-server output, so this runs (and
        is tested) on any host, not only Darwin. Kept ahead of the returncode
        heuristics in the caller: a dyld diagnostic is a fact, while signal 9
        is a guess that would otherwise blame memory (#8566).

        Nothing here fires without dyld's own framing. llama.cpp echoes GGUF
        metadata to stderr while loading, so "Library not loaded:" or "Symbol
        not found" can be a model's own general.name; a bare match outranked
        the status-127 and signal-15 branches and answered a diffusion-GGUF
        load with library advice. Every branch is now gated on the same guard
        rather than each carrying its own.
        """
        if not output:
            return None
        if not LlamaCppBackend._DYLD_FRAMING_RE.search(output):
            return None
        lowered = output.lower()

        # Built for a newer macOS than the host. Same two markers as
        # install_llama_prebuilt.looks_like_macos_incompatibility, so the
        # installer and the runtime agree on the wording.
        if ("built for macos" in lowered and "newer than running os" in lowered) or (
            "symbol not found" in lowered and "mtlresidency" in lowered
        ):
            return (
                "llama-server could not start: the installed llama.cpp runtime "
                "was built for a newer version of macOS than this Mac is "
                f"running. Update macOS, or {LlamaCppBackend._runtime_remedy(binary)} "
                "to get a build that matches this OS."
            )

        # "Library not loaded: <path>", then the "Reason:" that belongs to it.
        not_loaded = re.search(
            r"library not loaded:[ \t]*([^\r\n]+)",
            output,
            re.IGNORECASE,
        )
        if not_loaded:
            lib = LlamaCppBackend._plain_dyld_lib_name(not_loaded.group(1).strip())
            reason = LlamaCppBackend._dyld_reason_after(output, not_loaded.end())
            # dyld4 gives a per-path verdict for every path it searched, so
            # judge ours. Scanning the flattened list would let a stale Intel
            # /usr/local copy decide the diagnosis for a dylib simply absent.
            verdict = LlamaCppBackend._dyld_tried_verdict(reason, binary)
            verdict_l = verdict.lower()
            # macOS refused the Mach-O itself. Unsigned or altered code is a
            # hard SIGKILL on Apple Silicon, so never send the user hunting for
            # a corrupt GGUF or for free memory.
            if "code signature" in verdict_l or "not valid for use in process" in verdict_l:
                return (
                    f"llama-server could not start: macOS rejected the code "
                    f"signature of {lib}. The file was modified or is "
                    f"incompletely signed, so {LlamaCppBackend._runtime_remedy(binary)}."
                )
            # dyld4 says "mach-o file, but is an incompatible architecture";
            # the older wording is "mach-o, but wrong architecture".
            if "incompatible architecture" in verdict_l or "wrong architecture" in verdict_l:
                return (
                    f"llama-server could not start: {lib} was built for a "
                    "different CPU architecture than this Mac. Its install is "
                    f"for the wrong platform, so {LlamaCppBackend._runtime_remedy(binary)}."
                )
            # "no such file" (the modern per-path "tried:" list) and the older
            # bare "image not found" both mean absent; anything else means dyld
            # found it and refused it, where reinstalling a package is wrong.
            if not verdict_l or "no such file" in verdict_l or "image not found" in verdict_l:
                return LlamaCppBackend._missing_library_message(lib, binary)
            return LlamaCppBackend._unloadable_library_message(
                lib, verdict[: LlamaCppBackend._MACOS_REASON_CHARS], binary
            )

        # A symbol mismatch with no missing file. Where it was expected decides
        # the remedy: a system framework means the build wants a newer macOS
        # (reinstalling cannot make the OS export it); one of ours means our
        # own files are from different builds.
        # Only with dyld's own framing: "Symbol not found" is ordinary English
        # any wrapper on any platform can print, and claiming it would also
        # suppress the output tail the fallback carries.
        if "symbol not found" in lowered and any(
            marker in lowered for marker in ("dyld[", "dyld:", "referenced from:", "expected in:")
        ):
            expected = re.search(r"expected in:\s*(?:<[^>]*>\s*)?([^\r\n]+)", output, re.IGNORECASE)
            # Newer dyld quotes the path. Unstripped, "'/System/..." fails the
            # prefix test below and a too-new build is blamed on a mismatched
            # install instead of on the OS.
            expected_path = expected.group(1).strip().strip("'\"") if expected else ""
            if expected_path.startswith(("/System/", "/usr/lib/")):
                return (
                    "llama-server could not start: it needs a symbol this "
                    "version of macOS does not provide, so the installed "
                    "llama.cpp runtime was built for a newer macOS. Update "
                    f"macOS, or {LlamaCppBackend._runtime_remedy(binary)} to get "
                    "a build that matches this OS."
                )
            return (
                "llama-server could not start: its llama.cpp libraries are from "
                "a different build than the llama-server binary (a symbol it "
                "needs is missing). The install is mismatched, so "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )
        return None

    # The macOS shell's message for EBADARCH, printed with the offending path
    # and typically exit 126. Never reaches dyld, so it carries no dyld framing
    # and has to be matched on its own; the wording is macOS-specific enough
    # (Linux says "cannot execute binary file") to be safe anywhere.
    _BAD_CPU_TYPE_RE = re.compile(r"(?im)^.*:[ \t]*bad cpu type in executable[ \t]*$")

    @staticmethod
    def _classify_llama_start_failure(
        output: str,
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        returncode: Optional[int] = None,
        binary: Optional[str] = None,
        log_path: "Optional[Path | str]" = None,
        secrets: Sequence[Optional[str]] = (),
        extra_args: Optional[Sequence[str]] = None,
    ) -> str:
        """Classify, then redact, whatever the classification quoted.

        Redaction used to live only in the startup-diagnostics tail, so it
        covered the fallback and nothing else. Every other branch also quotes
        untrusted text -- a dyld message names the library it could not load,
        and that name comes from the child -- so a credential appearing there
        went out in the API error while the same credential in the tail was
        starred out. One boundary for every branch, rather than one per
        interpolation, which is the arrangement that let this through.

        Scrubbing twice is harmless: a redacted value no longer matches, so
        running the pass over an already-decorated message is a no-op.
        """
        return LlamaCppBackend._scrub_secret_values(
            LlamaCppBackend._classify_start_failure_text(
                output,
                gguf_path,
                model_identifier,
                returncode,
                binary,
                log_path,
                secrets,
                extra_args,
            ),
            secrets,
        )

    @staticmethod
    def _classify_start_failure_text(
        output: str,
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        returncode: Optional[int] = None,
        binary: Optional[str] = None,
        log_path: "Optional[Path | str]" = None,
        secrets: Sequence[Optional[str]] = (),
        extra_args: Optional[Sequence[str]] = None,
    ) -> str:
        """Explain *why* llama-server failed to start, from its output.

        Several distinct failures otherwise collapse into the same opaque
        "invalid GGUF or out of memory" message. Worst case: a diffusion GGUF
        loaded as a chat model -- valid file, plenty of memory, but llama.cpp
        has no such architecture, so the user is told to free memory that was
        never the problem (#5842). Pick the most specific message we can.

        ``output`` is the child's stdout and is never trusted, not even when it
        looks like something this module wrote: the only guard on our own
        framing is the one in _with_startup_diagnostics, which reads the
        ``message`` we built ourselves. An earlier revision short-circuited here
        when the *output* carried that framing, so a wrapper printing
        "llama-server output:" as its first line had its stdout returned
        verbatim: no redaction, no 2000-character cap, no log path (50067
        characters with an environment dump's credentials intact, against 2136
        scrubbed for the same bytes without the heading).
        """
        lowered = (output or "").lower()

        # First: every branch below advises reinstall, memory or administrator,
        # none of which lift a refusal to load a file that is present.
        blocked = code_integrity_block_reason(returncode) or code_integrity_block_reason(output)
        if blocked is not None:
            return code_integrity_user_message(binary or "the llama.cpp runtime", blocked)

        cuda_image_error = LlamaCppBackend._cuda_kernel_image_error(output)
        if cuda_image_error is not None:
            return LlamaCppBackend._cuda_kernel_image_message(cuda_image_error, binary, log_path)

        # The dynamic loader kills llama-server before main(), so nothing below
        # matches and the fallback blames the file or memory instead. The Linux
        # prebuilt links libgomp.so.1, which a stock container does not ship.
        # glibc prints "<prog>: error while loading shared libraries: <object>:
        # <diagnostic>[: <strerror>]" (elf/dl-catch.c fatal_error). <object> is
        # echoed verbatim, so an absolute dependency under a directory with
        # spaces keeps them ("/opt/My Runtime/libfoo.so: file too short"):
        # split on the ": " that introduces the diagnostic, not on whitespace.
        missing_lib = re.search(
            r"error while loading shared libraries:[ \t]*([^\r\n]+?)"
            r"(?::[ \t]+([^\r\n]*))?[ \t]*\r?$",
            output or "",
            re.MULTILINE,
        )
        if missing_lib:
            _lib = missing_lib.group(1).strip()
            _reason = (missing_lib.group(2) or "").strip()
            # glibc omits the object name entirely on its own allocation
            # failures ("...shared libraries: cannot create search path array"),
            # so the first word is prose, not a library. Report it unnamed.
            if not (
                "/" in _lib or "\\" in _lib or any(_e in _lib for _e in (".so", ".dylib", ".dll"))
            ):
                return LlamaCppBackend._unloadable_library_message(
                    "", f"{_lib} {_reason}".strip(), binary
                )
            # Only "cannot open shared object file" means absent; the same line
            # reports corrupt/incompatible libraries too, which are present.
            # glibc appends strerror(errno) to that diagnostic, and only ENOENT
            # ("No such file or directory") means the file is not there: EACCES
            # ("Permission denied", seen with an absolute DT_NEEDED path, an
            # unreadable parent directory, SELinux mislabels or container
            # sandboxing) means it exists and cannot be opened, where installing
            # a package is the wrong advice. A truncated tail keeps the
            # missing-library reading.
            _reason_l = _reason.lower()
            _errno_text = ""
            if "cannot open shared object file:" in _reason_l:
                _errno_text = _reason_l.split("cannot open shared object file:", 1)[1].strip()
            if not _reason_l or (
                "cannot open" in _reason_l and (not _errno_text or "no such file" in _errno_text)
            ):
                return LlamaCppBackend._missing_library_message(_lib, binary)
            return LlamaCppBackend._unloadable_library_message(_lib, _reason, binary)

        # macOS. dyld shares no wording with glibc, so every loader failure on
        # Apple Silicon fell past the branch above into the generic GGUF/memory
        # message, and none of them is about either (#8566).
        macos_detail = LlamaCppBackend._classify_macos_loader_failure(output or "", binary)
        if macos_detail:
            return macos_detail

        # The executable itself is for the wrong architecture, so the shell
        # rejects it before dyld runs and there is no loader diagnostic at all.
        if LlamaCppBackend._BAD_CPU_TYPE_RE.search(output or ""):
            return (
                "llama-server could not start: the executable was built for a "
                "different CPU architecture than this Mac, so "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )

        # Argument parsing runs before the model is touched, so a rejected flag is
        # never about the GGUF or memory. llama.cpp prints two shapes, and the
        # difference matters to the reader: an unknown flag is a typo or a flag
        # this build does not have, while a rejected VALUE is the right flag used
        # wrongly. Both are otherwise diagnosed as an invalid GGUF or an OOM.
        # Bounded, unlike the branches above, which have to see a dyld line whose
        # library name can itself be the pathological part. llama.cpp prints an
        # argument error and exits, so it is always at the end of the capture.
        scan_tail = (output or "")[-_FAILURE_SCAN_TAIL_CHARS:]

        def _short(value: str) -> str:
            """A quoted fragment of the child's output, kept to a readable length.

            The capture is a run of non-whitespace, and nothing stops a wrapper on
            LLAMA_SERVER_PATH from printing 64 KiB of it, so this is the bound that
            keeps an API error the size of an error rather than the size of the
            capture. _STARTUP_TAIL_CHARS bounds the surrounding diagnostics; this
            bounds the one piece interpolated straight from the child.
            """
            value = value.strip()
            return (
                value
                if len(value) <= _FAILURE_QUOTE_CHARS
                else (value[:_FAILURE_QUOTE_CHARS] + "...")
            )

        unknown_arg = re.search(r"error:\s*invalid argument:\s*(\S+)", scan_tail, re.IGNORECASE)
        if unknown_arg:
            # Both owners in one sentence, because nothing reaching here says which
            # one it was: Unsloth emits its own flags conditionally on the capability
            # probe, and a binary swapped underneath a cached probe rejects one of
            # those just as readily as it rejects a typo from the box. Sending every
            # reader to the extra arguments would point most of them at a box they
            # never opened.
            return (
                f"llama-server does not recognise the argument "
                f"'{_short(unknown_arg.group(1))}'. Check it in this model's extra "
                f"arguments, or reinstall llama.cpp if you did not set it."
            )

        bad_arg_value = re.search(
            r'error while handling argument "([^"]+)":\s*([^\r\n]*)',
            scan_tail,
            re.IGNORECASE,
        )
        if bad_arg_value:
            _arg = _short(bad_arg_value.group(1))
            _why = _short(bad_arg_value.group(2) or "")
            # "stoi" is what llama.cpp surfaces when std::stoi throws on a value
            # that is not a number; nobody outside the C++ standard library reads
            # that as an error message.
            if _why.lower() in {"stoi", "stof", "stod", "stoul", "stoll"}:
                _why = "the value is not a number"
            _tail = f": {_why}" if _why else ""
            # Whose flag it is decides the advice. Unsloth emits its own options
            # conditionally on the capability probe, so a build that reads
            # "--flash-attn on" differently rejects a value the box never held, and
            # sending that reader to edit their extra arguments points them at a
            # setting they cannot use to fix it. Named here only when the extras
            # really do carry the flag; an alias spelling falls back to the neutral
            # wording rather than guessing.
            _owned_by_user = False
            if extra_args:
                from core.inference.llama_server_args import _flag_name
                _failed_flag = _flag_name(_arg)
                _owned_by_user = _failed_flag is not None and any(
                    _flag_name(str(token)) == _failed_flag for token in extra_args
                )
            if _owned_by_user:
                return (
                    f"llama-server rejected the value for '{_arg}'{_tail}. Fix it in "
                    f"the extra arguments for this model."
                )
            return (
                f"llama-server rejected the value for '{_arg}'{_tail}. Check it in "
                f"this model's extra arguments, or reinstall llama.cpp if you did "
                f"not set it."
            )

        # Tensor parallelism (--split-mode tensor) is arch-gated in llama.cpp;
        # unsupported architectures abort the load with this marker. Point the
        # user at the toggle instead of a generic invalid-GGUF/OOM message.
        if "split_mode_tensor not implemented" in lowered:
            return (
                "Tensor parallelism is not supported for this model's "
                "architecture. Turn off Tensor Parallelism in the model "
                "settings and reload."
            )

        if LlamaCppBackend._is_sched_reserve_abort(output or ""):
            return LlamaCppBackend._sched_reserve_abort_message()

        # An older llama.cpp refusing a quantized KV cache under --split-mode tensor.
        # Naming the build is the point: the remedy is an update, where the generic
        # invalid-GGUF/OOM fallback sends the user to check their file or buy VRAM.
        # After the arch gate above, a different marker for a permanent per-model limit.
        if LlamaCppBackend._is_tensor_quant_kv_unsupported(lowered):
            return (
                "This llama.cpp build cannot use a quantized KV cache together "
                "with Tensor Parallelism (support landed in b9455). Update "
                "llama.cpp, or set the KV cache type to f16, or turn off Tensor "
                "Parallelism in the model settings."
            )

        # llama.cpp prints the four bytes it found with %c, so a binary header arrives as
        # unprintable characters; the generic fallback then blamed the user's memory (#8566).
        if "invalid magic characters" in lowered:
            # Not necessarily the main model: the projector and drafter report this too.
            base = os.path.basename(gguf_path) if gguf_path else ""
            named = f", loading {base}" if base else ""
            # A "._" model path proves the volume keeps xattrs in companions, whichever file
            # llama-server actually opened, and dot_clean is the remedy for the volume.
            remedy = (
                'This volume has no native extended attributes, so macOS keeps them in "._" '
                'companions: run "dot_clean -m" on the folder to remove them, or keep models '
                "on an APFS disk."
                if base.startswith("._")
                else "Re-download the model, or pick a different file."
            )
            return LlamaCppBackend._with_startup_diagnostics(
                f"llama-server opened a file that is not a GGUF{named}: it does not start with "
                "the GGUF magic. It may be that file or a companion loaded with it, such as a "
                f"vision projector or a drafter, which llama-server does not always name. {remedy}",
                output,
                log_path,
                secrets,
            )

        # Detect Ollama source up front so the arch branch can keep the
        # Ollama hint instead of the generic "unsupported arch" message.
        gguf = gguf_path or ""
        is_ollama = (
            ".studio_links" in gguf
            or os.sep + "ollama_links" + os.sep in gguf
            or os.sep + ".cache" + os.sep + "ollama" + os.sep in gguf
            or (model_identifier or "").startswith("ollama/")
        )

        # "unknown model architecture: '<arch>'": diffusion -> Images page,
        # Ollama -> Ollama hint, else a precise "unsupported" message. Exact
        # match so chat archs aren't misrouted.
        arch_match = re.search(r"unknown model architecture:\s*'([^']+)'", lowered)
        if arch_match:
            arch = arch_match.group(1)
            if is_audio_cpp_gguf_architecture(arch):
                return (
                    "This is an audio GGUF (speech, music or transcription), which llama-server "
                    "cannot run as a chat/completion model. Open it from Unsloth's Audio page instead."
                )
            if is_speech_gguf_architecture(arch):
                # Same wording rule as the pre-launch refusal above: name the build that runs,
                # not a page this file will not appear on.
                return (
                    f"'{arch}' is a text-to-speech GGUF, which llama-server cannot run as a "
                    "chat/completion model. Run the model's Transformers build (for example "
                    "unsloth/csm-1b) from Unsloth's Audio page instead."
                )
            if arch in LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES:
                return (
                    f"'{arch}' is an image / video generation GGUF, which llama-server "
                    "cannot run as a chat/completion model. Unsloth Studio cannot run "
                    "this architecture at all: neither the Images page nor the Video "
                    "page accepts it."
                )
            if arch in LlamaCppBackend._VIDEO_ARCHES:
                if LlamaCppBackend._video_arch_is_pickable(arch, gguf_path, model_identifier):
                    return (
                        f"'{arch}' is a text-to-video GGUF, which llama-server "
                        "cannot run as a chat/completion model. Open it from "
                        "Unsloth's Video page instead of a chat."
                    )
                return (
                    f"'{arch}' is a text-to-video GGUF, which llama-server cannot run as "
                    "a chat/completion model. Unsloth cannot assemble this particular "
                    "model from a GGUF, so the Video page does not list it either."
                )
            if (
                arch in LlamaCppBackend._IMAGE_ARCHES
                and LlamaCppBackend._image_arch_is_pickable(gguf_path, model_identifier)
            ) or (
                arch in LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES
                and LlamaCppBackend._ambiguous_image_arch_is_pickable(gguf_path, model_identifier)
            ):
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. Use "
                    "Unsloth's Images page to generate with local diffusion "
                    "GGUFs such as FLUX and Qwen-Image."
                )
            if arch in LlamaCppBackend._IMAGE_ARCHES:
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. No image engine "
                    "installed here can build this model family, so Unsloth's Images page "
                    "does not list it either. Updating Unsloth may add support for it."
                )
            if arch in LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES:
                # Same shared-arch resolution as the pre-launch refusal: unresolved means
                # the Images picker tags it image-diffusion-unsupported and hides the row.
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. Unsloth Studio "
                    f"could not tell which model family this file is -- '{arch}' is "
                    "shared by several -- so the Images page cannot assemble it either."
                )
            if is_ollama:
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )
            # Not "cannot be run": unlike the branches above, this includes an arch the
            # INSTALLED build predates (qwen4exp ran after a llama.cpp update).
            return (
                f"The installed llama.cpp does not recognise this GGUF's model "
                f"architecture ('{arch}'). The file is valid. If the model is newer "
                "than this llama.cpp build, updating llama.cpp may add support for it."
            )

        # Other Ollama compat failures that don't name an arch. Only when
        # the output shows a GGUF compat issue, not OOM / missing binaries.
        if is_ollama:
            gguf_compat_hints = (
                "key not found",
                "unknown model architecture",
                "failed to load model",
            )
            if any(h in lowered for h in gguf_compat_hints):
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )

        # GGUF converted for another llama.cpp format revision (#8513: pre-#24908 MiniMax-M3 lacks indexer keys).
        missing_key = re.search(r"key not found in model:\s*([\w.\-]+)", scan_tail, re.IGNORECASE)
        if missing_key:
            return LlamaCppBackend._with_startup_diagnostics(
                f"This GGUF is missing the metadata key '{_short(missing_key.group(1))}', "
                "which the installed llama.cpp requires to load it. The file was made "
                "for a different llama.cpp version of this model's format, usually an "
                "older or pre-release one; this is not a memory problem. Download an "
                "updated GGUF of this model, or update llama.cpp if the file is newer "
                "than this build.",
                output,
                log_path,
                secrets,
            )

        # A type this build lacks (prism-ml PQ2_0 / PTQ1_0 = 142 / 143) or packs differently (group-128 Q2_0, #11259).
        # A truncated download fails later instead, as "data is not within the file bounds".
        bad_type = re.search(r"tensor '([^']*)' has invalid ggml type (\d+)", scan_tail)
        bad_layout = re.search(r"tensor '[^']*' has offset \d+, expected \d+", scan_tail)
        if bad_type or bad_layout:
            detail = (
                f"tensor '{_short(bad_type.group(1))}' uses quantization type "
                f"{bad_type.group(2)}, which this build does not have"
                if bad_type
                else "its tensor layout does not match this build's quantization types"
            )
            return LlamaCppBackend._with_startup_diagnostics(
                "The installed llama.cpp cannot read this GGUF's weights "
                f"({detail}); this is not a memory problem. The file was likely made for "
                "a different llama.cpp build: a newer one, or a publisher's own fork. Pick "
                "another quant of this model, or use the llama.cpp build the "
                "model card names. If the model card says it runs on stock llama.cpp, "
                "the file may be damaged: download it again.",
                output,
                log_path,
                secrets,
            )

        # 127 without any loader line above is not specific to a library: it is
        # also what a shell wrapper entrypoint reports when its exec target is
        # gone (_llama_lib_dir supports those, as does a custom
        # LLAMA_SERVER_PATH), and what a "symbol lookup error" from a mismatched
        # runtime exits with. Name both causes instead of blaming a distro
        # package -- but still keep it off the GGUF and off memory.
        # A ROCm symbol lookup is also 127, but the generic text below would send
        # the user to reinstall a binary Vulkan and a shell launch both run fine.
        _rocm_miss = LlamaCppBackend._bundled_hip_symbol_miss(output or "")
        if _rocm_miss:
            _miss_obj, _miss_sym = _rocm_miss
            return (
                f"llama-server could not start: bundled {_short(os.path.basename(_miss_obj))} "
                f"looked up {_short(_miss_sym)} in a different ROCm than it was built "
                "against. That is a HIP/ROCR version mix, not a missing "
                "binary and not out of VRAM. Try the Vulkan backend, or "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )
        if returncode == 127:
            # Same provenance split as the library branches: the updater cannot
            # touch a pinned binary, so do not send its owner there.
            _remedy = (
                "run `unsloth studio update` to reinstall the llama.cpp runtime"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                "llama-server exited immediately (status 127): either the "
                "llama-server executable could not be found or run, or one of "
                f"the shared libraries it needs could not be loaded. Check the "
                f"llama-server log, and {_remedy}."
            )

        # SIGKILL with no diagnostic output is the OOM killer (e.g. a model too
        # large for the WSL VM's RAM cap); name it actionably.
        if returncode == -9:
            # macOS also SIGKILLs an invalid code signature before the process
            # can print, so memory is not the only reading there (#8566).
            if sys.platform == "darwin":
                # The log is often empty here, so name the likeliest cause and
                # the macOS-specific one without claiming the two are the only
                # ones: signal 9 is also a supervisor, or a plain `kill -9`.
                return LlamaCppBackend._with_startup_diagnostics(
                    "llama-server was stopped by macOS (signal 9) before it "
                    "started. This is most often out of memory, so try a "
                    "smaller or more quantized GGUF, or lower the context "
                    "length. macOS also sends signal 9 when it refuses a "
                    "binary's code signature, so if memory is not the problem, "
                    f"{LlamaCppBackend._runtime_remedy(binary)}.",
                    output,
                    log_path,
                    secrets,
                )
            return (
                "llama-server was stopped by the operating system (signal 9), "
                "most likely out of memory. Try a smaller or more quantized "
                "GGUF, lower the context length, or free memory (on WSL, raise "
                "the memory limit in .wslconfig)."
            )
        # SIGTERM is also how an unload/cancel or a supervisor stops the server,
        # so report it neutrally rather than blaming memory.
        if returncode == -15:
            return (
                "llama-server was terminated (signal 15) before it became "
                "healthy. If you cancelled or unloaded the model this is "
                "expected; otherwise check the llama-server log for the cause."
            )

        # A startup stall is not a bad GGUF; memory pressure or a proxy intercepting
        # the loopback probe can also keep /health from succeeding (#5740).
        if "health check timed out" in lowered:
            return (
                "llama-server started but stopped making progress before it became "
                "healthy on its local /health endpoint. Try a smaller context length or a more "
                "quantized GGUF, and if you use a VPN or HTTP proxy make sure "
                "localhost bypasses it (NO_PROXY=127.0.0.1,localhost)."
            )

        # Fallback: genuinely unknown failure (OOM, missing binary ...).
        # Nothing above recognised the output, so carry the evidence rather
        # than drop it; without the tail and the log path a report of this
        # message is unactionable and costs a round trip (#8566).
        return LlamaCppBackend._with_startup_diagnostics(
            "llama-server failed to start. "
            "Check that the GGUF file is valid and you have enough memory.",
            output,
            log_path,
            secrets,
        )

    # Enough of the tail to carry a stack trace or a ggml assert, bounded so a
    # chatty server cannot push an unreadable wall of text into an API error.
    _STARTUP_TAIL_CHARS = 2000

    # The labels _with_startup_diagnostics writes, so it can recognise its own
    # output and stay idempotent. Kept next to the writer: a label edited on one
    # side only would silently re-enable double appending.
    _DIAGNOSTICS_MARKERS = ("llama-server output:", "Full log: ")

    # Token shapes worth redacting on sight: a name-based list cannot cover a
    # credential the child obtained elsewhere, or one held in a variable whose
    # name says nothing (GITHUB_PAT, MY_THING).
    _SECRET_VALUE_RES = (
        re.compile(r"hf_(?:oauth_[A-Za-z0-9._~+/=-]{20,}|[A-Za-z0-9]{20,})"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
        # HTTP credentials, if a wrapper or a proxy echoes a request header.
        # token68 is [A-Za-z0-9._~+/-]* with optional "=" padding (RFC 9110
        # 11.2), so a narrower class leaves the tail of a base64 credential
        # readable; Basic (RFC 7617) is a base64 user:password pair and belongs
        # here for the same reason.
        re.compile(r"(?i)(?:bearer|basic)\s+[A-Za-z0-9._~+/\-]+=*"),
        # Our own --api-key, if a wrapper dumps its argv. token_urlsafe(32)
        # matches no shape and is in no env var, so catch it by its flag, as
        # _redacted_cmd_for_log already does for the log.
        re.compile(r"(?<=--api-key )\S+"),
        re.compile(r"(?<=--api-key=)\S+"),
        # scheme://user:secret@host, which is what DATABASE_URL / REDIS_URL and
        # a proxy variable carry. Keep the scheme, drop the userinfo.
        re.compile(r"(?<=://)[^/@\s]+(?=@)"),
    )

    # NAME = value / "NAME": "value", in the three shapes an environment dump
    # takes: shell-ish, JSON, and bare. The quoted arm ends on the delimiter it
    # opened with, not on either quote character: a JSON value may legitimately
    # contain an apostrophe, and rejecting both meant "DB_PASSWORD": "a' b"
    # fell through to the bare arm, which stops at whitespace and left " b"
    # standing in the API error. Escapes are consumed so a backslash-escaped
    # delimiter does not end the value early. Both arms are unambiguous
    # alternations of single characters, so matching stays linear; there is no
    # nested quantifier for a crafted line to exploit.
    # The name accepts the separators a config key uses, not only the ones a
    # shell variable may: a child dumping its own configuration writes
    # "db-password" or client.secret, and an identifier-only pattern never
    # reached is_secret_env_name for either. The predicate still decides.
    #
    # Third value arm: a truncated line ends the output mid-value, so
    # DB_PASSWORD="prefix supersecret has no closing delimiter. It cannot match
    # the terminated arm, and the bare arm stops at whitespace, which left the
    # rest of the credential in the API error. An opened-but-unclosed quote
    # runs to the end of the line. Ordered after the terminated arm so a
    # properly closed value still matches that one.
    _SECRET_ASSIGNMENT_RE = re.compile(
        r"""(?P<q>["']?)(?P<name>[A-Za-z_][A-Za-z0-9_.\-]{0,63})(?P=q)
            (?P<sep>[ \t]*[=:][ \t]*)
            (?:(?P<vq>["'])(?P<qval>(?:\\.|(?!(?P=vq))[^\\])*)(?P=vq)
             |  (?P<uq>["'])(?P<uval>[^\r\n]*)
             |  (?P<val>[^\s,;]+))""",
        re.VERBOSE,
    )

    @staticmethod
    def _redact_secret_assignments(text: str) -> str:
        """Redact the value beside any secret-looking NAME, however encoded.

        Positional, not literal: whatever follows the name is replaced, so
        JSON escaping, shell quoting or URL encoding cannot smuggle a value
        past the check the way a value-literal comparison allows.
        """
        try:
            from utils.prebuilt.child_env import is_secret_env_name
        except Exception:  # noqa: BLE001 - redaction must never break a load error
            return text

        def sub(m: "re.Match[str]") -> str:
            # A config key spells the same thing with a hyphen or a dot, and
            # the predicate's markers are underscored (API_KEY, PRIVATE_KEY),
            # so "api-key" missed while "api_key" matched. Normalise the
            # separator before asking; the predicate stays the one place that
            # decides what counts as a secret.
            name = m.group("name")
            if not is_secret_env_name(name.replace("-", "_").replace(".", "_")):
                return m.group(0)
            q, vq, uq = m.group("q"), m.group("vq"), m.group("uq")
            if vq:
                body = f"{vq}***{vq}"
            elif uq:
                # Unterminated: keep the opening delimiter, drop the rest of
                # the line. Inventing a closing quote would misreport what the
                # child printed.
                body = f"{uq}***"
            else:
                body = "***"
            return f"{q}{m.group('name')}{q}{m.group('sep')}{body}"

        try:
            return LlamaCppBackend._SECRET_ASSIGNMENT_RE.sub(sub, text)
        except Exception:  # noqa: BLE001
            return text

    @staticmethod
    def _scrub_secret_values(text: str, extra: Sequence[Optional[str]] = ()) -> str:
        """Redact credential-looking environment values out of ``text``.

        llama-server inherits nearly all of Unsloth's environment, so a wrapper
        script, a diagnostic build or a crash handler that echoes its env would
        otherwise put an API key straight into the load error.

        Two passes, because either alone has a hole. Matching the VALUE catches
        a credential under a name we never heard of, but only if the child
        printed it verbatim: a wrapper dumping its environment as JSON prints
        pa"ss\\word as pa\\"ss\\\\word, which is a different string and was
        reaching the API error fully reconstructible. Matching the NAME and
        redacting whatever follows it is immune to how the value was encoded,
        but only works for names that look like secrets. Doing both leaves a
        gap only for an unrecognised name AND an escaped value.
        """
        if not text:
            return text
        cleaned = LlamaCppBackend._redact_secret_assignments(text)
        # Secrets we minted and therefore know exactly: no pattern recognises
        # a token_urlsafe blob. Gathered together with the environment's own
        # values and replaced LONGEST FIRST, because two credentials can
        # overlap: with TOKEN_A=abcdefgh and TOKEN_B="abcdefgh VERYSECRET",
        # replacing the short one first rewrote the long one's only occurrence
        # to "*** VERYSECRET", after which the long one no longer matched
        # itself and its tail went out in the API error.
        literals = {literal for literal in extra if literal and len(literal) >= 8}
        # By name, reusing what decides which vars a managed server is denied,
        # so the two cannot drift: what we withhold must not come back out of
        # the child either. URL userinfo counts whatever it is named.
        try:
            from utils.prebuilt.child_env import URL_USERINFO_RE, is_secret_env_name
            for name, value in os.environ.items():
                if not value:
                    continue
                secret_name = is_secret_env_name(name)
                if len(value) < 8:
                    # Too short to replace globally ("1", "true", a port number
                    # would match everywhere), but a short password is still a
                    # password: redact it where it appears next to its name.
                    # The name and the value may each be quoted, because an env
                    # dump can be shell-ish (DB_PASSWORD='x'), JSON
                    # ({"DB_PASSWORD": "x"}), or bare, and only the bare form
                    # matched before.
                    if secret_name:
                        cleaned = re.sub(
                            rf"([\"']?{re.escape(name)}[\"']?\s*[=:]\s*)"
                            rf"([\"']?){re.escape(value)}\2",
                            r"\1\2***\2",
                            cleaned,
                        )
                    continue
                if secret_name or URL_USERINFO_RE.search(value):
                    literals.add(value)
        except Exception:  # noqa: BLE001 - redaction must never break a load error
            pass
        for literal in sorted(literals, key = len, reverse = True):
            cleaned = cleaned.replace(literal, "***")
        # By shape, for credentials that are not in our environment to match on
        # (a token the child read from its own config, a proxy's header echo).
        for pattern in LlamaCppBackend._SECRET_VALUE_RES:
            cleaned = pattern.sub("***", cleaned)
        return cleaned

    @staticmethod
    def _has_startup_diagnostics(text: str) -> bool:
        """Is ``text`` something _with_startup_diagnostics already produced?

        Matches our own framing, not the bare words: a server that printed
        "llama-server output:" itself must still get its tail attached.
        """
        return text.startswith(LlamaCppBackend._DIAGNOSTICS_MARKERS) or any(
            f"\n\n{marker}" in text for marker in LlamaCppBackend._DIAGNOSTICS_MARKERS
        )

    # Ceiling on how far the tail window is widened for redaction. A real
    # credential is far below this; the cap stops a pathological environment
    # value from turning a bounded slice into a scan of everything the child
    # ever printed.
    _MAX_SECRET_SCAN_CHARS = 262144

    @staticmethod
    def _max_secret_len(extra: Sequence[Optional[str]] = ()) -> int:
        """Longest value _scrub_secret_values could be asked to match, capped."""
        longest = 0
        for literal in extra:
            if literal:
                longest = max(longest, len(literal))
        try:
            from utils.prebuilt.child_env import URL_USERINFO_RE, is_secret_env_name
            for name, value in os.environ.items():
                if not value or len(value) <= longest:
                    continue
                if is_secret_env_name(name) or URL_USERINFO_RE.search(value):
                    longest = len(value)
        except Exception:  # noqa: BLE001 - a narrower window still redacts most
            pass
        return min(longest, LlamaCppBackend._MAX_SECRET_SCAN_CHARS)

    @staticmethod
    def _with_startup_diagnostics(
        message: str,
        output: Optional[str],
        log_path: "Optional[Path | str]",
        extra_secrets: Sequence[Optional[str]] = (),
    ) -> str:
        """Append llama-server's output tail and log path to ``message``.

        Returns ``message`` unchanged when there is neither, so the classified
        messages and the no-output case keep their exact existing text.

        Idempotent: a message that already carries a diagnostics block is
        returned as is. Only one call site exists today, so this cannot fire
        yet; it is here so a second caller (a retry that re-classifies an
        already-decorated message, say) cannot silently double the tail.
        """
        if LlamaCppBackend._has_startup_diagnostics(message):
            return message
        parts = [message]
        # Slice before filtering: _drain_stdout keeps an unterminated line
        # whole, so a runaway progress line would otherwise be walked
        # character by character just to throw all but the last 2000 away.
        #
        # Widened by the longest secret we could be asked to redact. The
        # scrubber matches a whole value, so a credential longer than this
        # window (a PEM key or a service-account blob dumped by a wrapper) was
        # cut by the slice, and the surviving suffix matched neither the
        # literal nor any token shape -- so it reached the API error. A shorter
        # secret cannot straddle the boundary: if its end is inside a window
        # wider than the secret, so is its start.
        _window = LlamaCppBackend._STARTUP_TAIL_CHARS * 4 + LlamaCppBackend._max_secret_len(
            extra_secrets
        )
        raw = (output or "")[-_window:]
        # Control characters (progress bars, colour codes) would corrupt the
        # API error; keep newlines and tabs, which carry the structure.
        tail = "".join(ch for ch in raw if ch in "\n\t" or ch.isprintable()).strip()
        if tail:
            tail = LlamaCppBackend._scrub_secret_values(tail, extra_secrets)
            tail = tail[-LlamaCppBackend._STARTUP_TAIL_CHARS :].lstrip()
            parts.append(f"llama-server output:\n{tail}")
        if log_path:
            parts.append(f"Full log: {log_path}")
        return "\n\n".join(parts)
