# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ctypes over onnxruntime-genai's C API, for AMD's RyzenAI NPU build.

AMD ships its ORT and RyzenAI EP signed, but the ryzenai-server.exe around them is not, and
Smart App Control refuses it. Loading the DLLs into Studio's own Python avoids starting any EXE.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_bool, c_char_p, c_double, c_int32, c_size_t, c_void_p
from typing import Iterator, Optional

# Loaded in this order: ORT and the EP resolve by name once the genai DLL loads.
RUNTIME_DLLS = ("onnxruntime.dll", "onnxruntime_providers_ryzenai.dll", "onnxruntime-genai.dll")

_SIGNATURES = {
    "OgaResultGetError": (c_char_p, [c_void_p]),
    "OgaDestroyResult": (None, [c_void_p]),
    "OgaCreateModel": (c_void_p, [c_char_p, POINTER(c_void_p)]),
    "OgaDestroyModel": (None, [c_void_p]),
    "OgaCreateTokenizer": (c_void_p, [c_void_p, POINTER(c_void_p)]),
    "OgaDestroyTokenizer": (None, [c_void_p]),
    "OgaCreateTokenizerStream": (c_void_p, [c_void_p, POINTER(c_void_p)]),
    "OgaDestroyTokenizerStream": (None, [c_void_p]),
    "OgaTokenizerStreamDecode": (c_void_p, [c_void_p, c_int32, POINTER(c_char_p)]),
    "OgaCreateSequences": (c_void_p, [POINTER(c_void_p)]),
    "OgaDestroySequences": (None, [c_void_p]),
    "OgaSequencesGetSequenceCount": (c_size_t, [c_void_p, c_size_t]),
    "OgaTokenizerEncode": (c_void_p, [c_void_p, c_char_p, c_void_p]),
    "OgaCreateGeneratorParams": (c_void_p, [c_void_p, POINTER(c_void_p)]),
    "OgaDestroyGeneratorParams": (None, [c_void_p]),
    "OgaGeneratorParamsSetSearchNumber": (c_void_p, [c_void_p, c_char_p, c_double]),
    "OgaGeneratorParamsSetSearchBool": (c_void_p, [c_void_p, c_char_p, c_bool]),
    "OgaCreateGenerator": (c_void_p, [c_void_p, c_void_p, POINTER(c_void_p)]),
    "OgaDestroyGenerator": (None, [c_void_p]),
    "OgaGenerator_AppendTokenSequences": (c_void_p, [c_void_p, c_void_p]),
    "OgaGenerator_IsDone": (c_bool, [c_void_p]),
    "OgaGenerator_GenerateNextToken": (c_void_p, [c_void_p]),
    "OgaGenerator_GetNextTokens": (
        c_void_p,
        [c_void_p, POINTER(POINTER(c_int32)), POINTER(c_size_t)],
    ),
}


class OgaError(RuntimeError):
    pass


class Oga:
    """The genai C API from ``dll_dir``. Raises OSError when Windows refuses a DLL."""

    def __init__(self, dll_dir: str) -> None:
        dll_dir = os.path.abspath(dll_dir)
        self._dll_dir = os.add_dll_directory(dll_dir)
        lib = None
        for name in RUNTIME_DLLS:
            lib = ctypes.CDLL(os.path.join(dll_dir, name))
        for name, (restype, argtypes) in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        self.lib = lib

    def check(self, result: Optional[int]) -> None:
        if result:
            message = self.lib.OgaResultGetError(result)
            self.lib.OgaDestroyResult(result)
            raise OgaError((message or b"onnxruntime-genai failed").decode("utf-8", "replace"))

    def new(self, create, *args) -> c_void_p:
        out = c_void_p()
        self.check(create(*args, byref(out)))
        return out


class Model:
    def __init__(self, oga: Oga, path: str) -> None:
        self.oga = oga
        self.model = oga.new(oga.lib.OgaCreateModel, os.path.abspath(path).encode())
        self.tokenizer = oga.new(oga.lib.OgaCreateTokenizer, self.model)

    def count_tokens(self, text: str) -> int:
        lib, seqs = self.oga.lib, self.oga.new(self.oga.lib.OgaCreateSequences)
        try:
            self.oga.check(lib.OgaTokenizerEncode(self.tokenizer, text.encode(), seqs))
            return int(lib.OgaSequencesGetSequenceCount(seqs, 0))
        finally:
            lib.OgaDestroySequences(seqs)

    def generate(self, prompt: str, max_new_tokens: int, **search) -> Iterator[str]:
        """Decoded text pieces; closing the iterator frees the generator."""
        oga, lib = self.oga, self.oga.lib
        seqs = oga.new(lib.OgaCreateSequences)
        params = gen = stream = None
        try:
            oga.check(lib.OgaTokenizerEncode(self.tokenizer, prompt.encode(), seqs))
            total = lib.OgaSequencesGetSequenceCount(seqs, 0) + max_new_tokens
            params = oga.new(lib.OgaCreateGeneratorParams, self.model)
            oga.check(lib.OgaGeneratorParamsSetSearchNumber(params, b"max_length", float(total)))
            for key, value in search.items():
                if isinstance(value, bool):
                    oga.check(lib.OgaGeneratorParamsSetSearchBool(params, key.encode(), value))
                else:
                    oga.check(
                        lib.OgaGeneratorParamsSetSearchNumber(params, key.encode(), float(value))
                    )
            gen = oga.new(lib.OgaCreateGenerator, self.model, params)
            oga.check(lib.OgaGenerator_AppendTokenSequences(gen, seqs))
            stream = oga.new(lib.OgaCreateTokenizerStream, self.tokenizer)
            data, count, piece = POINTER(c_int32)(), c_size_t(), c_char_p()
            while not lib.OgaGenerator_IsDone(gen):
                oga.check(lib.OgaGenerator_GenerateNextToken(gen))
                oga.check(lib.OgaGenerator_GetNextTokens(gen, byref(data), byref(count)))
                oga.check(lib.OgaTokenizerStreamDecode(stream, data[0], byref(piece)))
                yield (piece.value or b"").decode("utf-8", "replace")
        finally:
            for destroy, handle in (
                (lib.OgaDestroyTokenizerStream, stream),
                (lib.OgaDestroyGenerator, gen),
                (lib.OgaDestroyGeneratorParams, params),
                (lib.OgaDestroySequences, seqs),
            ):
                if handle:
                    destroy(handle)

    def close(self) -> None:
        self.oga.lib.OgaDestroyTokenizer(self.tokenizer)
        self.oga.lib.OgaDestroyModel(self.model)
