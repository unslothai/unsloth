from contextlib import contextmanager
from copy import deepcopy
from functools import partial

from hqq.core.quantize import Quantizer
from pack import pack, unpack


@contextmanager
def patch_hqq_packing():
    """
    Context manager that patches `Quantizer.pack` and `Quantizer.unpack` to use custom packing for i4 / i8

    """
    hqq_bitpack_map_original = deepcopy(Quantizer.bit_to_packing)
    hqq_pack_original = deepcopy(Quantizer.pack)
    hqq_unpack_original = deepcopy(Quantizer.unpack)

    Quantizer.pack["8bit_u8"] = partial(pack, nbits=8)
    Quantizer.unpack["8bit_u8"] = partial(unpack, nbits=8)
    Quantizer.pack["4bit_u8"] = partial(pack, nbits=4)
    Quantizer.unpack["4bit_u8"] = partial(unpack, nbits=4)

    try:
        yield
    finally:
        Quantizer.bit_to_packing = hqq_bitpack_map_original
        Quantizer.pack = hqq_pack_original
        Quantizer.unpack = hqq_unpack_original
