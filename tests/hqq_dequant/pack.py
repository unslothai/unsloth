import torch


# h/t jlebar
def pack_2xint4(t):
    """
    The packing format is such that consecutive rows are packed into a lower / upper bits
    E.g.,
    Original, unpacked B (dtype i8):
    [
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9, 10, 11]
        [12, 13, 14, 15]
    ]
    Packed B:
    [
        [0|4, 1|5, 2|6, 3|7]
        [8|12, 9|13, 10|14, 11|15]
    ]
    (Note each entry in `Packed B` is shown lsb->msb)

    Args:
        t: torch.Tensor of shape (M, N)

    Returns:
        torch.Tensor of shape (M // 2, N)
    """
    assert t.dtype in (torch.int8, torch.uint8)
    t = t.reshape(t.shape[0] // 2, 2, t.shape[1]).permute(1, 0, 2)
    return (t[0] & 0xF) | ((t[1] << 4) & 0xF0)


def unpack_2xint4(packed):
    assert packed.dtype in (torch.int8, torch.uint8)
    lower, upper = packed & 0xF, (packed >> 4) & 0xF
    unpacked = torch.stack([lower, upper]).permute(1, 0, 2).reshape(-1, packed.shape[1])
    return unpacked


def pack(q, nbits, pack_dtype=torch.uint8, **kwargs):
    """
    Packs subbyte types into byte.

    Args:
        q: torch.Tensor of shape (M, N)
        nbits: 4 or 8

    In the case of 4 bits, the packed format is such that consecutive rows are packed into a lower / upper bits (see pack_2xint4)
    such that returned tensor is (M // 2, N)

    For 8 bits, the returned tensor is same as input with contiguity ensured.

    TODO:
    - Add support for 1 and 2 bit types
    """
    if nbits == 4:
        return pack_2xint4(q.to(pack_dtype))
    elif nbits == 8:
        return q.contiguous().to(pack_dtype)
    else:
        raise NotImplementedError("nbits must be 4 or 8")


def unpack(packed, nbits, compute_dtype=None, **kwargs):
    assert packed.dtype in (torch.int8, torch.uint8)
    if nbits == 4:
        unpacked = unpack_2xint4(packed)
        return (
            unpacked.contiguous().to(compute_dtype)
            if compute_dtype is not None
            else unpacked.contiguous()
        )
    elif nbits == 8:
        return (
            packed.contiguous().to(compute_dtype)
            if compute_dtype is not None
            else packed.contiguous()
        )
    else:
        raise NotImplementedError("nbits must be 4 or 8")


if __name__ == "__main__":
    q = torch.randint(0, 16, (8, 8), dtype=torch.uint8)
    packed = pack(q, nbits=8)
    unpacked = unpack(packed, nbits=8)
    assert torch.allclose(q, unpacked)

    packed = pack(q, nbits=4)
    assert packed.shape == (q.shape[0] // 2, q.shape[1])
    unpacked = unpack(packed, nbits=4)
    assert torch.allclose(q, unpacked)
