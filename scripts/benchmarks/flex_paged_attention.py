# Adapted from attention-gym
# Original source: https://github.com/pytorch-labs/attention-gym
# License: BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
# Copyright (c) 2023, Driss Guessous

# the original implementation has some bugs and has some feature that lives outside of the PageTable class

from typing import Optional
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    noop_mask,
    create_block_mask,
)

create_block_mask = torch.compile(create_block_mask)


def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
    return (x + multiple - 1) // multiple


class PagedKVCache(torch.nn.Module):
    def __init__(self, page_table, n_heads, head_dim, dtype):
        super().__init__()
        cache_shape = (1, n_heads, page_table.n_pages * page_table.page_size, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

        self.page_table = page_table

    def update(self, input_pos, k_val, v_val, batch_idx=None):
        assert batch_idx is not None, "batch_idx is required for paged kv cache, are you using non-paged attention?"

        if batch_idx.ndim == 1:
            # batch_idx should be [B] (decode)
            return self.page_table.assign(batch_idx, input_pos, k_val, v_val, self.k_cache, self.v_cache)
        else:
            assert batch_idx.ndim == 2, "batch_idx must be 1D or 2D"
            # batch_idx should be [1, L] (batch prefill)
            return self.page_table.assign_prefill_no_paging(batch_idx, input_pos, k_val, v_val, self.k_cache, self.v_cache)


class PageTable:
    """
    PageTable is a modified version of PagedAttention from attention-gym.

    PageTable improves it by:
    - maintaining a cpu copy of the page table, to avoid device-to-host transfers
    - support batch prefill
    - fix the bug in the original code in mask_mod and score_mod by mapping physical batch index to logical batch index
    - subsuming the free_batch_idx into the page table, so we don't need to maintain it separately
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
    ):
        self.n_pages = n_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.device = device

        # page table: [logical_batch_idx, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones((max_batch_size, self.n_pages), dtype=torch.int64, device=device)
        self.page_table[0, :] = 0  # page 0 is reserved for simpler code in assign_prefill_no_paging
        self.page_table_cpu = [[] for _ in range(max_batch_size)]

        self.capacity = [0 for _ in range(max_batch_size)]  # capacity: batch_idx -> number of pages allocated * page size
        self.free_pages = list(reversed(range(1, n_pages)))  # page 0 is reserved for simpler code in assign_prefill_no_paging
        self.free_batch_idx = list(reversed(range(1, max_batch_size)))  # batch_idx 0 is reserved for no-op

        # [logical_batch_idx, physical_page_idx] -> logical_page_idx
        self.physical_to_logical = -torch.ones((max_batch_size, n_pages), dtype=torch.int64, device=device)

    def can_reserve(self, size: int, batch_idx_int: int | None = None) -> bool:
        """check if we can reserve new pages for an existing request or a new request, without gpu operations"""
        if batch_idx_int is None:
            # check if we can schedule a new request
            return self.pages_available * self.page_size >= size and len(self.free_batch_idx) > 0
        else:
            # check if we can reserve new pages for an existing request
            return self.reserve(batch_idx_int, None, size, dry_run=True)

    def allocate(self) -> int:
        """allocate a new batch"""
        batch_idx = self.free_batch_idx.pop()

        self.capacity[batch_idx] = 0
        self.physical_to_logical[batch_idx, :] = -1
        self.page_table[batch_idx, :] = -1
        return batch_idx

    @property
    def pages_available(self) -> int:
        return len(self.free_pages)

    def reserve(self, batch_idx_int: int, batch_idx: torch.Tensor, seq_len: int, dry_run: bool = False) -> bool:
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx_int (int): batch index to be reserved;
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.

        Returns:
            bool: True if the reservation was successful, False if the reservation was not successful (no space, and in this case, no update is done)
        """

        if seq_len <= self.capacity[batch_idx_int]:
            return True

        num_pages_to_allocate = _cdiv(seq_len - self.capacity[batch_idx_int], self.page_size)

        can_allocate = num_pages_to_allocate <= self.pages_available
        if dry_run:
            return can_allocate

        if not can_allocate:
            raise RuntimeError(
                f"Cannot reserve {num_pages_to_allocate} pages for a sequence of length {seq_len} "
                f"in batch {batch_idx_int}. Only {self.pages_available} pages available. "
                f"Current capacity is {self.capacity[batch_idx_int]} tokens."
            )

        start_page_idx = self.capacity[batch_idx_int] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages_list = self.free_pages[-num_pages_to_allocate:]
        allocated_pages = torch.tensor(allocated_pages_list, device=self.device)
        # update page table
        self.page_table[batch_idx, start_page_idx:end_page_idx] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx,
            end_page_idx,
            device=self.device,
        )
        # update cpu side metadata
        self.page_table_cpu[batch_idx_int] += allocated_pages_list
        self.free_pages = self.free_pages[:-num_pages_to_allocate]
        self.capacity[batch_idx_int] += num_pages_to_allocate * self.page_size
        return True

    def erase(self, batch_idx: int) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (int): batch index to be removed;
        """
        # NOTE: the GPU side data will only be reset/overwritten when we allocate it for a new batch
        self.free_batch_idx.append(batch_idx)
        allocated_pages_cpu = self.page_table_cpu[batch_idx]
        self.free_pages.extend(reversed(allocated_pages_cpu))
        self.page_table_cpu[batch_idx] = []

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            cache (Tensor): the cache to store the values; shape:`(1, H, MAX_S, D)`
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        _, H_cache, MAX_S, D_cache = k_cache.shape
        assert H_cache == H, "number of heads must match"
        assert MAX_S >= S, "cache must have enough space"
        assert D_cache == K_D, "hidden dim must match"
        assert input_pos.shape == (B, S), "input_pos must have the same shape as val"
        assert batch_idx.shape == (B,), "batch_idx must have one dimension only"

        V_D = v_val.shape[3]
        if B != batch_idx.shape[0]:
            raise RuntimeError(f"Expect val and batch_idx have the same batch size but got B={B} and B={batch_idx.shape[0]}.")
        if H != k_cache.shape[1]:
            raise RuntimeError(f"Expect val and cache has the same number of heads but got H={H} and H={k_cache.shape[1]}.")
        if S != input_pos.shape[1]:
            raise RuntimeError(f"Expect val and input_pos has the same length but got S={S} and S={input_pos.shape[0]}.")
        if K_D != k_cache.shape[3]:
            raise RuntimeError(f"Expect k_val and k_cache has the same hidden dim but got D={K_D} and D={k_cache.shape[3]}.")
        if V_D != v_cache.shape[3]:
            raise RuntimeError(f"Expect v_val and v_cache has the same hidden dim but got D={V_D} and D={v_cache.shape[3]}.")

        # find address
        logical_block_idx = input_pos // self.page_size  # [B, S]
        logical_block_offset = input_pos % self.page_size  # [B, S]

        # NOTE: this code path is only used for decoding. For batch prefill, use assign_prefill_no_paging() instead
        physical_block_idx = torch.gather(self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)).to(torch.int32)  # [B, S]

        addr = (physical_block_idx * self.page_size + logical_block_offset).view(-1)  # [B*S]

        k_val = k_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, K_D)
        v_val = v_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, V_D)

        k_cache[:, :, addr, :] = k_val
        v_cache[:, :, addr, :] = v_val

        return k_cache, v_cache

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_sizebut got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)

        assert batch_idx.ndim == 1, "batch_idx must be a 1D tensor"
        assert batch_idx.shape[0] == B, "batch_idx must have the same shape as block_mask"
        assert B <= self.max_batch_size, "batch_idx must be less than or equal to max_batch_size"

        page_table = self.page_table[batch_idx]

        def transform(num_blocks, indices):
            """
            transform the block mask from [B, H, num_q_blocks, num_logical_kv_blocks]
            to [B, H, num_q_blocks, num_physical_kv_blocks]

            kv_num_blocks: [B, H, num_q_blocks] -> unchanged
            kv_indices: [B, H, num_q_blocks, num_logical_kv_blocks] -> [B, H, num_q_blocks, num_physical_kv_blocks]
            """
            if num_blocks is None:
                return None, None
            new_kv_num_blocks = num_blocks.clone()
            new_kv_indices = torch.zeros((B, H, ROWS, self.n_pages), dtype=torch.int32, device=device)
            new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(page_table, 1, indices.view(B, -1).to(torch.int64)).view(block_mask.kv_indices.shape).to(torch.int32)
            )
            return new_kv_num_blocks, new_kv_indices

        new_kv_num_blocks, new_kv_indices = transform(block_mask.kv_num_blocks, block_mask.kv_indices)
        new_full_kv_num_blocks, new_full_kv_indices = transform(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, batch_idx)

        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_logical_kv_idx(self, physical_batch_idx: torch.Tensor, physical_kv_idx: torch.Tensor, batch_idx: torch.Tensor):
        logical_batch_idx = batch_idx[physical_batch_idx]
        physical_kv_block = physical_kv_idx // self.page_size
        physical_kv_offset = physical_kv_idx % self.page_size
        logical_block_idx = self.physical_to_logical[logical_batch_idx, physical_kv_block]
        logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
        is_valid = logical_block_idx >= 0
        safe_logical_kv_idx = logical_kv_idx.clamp(min=0)
        return is_valid, safe_logical_kv_idx

    def get_mask_mod(self, mask_mod: Optional[_mask_mod_signature], batch_idx: torch.Tensor) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            is_valid, safe_logical_kv_idx = self.get_logical_kv_idx(b, physical_kv_idx, batch_idx)
            return torch.where(is_valid, mask_mod(b, h, q_idx, safe_logical_kv_idx), False)

        return new_mask_mod

    # NOTE: not used in the current codebase
    def get_score_mod(self, score_mod: Optional[_score_mod_signature], batch_idx: torch.Tensor) -> _score_mod_signature:
        """
        Converts a score_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            score_mod (_score_mod_signature): score_mod based on the logical block index.
        """
        if score_mod is None:
            score_mod = _identity

        def new_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            is_valid, safe_logical_kv_idx = self.get_logical_kv_idx(b, physical_kv_idx, batch_idx)
            return torch.where(
                is_valid,
                score_mod(score, b, h, q_idx, safe_logical_kv_idx),
                float("-inf"),
            )

        return new_score_mod

    def create_causal_blockmask(self, B, L):
        """A minimal, unoptimized causal block mask creation function"""

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(causal, B=B, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=self.page_size, device=self.device)

    def create_prefill_blockmask_no_paging(self, batch_idx: Tensor, BLOCK_SIZE: int = 128):
        """
        there's no prefix sharing implemented, batch_idx is the document id, batch_idx is not guaranteed to be sorted
        """
        assert batch_idx.ndim == 2, "batch_idx must be a 2D tensor"
        assert batch_idx.shape[0] == 1, "batch_idx must have batch size 1"
        L = batch_idx.shape[1]
        docs = batch_idx.view(-1)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        return create_block_mask(document_causal, B=1, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=BLOCK_SIZE)

    # we assign prefill to the cache, similar to assign(), except we don't return the k_cache, v_cache, we only return the k_val, v_val
    def assign_prefill_no_paging(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        assigns kv and returns the original kv

        batch_idx: [1, L]
        input_pos: [1, L]
        k_val: [1, H, L, D]
        v_val: [1, H, L, D]
        k_cache: [1, H, MAX_S, D]
        v_cache: [1, H, MAX_S, D]
        """

        assert batch_idx.ndim == 2, "batch_idx must be a 2D tensor"
        assert input_pos.ndim == 2, "input_pos must be a 2D tensor"
        assert k_val.ndim == 4, "k_val must be a 4D tensor"
        assert v_val.ndim == 4, "v_val must be a 4D tensor"
        assert k_cache.ndim == 4, "k_cache must be a 4D tensor"
        assert v_cache.ndim == 4, "v_cache must be a 4D tensor"
        assert batch_idx.shape[0] == 1, "batch_idx must have batch size 1"

        input_pos_block_idx = input_pos // self.page_size
        input_pos_offset_in_block = input_pos % self.page_size
        physical_kv_idx = self.page_table[batch_idx, input_pos_block_idx] * self.page_size + input_pos_offset_in_block
        k_cache[:, :, physical_kv_idx.view(-1), :] = k_val
        v_cache[:, :, physical_kv_idx.view(-1), :] = v_val

        return k_val, v_val
