# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU runtime patches for Ulysses sequence-parallel collectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ULYSSES_INFERENCE_A2A_PATCHED = "_tpu_ulysses_inference_all_to_all_4d_patched"


def _eager_all_to_all_single(send_buf: torch.Tensor, group: Any) -> torch.Tensor:
    recv_buf = torch.empty_like(send_buf)
    torch.distributed.all_to_all_single(recv_buf, send_buf, group=group)
    return recv_buf


def _ulysses_all_to_all_4d_layout(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    seq_world_size: int,
    group: Any,
    all_to_all_fn: Callable[[torch.Tensor, Any], torch.Tensor],
) -> torch.Tensor:
    """Shared ``(2, 1)`` / ``(1, 2)`` 4D Ulysses all-to-all layout transform.

    Parameterised by ``all_to_all_fn`` so the eager inference patch and the
    Dynamo-traceable functional-collective path share one layout definition.
    """
    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = input.shape
        send_buf = (
            input.reshape(bs, shard_seqlen, seq_world_size, hc // seq_world_size, hs)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        recv_buf = all_to_all_fn(send_buf, group)
        return (
            recv_buf.reshape(
                shard_seqlen * seq_world_size, bs, hc // seq_world_size, hs
            )
            .transpose(0, 1)
            .contiguous()
        )
    if scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input.shape
        send_buf = (
            input.reshape(bs, seq_world_size, seqlen // seq_world_size, shard_hc, hs)
            .permute(1, 3, 2, 0, 4)
            .contiguous()
        )
        recv_buf = all_to_all_fn(send_buf, group)
        return (
            recv_buf.reshape(
                shard_hc * seq_world_size, seqlen // seq_world_size, bs, hs
            )
            .permute(2, 1, 0, 3)
            .contiguous()
        )
    raise RuntimeError(
        f"Invalid (scatter_idx, gather_idx)=({scatter_idx}, {gather_idx}); "
        "expected (2, 1) or (1, 2)"
    )


def _patch_ulysses_inference_all_to_all_4d() -> None:
    """Run inference Ulysses 4D all-to-all without autograd boxing.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/distributed/comm.py`` (``all_to_all_4D`` and
    ``SeqAllToAll4D``).

    Scope: process-wide. ``comm.all_to_all_4D`` and ``SeqAllToAll4D`` are shared
    by every Ulysses model (LTX-2, Wan, Kandinsky6, ...); every module that
    imports ``SeqAllToAll4D`` binds the same class, so setting ``apply`` once
    covers them all. ``SeqAllToAll4D`` defines no backward, so skipping
    ``torch.autograd.Function.apply`` (called 288 times per LTX-2 step under
    ``torch.no_grad()``) changes nothing but host time. The ``(1, 2)`` layout
    uses one ``permute`` instead of upstream's chained transposes (``(2, 1)``
    was already a single ``transpose(0, 2)``) and both are bit-exact with
    upstream. ``use_sync=True``, ``_output_destination``,
    a world size of 1 and other index pairs go to upstream ``all_to_all_4D``.
    Modules that imported ``all_to_all_4D`` by name keep upstream's function.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni's inference ``SeqAllToAll4D`` / ``all_to_all_4D``
    skip autograd boxing and the double transpose.
    """
    from vllm_omni.diffusion.distributed import comm

    if getattr(comm.all_to_all_4D, _ULYSSES_INFERENCE_A2A_PATCHED, False):
        return

    orig_all_to_all_4d = comm.all_to_all_4D

    def _fast_all_to_all_4d(
        input: torch.Tensor,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        group: Any = None,
        use_sync: bool = False,
        *,
        _output_destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input.dim() == 4, (
            f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"
        )
        seq_world_size = torch.distributed.get_world_size(group)
        if (
            (scatter_idx, gather_idx) not in ((2, 1), (1, 2))
            or seq_world_size <= 1
            or use_sync
            or _output_destination is not None
        ):
            return orig_all_to_all_4d(
                input,
                scatter_idx,
                gather_idx,
                group=group,
                use_sync=use_sync,
                _output_destination=_output_destination,
            )
        return _ulysses_all_to_all_4d_layout(
            input,
            scatter_idx,
            gather_idx,
            seq_world_size,
            group,
            _eager_all_to_all_single,
        )

    def _fast_seq_all_to_all_4d_apply(
        group: Any,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        return _fast_all_to_all_4d(
            input,
            scatter_idx,
            gather_idx,
            group=group,
            use_sync=use_sync,
        )

    setattr(_fast_all_to_all_4d, _ULYSSES_INFERENCE_A2A_PATCHED, True)
    comm.all_to_all_4D = _fast_all_to_all_4d
    comm.SeqAllToAll4D.apply = staticmethod(_fast_seq_all_to_all_4d_apply)
    logger.info("Applied TPU patch: inference Ulysses all_to_all_4D without autograd.")
