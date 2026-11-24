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
"""Unit tests for src/vllm_omni_torchtpu/patches/ulysses.py."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.diffusion.distributed import comm as diffusion_comm

from vllm_omni_torchtpu.patches import ulysses as ulysses_patches

# Capture upstream functions at collection, before any test installs patches.
_UPSTREAM_ALL_TO_ALL_4D = diffusion_comm.all_to_all_4D
_UPSTREAM_SEQ_ALL_TO_ALL_4D_APPLY = diffusion_comm.SeqAllToAll4D.apply


def test_ulysses_inference_all_to_all_4d(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.attention.parallel import ulysses as ulysses_mod
    from vllm_omni.diffusion.distributed import comm
    from vllm_omni.diffusion.models.ltx2 import ltx2_sequence_parallel

    # Every importer binds the one SeqAllToAll4D class, so one assignment
    # covers them all.
    assert ulysses_mod.SeqAllToAll4D is comm.SeqAllToAll4D
    assert ltx2_sequence_parallel.SeqAllToAll4D is comm.SeqAllToAll4D

    delegated: list[dict] = []
    sentinel = torch.zeros(1)

    def upstream_spy(*args, **kwargs):
        delegated.append(kwargs)
        if kwargs.get("_output_destination") is not None:
            return sentinel
        return _UPSTREAM_ALL_TO_ALL_4D(*args, **kwargs)

    monkeypatch.setattr(comm, "all_to_all_4D", upstream_spy)
    monkeypatch.setattr(comm.SeqAllToAll4D, "apply", _UPSTREAM_SEQ_ALL_TO_ALL_4D_APPLY)
    ulysses_patches._patch_ulysses_inference_all_to_all_4d()
    ulysses_patches._patch_ulysses_inference_all_to_all_4d()  # idempotent
    assert comm.all_to_all_4D is not upstream_spy
    assert comm.SeqAllToAll4D.apply is not _UPSTREAM_SEQ_ALL_TO_ALL_4D_APPLY

    # Bit-exact with upstream for (2, 1) and (1, 2) across 4 simulated ranks,
    # through both all_to_all_4D and SeqAllToAll4D.apply.
    world_size = 4
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: world_size
    )

    def _run_simulated_world(fn, rank_inputs, scatter_idx, gather_idx):
        captured_sends = []

        def collect_send(recv_buf, send_buf, group=None):
            del group
            captured_sends.append(send_buf.clone())
            recv_buf.zero_()

        monkeypatch.setattr(torch.distributed, "all_to_all_single", collect_send)
        for r in range(world_size):
            fn(rank_inputs[r], scatter_idx, gather_idx, group="pg")
        # Transpose rank chunks across the world: recv[dst, src] = send[src, dst]
        transposed = torch.stack(captured_sends, dim=0).transpose(0, 1).contiguous()
        outputs = []
        for r in range(world_size):
            monkeypatch.setattr(
                torch.distributed,
                "all_to_all_single",
                lambda recv_buf, send_buf, group=None, rr=transposed[r]: recv_buf.copy_(
                    rr
                ),
            )
            outputs.append(fn(rank_inputs[r], scatter_idx, gather_idx, group="pg"))
        return outputs

    def _via_seq_apply(t, scatter_idx, gather_idx, group="pg"):
        return comm.SeqAllToAll4D.apply(group, t, scatter_idx, gather_idx, False)

    torch.manual_seed(0)
    for shape, scatter_idx, gather_idx in (
        ((2, 3, 8, 4), 2, 1),
        ((2, 12, 2, 4), 1, 2),
    ):
        inputs = [torch.randn(*shape) for _ in range(world_size)]
        want = _run_simulated_world(
            _UPSTREAM_ALL_TO_ALL_4D, inputs, scatter_idx, gather_idx
        )
        for fn in (comm.all_to_all_4D, _via_seq_apply):
            got = _run_simulated_world(fn, inputs, scatter_idx, gather_idx)
            for g, w in zip(got, want, strict=True):
                torch.testing.assert_close(g, w, rtol=0, atol=0)
    assert delegated == []

    # use_sync=True goes to upstream, which synchronizes.
    sync_calls: list[bool] = []
    monkeypatch.setattr(
        comm.current_omni_platform, "synchronize", lambda: sync_calls.append(True)
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda recv_buf, send_buf, group=None: recv_buf.copy_(send_buf),
    )
    x = torch.randn(2, 3, 8, 4)
    comm.all_to_all_4D(x, 2, 1, group="pg", use_sync=True)
    assert sync_calls == [True] and len(delegated) == 1

    # _output_destination goes to upstream's owned-destination path.
    dest = torch.empty(1, 3, 8, 4)
    assert comm.all_to_all_4D(x, 1, 2, group="pg", _output_destination=dest) is (
        sentinel
    )
    assert delegated[-1]["_output_destination"] is dest

    # A world size of 1 returns upstream's fresh contiguous copy.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    out = comm.all_to_all_4D(x.transpose(1, 2), 2, 1, group="pg")
    assert out is not x and out.is_contiguous()
    torch.testing.assert_close(out, x.transpose(1, 2), rtol=0, atol=0)

    # Non-4D input fails like upstream.
    with pytest.raises(AssertionError, match="4D"):
        comm.all_to_all_4D(torch.randn(2, 3, 4), 2, 1, group="pg")
