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
"""Unit tests for src/vllm_omni_torchtpu/patches/ltx2.py."""

from __future__ import annotations

import contextlib
import copy
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm_omni.diffusion.hooks import sequence_parallel
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_conditioning,
    ltx2_denoise,
    ltx2_guidance,
    ltx2_runtime,
    ltx2_transformer,
)

from vllm_omni_torchtpu import rope as omni_rope
from vllm_omni_torchtpu.patches import ltx2 as ltx2_patches

# Capture upstream methods at collection, before any test installs patches.
_UPSTREAM_ROPE_FORWARD = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed.forward
_UPSTREAM_APPLY_SPLIT_ROTARY_EMB = ltx2_transformer.apply_split_rotary_emb
_UPSTREAM_DENOISE_TIMESTEP_KWARGS = ltx2_runtime.LTXRuntime._denoise_timestep_kwargs
_UPSTREAM_PREDICT_NOISE = ltx2_guidance.LTXGuidanceExecutor.predict_noise
_UPSTREAM_GUIDE_MODALITY = ltx2_guidance.LTXGuidanceExecutor._guide_modality
_UPSTREAM_SYNC_GUIDANCE_STEP = (
    ltx2_runtime.LTXRuntime._synchronize_guidance_parallel_step_output
)
_UPSTREAM_LTX2_ATTN_FORWARD = ltx2_transformer.LTX2Attention.forward


class _DummyLTX2Pipeline(
    ltx2_conditioning.LTXI2VConditioningMixin,
    ltx2_runtime.LTXRuntime,
):
    def __init__(self) -> None:
        self.guidance_executor = ltx2_guidance.LTXGuidanceExecutor()
        self.unified_text_image_entry = True


@pytest.fixture
def upstream_denoise_timestep_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset LTXRuntime._denoise_timestep_kwargs to the unpatched upstream method."""
    monkeypatch.setattr(
        ltx2_runtime.LTXRuntime,
        "_denoise_timestep_kwargs",
        _UPSTREAM_DENOISE_TIMESTEP_KWARGS,
    )


@pytest.mark.parametrize(
    ("is_per_token", "expected_sharded"),
    [
        pytest.param(False, False, id="1d_per_batch_unsplit"),
        pytest.param(True, True, id="2d_per_token_sharded"),
    ],
)
def test_ltx2_sp_plan_timestep_sharding(
    monkeypatch: pytest.MonkeyPatch,
    is_per_token: bool,
    expected_sharded: bool,
) -> None:
    """1-D (B,) timesteps bypass SP sharding; 2-D (B, T) I2V timesteps are sharded."""
    batch_size = 2
    video_tokens = 16
    ulysses_degree = 2
    shards: list[tuple[tuple[int, ...], int]] = []

    def recording_sp_shard(x: torch.Tensor, dim: int, **kwargs: object) -> torch.Tensor:
        del kwargs
        shards.append((tuple(x.shape), dim))
        return x.narrow(dim, 0, x.size(dim) // ulysses_degree)

    monkeypatch.setattr(sequence_parallel, "sp_shard", recording_sp_shard)
    plan = ltx2_transformer.LTX2VideoTransformer3DModel._build_sp_plan("split")[""]
    hook = sequence_parallel.SequenceParallelSplitHook(
        plan,
        sequence_parallel.SequenceParallelConfig(ulysses_degree=ulysses_degree),
    )

    per_batch = torch.tensor([900.0, 450.0])
    assert per_batch.shape == (batch_size,)
    if not is_per_token:
        out = hook._prepare_sp_input(per_batch, plan["timestep"])
        assert out is per_batch
        assert shards == []
    else:
        per_token = per_batch.reshape(batch_size, 1).expand(batch_size, video_tokens)
        per_token = per_token.contiguous()
        out = hook._prepare_sp_input(per_token, plan["timestep"])
        assert expected_sharded
        assert out.shape == (batch_size, video_tokens // ulysses_degree)
        assert shards == [((batch_size, video_tokens), 1)]


def test_ltx2_upstream_expands_unconditioned_timestep(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Confirm upstream vLLM-Omni expands unconditioned timesteps to (B, T)."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    uncond_ctx = SimpleNamespace(
        conditioning_mask=None, conditioning_mask_for_model=None
    )

    kw_upstream = pipe._denoise_timestep_kwargs(
        ts,
        None,
        uncond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_upstream["timestep"].shape == (batch_size, video_tokens)
    assert kw_upstream["audio_timestep"].shape == (batch_size, audio_tokens)


def test_ltx2_per_batch_timestep_patch_is_idempotent(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Installing the per-batch timestep patch twice leaves the same wrapper."""
    del upstream_denoise_timestep_kwargs
    ltx2_patches._patch_ltx2_per_batch_timestep()
    patched = ltx2_runtime.LTXRuntime._denoise_timestep_kwargs
    assert getattr(patched, ltx2_patches._LTX2_PER_BATCH_TIMESTEP_PATCHED, False)

    ltx2_patches._patch_ltx2_per_batch_timestep()
    assert ltx2_runtime.LTXRuntime._denoise_timestep_kwargs is patched


def test_ltx2_per_batch_timestep_patch_keeps_t2v_1d(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Unconditioned T2V keeps 1-D (B,) video and audio timesteps after patching."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    uncond_ctx = SimpleNamespace(
        conditioning_mask=None, conditioning_mask_for_model=None
    )

    ltx2_patches._patch_ltx2_per_batch_timestep()
    kw_uncond = pipe._denoise_timestep_kwargs(
        ts,
        None,
        uncond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_uncond["timestep"].shape == (batch_size,)
    assert kw_uncond["audio_timestep"].shape == (batch_size,)


def test_ltx2_per_batch_timestep_patch_preserves_i2v_conditioning_mask(
    upstream_denoise_timestep_kwargs: None,
) -> None:
    """Conditioned I2V still overrides video timestep with the 2-D (B, T) mask."""
    del upstream_denoise_timestep_kwargs
    batch_size = 2
    video_tokens = 16
    audio_tokens = 4
    cond_tokens = 4
    pipe = _DummyLTX2Pipeline()
    ts = torch.full((batch_size,), 900.0)
    mask = torch.zeros(batch_size, video_tokens)
    mask[:, :cond_tokens] = 1.0
    cond_ctx = SimpleNamespace(conditioning_mask=mask, conditioning_mask_for_model=mask)

    ltx2_patches._patch_ltx2_per_batch_timestep()
    kw_cond = pipe._denoise_timestep_kwargs(
        ts,
        None,
        cond_ctx,
        video_token_count=video_tokens,
        audio_token_count=audio_tokens,
    )
    assert kw_cond["timestep"].shape == (batch_size, video_tokens)
    assert torch.all(kw_cond["timestep"][:, :cond_tokens] == 0.0)
    assert torch.all(kw_cond["timestep"][:, cond_tokens:] == 900.0)
    assert kw_cond["audio_timestep"].shape == (batch_size,)


@pytest.fixture
def table_syncs(monkeypatch: pytest.MonkeyPatch) -> list[list[torch.Tensor]]:
    """Records the split-RoPE patch's synchronize_tensors calls instead of syncing."""
    syncs: list[list[torch.Tensor]] = []

    def record(tensors: list[torch.Tensor], wait: bool) -> None:
        assert wait is False
        syncs.append(tensors)

    monkeypatch.setattr("vllm_torchtpu.utils.synchronize_tensors", record)
    return syncs


@pytest.fixture
def ltx2(monkeypatch: pytest.MonkeyPatch, table_syncs: list[list[torch.Tensor]]):
    """The LTX-2 transformer module with _patch_ltx2_split_rope reinstalled."""
    del table_syncs
    monkeypatch.setattr(
        ltx2_transformer.LTX2AudioVideoRotaryPosEmbed,
        "forward",
        _UPSTREAM_ROPE_FORWARD,
    )
    for name, value in (
        ("apply_split_rotary_emb", _UPSTREAM_APPLY_SPLIT_ROTARY_EMB),
        ("get_tensor_model_parallel_world_size", lambda: 1),
        ("get_tensor_model_parallel_rank", lambda: 0),
    ):
        monkeypatch.setattr(ltx2_transformer, name, value)
    ltx2_patches._patch_ltx2_split_rope()
    assert getattr(
        ltx2_transformer.LTX2AudioVideoRotaryPosEmbed.forward,
        ltx2_patches._SPLIT_ROPE_PATCHED,
    )
    return ltx2_transformer


def _ltx2_rope(rope_type: str = "split", heads: int = 3, head_dim: int = 8):
    return ltx2_transformer.LTX2AudioVideoRotaryPosEmbed(
        dim=heads * head_dim, rope_type=rope_type, num_attention_heads=heads
    )


def _ltx2_coords(batch: int = 2, tokens: int = 5) -> torch.Tensor:
    return 10 * torch.rand(batch, 3, tokens, 2)


def _ltx2_sp_shard(
    tables: tuple[torch.Tensor, torch.Tensor], start: int = 4, end: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    # A fresh dim-2 slice of each table, as the sequence-parallel hook cuts.
    return tuple(t[:, :, start:end, :] for t in tables)


def test_ltx2_split_rope_lane_mask_built_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cached lane mask is computed on the host and uploaded once: no
    # arange is created on the target device, where it would stay a deferred
    # graph until the first rotation reads it.
    arange_devices = []
    real_arange = torch.arange

    def recording_arange(*args, **kwargs):
        arange_devices.append(kwargs.get("device"))
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", recording_arange)
    omni_rope._first_half_lanes.cache_clear()
    device = torch.device("cpu")
    # A default device other than the host ("meta" stands in for TPU) must
    # not move the arange off the host.
    with torch.device("meta"):
        lanes = omni_rope._first_half_lanes(16, 8, device)
    assert arange_devices == ["cpu"]
    assert lanes.device == device
    assert lanes.tolist() == ([True] * 4 + [False] * 4) * 2
    assert omni_rope._first_half_lanes(16, 8, device) is lanes
    assert arange_devices == ["cpu"]
    omni_rope._first_half_lanes.cache_clear()


def test_ltx2_split_rope_tables_found_by_identity(
    ltx2, table_syncs, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every attention layer of a forward is handed the same (cos, sin)
    # objects: after their first rotation they are found by identity, with no
    # storage or layout query. A fresh slice of the same tables reads them.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    tables = rope(_ltx2_coords(batch=2, tokens=8), out_dtype=torch.bfloat16)
    cases = [
        (_ltx2_sp_shard(tables), torch.randn(2, 4, 24, dtype=torch.bfloat16)),
        (tables, torch.randn(2, 8, 24, dtype=torch.bfloat16)),  # not sharded
    ]
    wants = [ltx2.apply_split_rotary_emb(x, freqs, head_dim=8) for freqs, x in cases]
    syncs = len(table_syncs)

    def layout_query(self, *args, **kwargs):
        raise AssertionError("layout query")

    for name in ("is_contiguous", "storage_offset", "untyped_storage"):
        monkeypatch.setattr(torch.Tensor, name, layout_query)
    for (freqs, x), want in zip(cases, wants, strict=True):
        got = ltx2.apply_split_rotary_emb(x, freqs, head_dim=8)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    with pytest.raises(AssertionError, match="layout query"):
        ltx2.apply_split_rotary_emb(cases[0][1], _ltx2_sp_shard(tables), head_dim=8)
    assert len(table_syncs) == syncs


def test_ltx2_split_rope_identity_checks_width(
    ltx2, table_syncs, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The lanes cut for tensor parallelism depend on the width of q/k, so a
    # pair found by identity is reused only for the width it was cut for.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=4, head_dim=8)
    coords = _ltx2_coords()
    tables = rope(coords)
    cos, sin = _UPSTREAM_ROPE_FORWARD(rope, coords)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_rank", lambda: 1)
    x = torch.randn(2, 5, 2 * 8)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, (cos[:, 2:], sin[:, 2:]), head_dim=8)
    for _ in range(2):
        out = ltx2.apply_split_rotary_emb(x, tables, head_dim=8)
        torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)
    assert len(table_syncs) == 2  # the tables, then the lanes of rank 1

    x_full = torch.randn(2, 5, 4 * 8)
    out = ltx2.apply_split_rotary_emb(x_full, tables, head_dim=8)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x_full, (cos, sin), head_dim=8)
    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)
    assert len(table_syncs) == 2


def test_ltx2_split_rope_owned_tables_follow_new_tables(ltx2, table_syncs) -> None:
    # A new request builds new tables with the same slice layout. Their owned
    # copies replace the previous request's, whose tables are then released.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    x = torch.randn(2, 4, 24, dtype=torch.bfloat16)

    tables = rope(_ltx2_coords(batch=2, tokens=8), out_dtype=torch.bfloat16)
    old_table = weakref.ref(tables[0])
    ltx2.apply_split_rotary_emb(x, _ltx2_sp_shard(tables), head_dim=8)

    coords = _ltx2_coords(batch=2, tokens=8)
    tables = rope(coords, out_dtype=torch.bfloat16)
    syncs = len(table_syncs)
    out = ltx2.apply_split_rotary_emb(x, _ltx2_sp_shard(tables), head_dim=8)
    assert len(table_syncs) == syncs + 1
    upstream = _UPSTREAM_ROPE_FORWARD(rope, coords, None, torch.bfloat16)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, _ltx2_sp_shard(upstream), head_dim=8)
    torch.testing.assert_close(out, want.to(torch.bfloat16), rtol=0, atol=0)

    table_syncs.clear()  # the fixture's record holds the tables too
    gc.collect()
    assert old_table() is None


def test_ltx2_split_rope_rotates_views_of_unknown_storage(
    ltx2, table_syncs, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A storage without a data pointer cannot be told apart from another, so
    # its slices are rotated as they are, without an owned copy.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    coords = _ltx2_coords(batch=2, tokens=8)
    tables = rope(coords, out_dtype=torch.bfloat16)
    syncs = len(table_syncs)
    monkeypatch.setattr(torch.UntypedStorage, "data_ptr", lambda self: 0)
    x = torch.randn(2, 4, 24, dtype=torch.bfloat16)
    upstream = _UPSTREAM_ROPE_FORWARD(rope, coords, None, torch.bfloat16)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, _ltx2_sp_shard(upstream), head_dim=8)
    for _ in range(2):
        out = ltx2.apply_split_rotary_emb(x, _ltx2_sp_shard(tables), head_dim=8)
        torch.testing.assert_close(out, want.to(torch.bfloat16), rtol=0, atol=0)
    assert len(table_syncs) == syncs


@pytest.mark.parametrize(
    "inference", [False, True], ids=["grad_mode", "inference_mode"]
)
def test_ltx2_split_rope_layout_change_releases_tables(
    ltx2, table_syncs, inference: bool
) -> None:
    # A call with another token count (two-stage refinement, another
    # resolution) builds tables with a new slice layout. The previous tables,
    # the slices cut from them and their owned copies are released as soon as
    # the rope module replaces its tables, before any rotation with the new
    # ones. In inference mode a slice does not reference its base tensor, so
    # there the slice (which pins the storage) is the object to watch.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    bf16 = torch.bfloat16
    with torch.inference_mode(inference):
        tables = rope(_ltx2_coords(batch=2, tokens=8), out_dtype=bf16)
        shard = _ltx2_sp_shard(tables)
        ltx2.apply_split_rotary_emb(
            torch.randn(2, 4, 24, dtype=bf16), shard, head_dim=8
        )
        old = [weakref.ref(t) for t in (tables[0], shard[0], table_syncs[-1][0])]
        del tables, shard
        table_syncs.clear()  # the fixture's record holds the tables too

        coords = _ltx2_coords(batch=2, tokens=12)
        tables = rope(coords, out_dtype=bf16)
        table_syncs.clear()
        gc.collect()
        assert [ref() is None for ref in old] == [True, True, True]

        # The new layout rotates like upstream.
        x = torch.randn(2, 6, 24, dtype=bf16)
        out = ltx2.apply_split_rotary_emb(x, _ltx2_sp_shard(tables, 6, 12), head_dim=8)
        upstream = _UPSTREAM_ROPE_FORWARD(rope, coords, None, bf16)
        want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(
            x, _ltx2_sp_shard(upstream, 6, 12), head_dim=8
        )
        torch.testing.assert_close(out, want.to(bf16), rtol=0, atol=0)


def test_ltx2_split_rope_same_layout_tables_coexist(
    ltx2, table_syncs, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two tables with one slice layout: under tensor parallelism the audio
    # self-attention tables and the audio cross-attention tables have the
    # same shape, and every block rotates with one and then the other. Each
    # keeps its owned lanes; neither is copied again in later blocks.
    torch.manual_seed(0)
    ropes = [_ltx2_rope(heads=4, head_dim=8) for _ in range(2)]
    coords = [_ltx2_coords(batch=2, tokens=5) for _ in ropes]
    tables = [rope(c) for rope, c in zip(ropes, coords, strict=True)]
    assert tables[0][0].shape == tables[1][0].shape
    assert not torch.equal(tables[0][0], tables[1][0])
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_rank", lambda: 1)
    x = torch.randn(2, 5, 2 * 8)
    wants = []
    for rope, c in zip(ropes, coords, strict=True):
        cos, sin = _UPSTREAM_ROPE_FORWARD(rope, c)
        wants.append(
            _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, (cos[:, 2:], sin[:, 2:]), head_dim=8)
        )
    syncs = len(table_syncs)
    for _block in range(3):
        for freqs, want in zip(tables, wants, strict=True):
            out = ltx2.apply_split_rotary_emb(x, freqs, head_dim=8)
            torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)
    assert len(table_syncs) == syncs + 2  # one pair of owned lanes per table


@pytest.mark.parametrize("batch", [1, 2], ids=["b1", "b2"])
def test_ltx2_split_rope_shard_cache_reused_for_sp_views(
    ltx2, table_syncs, batch: int
) -> None:
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    coords = _ltx2_coords(batch=batch, tokens=8)
    cos_full, sin_full = rope(coords, out_dtype=torch.bfloat16)
    assert len(table_syncs) == 1

    # Unsharded full-sequence tables need no extra shard materialization.
    x_full = torch.randn(batch, 8, 3 * 8, dtype=torch.bfloat16)
    ltx2.apply_split_rotary_emb(x_full, (cos_full, sin_full), head_dim=8)
    assert len(table_syncs) == 1

    # Sequence-parallel slices (both rank 0 offset=0 and rank 1 offset>0)
    # materialize once into zero-offset contiguous buffers and reuse across blocks.
    upstream_cos, upstream_sin = _UPSTREAM_ROPE_FORWARD(
        rope, coords, None, torch.bfloat16
    )
    for rank, (start, end) in enumerate(((0, 4), (4, 8))):
        x_shard = torch.randn(batch, 4, 3 * 8, dtype=torch.bfloat16)
        freqs_view = (cos_full[:, :, start:end, :], sin_full[:, :, start:end, :])
        out1 = ltx2.apply_split_rotary_emb(x_shard, freqs_view, head_dim=8)
        assert len(table_syncs) == 2 + rank
        # The synced copies own a buffer of the slice's shape.
        for owned, view in zip(table_syncs[-1], freqs_view, strict=True):
            assert torch.equal(owned, view)
            assert owned.is_contiguous() and owned.storage_offset() == 0
            assert owned.untyped_storage().nbytes() == view.numel() * 2
        # Fresh view object over the same storage hits the shard cache across 48 blocks.
        out2 = ltx2.apply_split_rotary_emb(
            x_shard,
            (cos_full[:, :, start:end, :], sin_full[:, :, start:end, :]),
            head_dim=8,
        )
        assert len(table_syncs) == 2 + rank
        want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(
            x_shard,
            (upstream_cos[:, :, start:end, :], upstream_sin[:, :, start:end, :]),
            head_dim=8,
        ).to(torch.bfloat16)
        torch.testing.assert_close(out1, want, rtol=0, atol=0)
        torch.testing.assert_close(out2, want, rtol=0, atol=0)


def test_ltx2_split_rope_torch_compile_skips_lane_cache(ltx2, table_syncs) -> None:
    # Under torch.compile(fullgraph=True), split RoPE builds the lane mask
    # inline instead of caching a FakeTensor in _first_half_lanes, and skips
    # eager shard-cache materialization while matching upstream output exactly.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=3, head_dim=8)
    coords = _ltx2_coords(batch=2, tokens=8)
    tables = rope(coords, out_dtype=torch.bfloat16)
    cos_shard, sin_shard = _ltx2_sp_shard(tables)
    x = torch.randn(2, 4, 24, dtype=torch.bfloat16)
    upstream = _UPSTREAM_ROPE_FORWARD(rope, coords, None, torch.bfloat16)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, _ltx2_sp_shard(upstream), head_dim=8).to(
        torch.bfloat16
    )

    def compiled_rotate(
        t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return ltx2.apply_split_rotary_emb(t, (cos, sin), head_dim=8)

    omni_rope._first_half_lanes.cache_clear()
    syncs_before = len(table_syncs)
    try:
        fn = torch.compile(compiled_rotate, backend="eager", fullgraph=True)
        got = fn(x, cos_shard, sin_shard)
        assert omni_rope._first_half_lanes.cache_info().currsize == 0
        assert len(table_syncs) == syncs_before
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        omni_rope._first_half_lanes.cache_clear()


def _ltx2_sigma_harness(monkeypatch: pytest.MonkeyPatch, prepare):
    """Real predict_noise with a spy on the real _guide_modality."""
    monkeypatch.setattr(ltx2_denoise, "prepare_scheduler_stage", prepare)
    monkeypatch.setattr(
        ltx2_guidance.LTXGuidanceExecutor, "predict_noise", _UPSTREAM_PREDICT_NOISE
    )
    seen_calls: list[tuple[torch.Tensor, float | None]] = []

    def spy_guide(
        sample,
        splits,
        sigma,
        guidance,
        *,
        model_sigma=None,
        sigma_scalar=None,
        preserve_positive_velocity=False,
        rescale_token_count=None,
    ):
        seen_calls.append((sigma, sigma_scalar))
        return _UPSTREAM_GUIDE_MODALITY(
            sample,
            splits,
            sigma,
            guidance,
            model_sigma=model_sigma,
            sigma_scalar=sigma_scalar,
            preserve_positive_velocity=preserve_positive_velocity,
            rescale_token_count=rescale_token_count,
        )

    monkeypatch.setattr(
        ltx2_guidance.LTXGuidanceExecutor, "_guide_modality", staticmethod(spy_guide)
    )
    ltx2_patches._patch_ltx2_guidance_sigma_scalar()
    ltx2_patches._patch_ltx2_guidance_sigma_scalar()  # idempotent

    def _dummy_transformer(**kw):
        return kw["hidden_states"] * 0.5, kw["audio_hidden_states"] * 0.25

    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0])),
        _build_transformer_kwargs=lambda _fctx, _dctx, **kw: kw,
        _transformer_cache_context=lambda _name: contextlib.nullcontext(),
        transformer=_dummy_transformer,
        _video_guidance_model_sigma=lambda sigma, _dctx: sigma,
    )
    plan = ltx2_guidance.LTXGuidancePlan.build(
        ltx2_guidance.LTXGuidanceSpec(
            video=ltx2_guidance.LTXModalityGuidance(cfg_scale=3.0),
            audio=ltx2_guidance.LTXModalityGuidance(cfg_scale=3.0),
        )
    )
    prompt_ctx = SimpleNamespace(
        positive_connector_prompt_embeds=torch.ones(1, 2, 4),
        negative_connector_prompt_embeds=torch.zeros(1, 2, 4),
        positive_connector_audio_prompt_embeds=torch.ones(1, 2, 2),
        negative_connector_audio_prompt_embeds=torch.zeros(1, 2, 2),
    )
    state = SimpleNamespace(video=torch.randn(1, 4, 4), audio=torch.randn(1, 2, 2))

    def make_fctx(audio_scheduler, parallel=False):
        return SimpleNamespace(
            guidance_parallel_ready=parallel,
            prompt_context=prompt_ctx,
            attention_kwargs=None,
            audio_scheduler=audio_scheduler,
            original_audio_num_frames=None,
        )

    def predict(fctx, ts=500.0):
        seen_calls.clear()
        ltx2_guidance.LTXGuidanceExecutor().predict_noise(
            pipe, plan, 1, torch.tensor([ts]), state, fctx, None
        )
        assert len(seen_calls) == 2
        for sigma, scalar in seen_calls:
            assert scalar is not None
            assert scalar == sigma.to(torch.float32).item()
        return [scalar for _, scalar in seen_calls]

    return pipe, make_fctx, predict


def _no_tpu_attrs(obj: object) -> bool:
    return not [k for k in vars(obj) if k.startswith("_tpu_")]


def test_ltx2_guidance_sigma_scalar_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real predict_noise (and predict_parallel_guidance via
    # guidance_parallel_ready=True) passes state.video / state.audio and gets
    # the cached float32 sigma for both modalities.
    pipe, make_fctx, predict = _ltx2_sigma_harness(
        monkeypatch,
        lambda pipeline, req, *a, **kw: (
            SimpleNamespace(sigmas=torch.tensor([0.8, 0.4, 0.0])),
            None,
            torch.tensor([800.0, 400.0]),
        ),
    )
    aud_sched, _, _ = ltx2_denoise.prepare_scheduler_stage(pipe, None)
    assert _no_tpu_attrs(pipe.scheduler) and _no_tpu_attrs(aud_sched)
    assert predict(make_fctx(aud_sched)) == pytest.approx([0.5, 0.4])

    # Replacing scheduler.sigmas with a new tensor refreshes the scalars.
    pipe.scheduler.sigmas = torch.tensor([0.9, 0.25, 0.0])
    assert predict(make_fctx(aud_sched), ts=250.0) == pytest.approx([0.25, 0.4])

    # The CFG-parallel path receives the cached scalars too.
    monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        ltx2_guidance,
        "get_guidance_parallel_group",
        lambda: SimpleNamespace(
            all_gather=lambda val, separate_tensors=True: [val, val]
        ),
    )
    fctx_par = make_fctx(aud_sched, parallel=True)
    assert predict(fctx_par, ts=250.0) == pytest.approx([0.25, 0.4])


def test_ltx2_guidance_sigma_scalar_cache_never_serves_stale_sigmas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the cache was validated by storage address and stored on the
    # scheduler. Upstream prepare_scheduler_stage deep-copies the video
    # scheduler into audio_scheduler, and a new schedule can land at the old
    # address (simulated here by writing it into the same storage), so the
    # previous schedule's scalars were served.
    def prepare(pipeline, schedule, *a, **kw):
        audio_scheduler = copy.deepcopy(pipeline.scheduler)
        pipeline.scheduler.sigmas.copy_(schedule)
        audio_scheduler.sigmas = schedule.clone()
        return audio_scheduler, None, schedule[:-1] * 1000

    pipe, make_fctx, predict = _ltx2_sigma_harness(monkeypatch, prepare)
    for values in ([1.0, 0.5, 0.0], [0.9, 0.25, 0.0], [0.8, 0.125, 0.0]):
        aud_sched, _, _ = ltx2_denoise.prepare_scheduler_stage(
            pipe, torch.tensor(values)
        )
        assert _no_tpu_attrs(pipe.scheduler) and _no_tpu_attrs(aud_sched)
        assert predict(make_fctx(aud_sched)) == pytest.approx([values[1]] * 2)

    # An in-place schedule edit between stages is seen through the version
    # counter.
    pipe.scheduler.sigmas.copy_(torch.tensor([0.7, 0.375, 0.0]))
    assert predict(make_fctx(aud_sched)) == pytest.approx([0.375, 0.125])


def test_ltx2_async_guidance_step_avoids_blocking_sync(
    monkeypatch: pytest.MonkeyPatch, table_syncs: list[list[torch.Tensor]]
) -> None:
    from vllm_omni.diffusion.distributed import parallel_state

    monkeypatch.setattr(
        ltx2_runtime.LTXRuntime,
        "_synchronize_guidance_parallel_step_output",
        _UPSTREAM_SYNC_GUIDANCE_STEP,
    )
    blocking_syncs: list[bool] = []
    monkeypatch.setattr(
        ltx2_runtime.current_omni_platform,
        "synchronize",
        lambda *args, **kwargs: blocking_syncs.append(True),
    )
    ltx2_patches._patch_ltx2_async_guidance_step()
    patched = ltx2_runtime.LTXRuntime._synchronize_guidance_parallel_step_output
    assert getattr(patched, ltx2_patches._LTX2_ASYNC_STEP_PATCHED, False)
    ltx2_patches._patch_ltx2_async_guidance_step()
    assert ltx2_runtime.LTXRuntime._synchronize_guidance_parallel_step_output is patched

    v = torch.randn(4, 2).t()
    a = torch.randn(2, 2).t()
    assert not v.is_contiguous()

    # Single-rank CFG keeps the latents as-is but still flushes every step:
    # with cached sigmas nothing else cuts the lazy graph.
    monkeypatch.setattr(
        parallel_state, "get_classifier_free_guidance_world_size", lambda: 1
    )
    out_single = patched(None, (v, a), guidance_parallel_ready=True)
    assert out_single[0] is v and out_single[1] is a
    assert len(table_syncs) == 1
    assert table_syncs[0][0] is v and table_syncs[0][1] is a

    # A single-pass phase (guidance_parallel_ready=False, e.g. the two-stage
    # refine) under multi-rank CFG also flushes without copying.
    monkeypatch.setattr(
        parallel_state, "get_classifier_free_guidance_world_size", lambda: 2
    )
    out_not_ready = patched(None, (v, a), guidance_parallel_ready=False)
    assert out_not_ready[0] is v and out_not_ready[1] is a
    assert len(table_syncs) == 2
    assert table_syncs[1][0] is v and table_syncs[1][1] is a

    # Multi-rank CFG materializes contiguous latents and flushes non-blocking.
    out_multi = patched(None, (v, a), guidance_parallel_ready=True)
    assert out_multi[0].is_contiguous() and out_multi[1].is_contiguous()
    torch.testing.assert_close(out_multi[0], v)
    assert len(table_syncs) == 3 and len(table_syncs[2]) == 2
    assert table_syncs[2][0] is out_multi[0]

    # No path calls upstream's blocking device synchronize.
    assert blocking_syncs == []


def test_ltx2_fast_attention_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    monkeypatch.setattr(
        ltx2_transformer.LTX2Attention, "forward", _UPSTREAM_LTX2_ATTN_FORWARD
    )
    ltx2_patches._patch_ltx2_fast_attention_dispatch()
    ltx2_patches._patch_ltx2_fast_attention_dispatch()  # idempotent
    assert ltx2_transformer.LTX2Attention.forward is not _UPSTREAM_LTX2_ATTN_FORWARD

    def default_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        perturbation_mask=None,
    ):
        del attn, encoder_hidden_states, attention_mask
        del query_rotary_emb, key_rotary_emb
        out = hidden_states
        return out if perturbation_mask is None else out + perturbation_mask

    class _ExtraKwProc:
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            query_rotary_emb=None,
            key_rotary_emb=None,
            *,
            perturbation_mask=None,
            extra_accepted=None,
        ):
            out = default_call(
                None, attn, hidden_states, perturbation_mask=perturbation_mask
            )
            return out if extra_accepted is None else out + extra_accepted

    class _NoMaskProc:
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            query_rotary_emb=None,
            key_rotary_emb=None,
        ):
            return hidden_states * 2

    monkeypatch.setattr(
        ltx2_transformer.LTX2AudioVideoAttnProcessor, "__call__", default_call
    )
    dummy_attn = object.__new__(ltx2_transformer.LTX2Attention)
    torch.nn.Module.__init__(dummy_attn)
    x = torch.ones(2, 4, 8)
    pmask = torch.full((2, 1, 1), 2.0)

    def both(*args, **kwargs):
        got = dummy_attn(*args, **kwargs)
        want = _UPSTREAM_LTX2_ATTN_FORWARD(dummy_attn, *args, **kwargs)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        return got

    with patch.object(inspect, "signature", wraps=inspect.signature) as spy_sig:
        # The default processor's plain call never reflects.
        dummy_attn.processor = ltx2_transformer.LTX2AudioVideoAttnProcessor()
        spy_sig.reset_mock()
        dummy_attn(x, perturbation_mask=pmask)
        dummy_attn(x)
        assert spy_sig.call_count == 0
        torch.testing.assert_close(dummy_attn(x, perturbation_mask=pmask), x + 2.0)

        # Extra kwargs reflect once per processor class, then hit the cache.
        dummy_attn.processor = _ExtraKwProc()
        spy_sig.reset_mock()
        dummy_attn(x, perturbation_mask=pmask, extra_accepted=1.0, ignored_kw=9.0)
        out = dummy_attn(x, perturbation_mask=pmask, extra_accepted=1.0, ignored_kw=9.0)
        assert spy_sig.call_count == 1
        torch.testing.assert_close(out, x + 3.0)

    # A custom processor without perturbation_mask is not handed one (it is
    # filtered out with upstream's warning), matching upstream in every case.
    dummy_attn.processor = _NoMaskProc()
    both(x)
    both(x, perturbation_mask=pmask)
    dummy_attn.processor = _ExtraKwProc()
    both(x, perturbation_mask=pmask, extra_accepted=1.0, ignored_kw=9.0)

    # Distinct plain-function processors must not collide on type(processor)
    # in sig_cache.
    def _fn_proc_a(
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        *,
        alpha=0.0,
    ):
        del attn, encoder_hidden_states, attention_mask
        del query_rotary_emb, key_rotary_emb
        return hidden_states + alpha

    def _fn_proc_b(
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        *,
        beta=0.0,
    ):
        del attn, encoder_hidden_states, attention_mask
        del query_rotary_emb, key_rotary_emb
        return hidden_states + beta

    dummy_attn.processor = _fn_proc_a
    torch.testing.assert_close(dummy_attn(x, alpha=2.0, beta=5.0), x + 2.0)
    dummy_attn.processor = _fn_proc_b
    torch.testing.assert_close(dummy_attn(x, alpha=2.0, beta=5.0), x + 5.0)
