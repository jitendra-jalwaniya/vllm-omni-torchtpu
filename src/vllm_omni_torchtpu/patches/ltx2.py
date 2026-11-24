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
"""TPU runtime patches for vLLM-Omni LTX-2 diffusion models."""

from __future__ import annotations

import weakref
from typing import Any

import torch
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SPLIT_ROPE_PATCHED = "_tpu_split_rope_patched"
# Upper bounds on the split-RoPE rotation caches. A single LTX-2 request uses
# a small handful of distinct table slices (video, cross-attention, audio) and
# tensor objects. A rope module that replaces its tables releases their owned
# copies itself; clearing on overflow bounds HBM for tables that are never
# replaced (a dropped pipeline).
_MAX_OWNED_ROPE_LAYOUTS = 8
_MAX_ROPE_VIEW_ENTRIES = 16
_LTX2_PER_BATCH_TIMESTEP_PATCHED = "_tpu_ltx2_per_batch_timestep_patched"
_LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED = "_tpu_ltx2_guidance_sigma_scalar_patched"
# Live entries are one video and one audio schedule per stage; the bound only
# drops entries whose sigma tensors are gone.
_MAX_SIGMA_CACHE_ENTRIES = 8
_LTX2_ASYNC_STEP_PATCHED = "_tpu_ltx2_async_step_patched"


def apply_ltx2_patches() -> None:
    """Apply LTX-2-specific diffusion runtime patches."""
    _patch_ltx2_split_rope()
    _patch_ltx2_per_batch_timestep()
    _patch_ltx2_guidance_sigma_scalar()
    _patch_ltx2_async_guidance_step()


def _patch_ltx2_split_rope() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTX-2 split RoPE rotates q and
    # k in their (B, T, H * D) layout instead of reshaping them per head.
    #
    # Problem: LTX2AudioVideoRotaryPosEmbed builds split-RoPE tables as
    #   (B, H, T, D/2), so apply_split_rotary_emb transposes every (B, T, H * D)
    #   q and k to (B, H, T, D) and back in each attention layer. The pipeline
    #   passes the same coords tensor at every denoising step, and the rope
    #   modules rebuild the tables each time.
    # Fix: the rope module returns the tables in the layout of q and k,
    #   (B, 1, T, H * D) with the sign of each half folded into sin
    #   (split_rope_natural_tables), and apply_split_rotary_emb rotates them
    #   elementwise. Each rope module keeps the tables of its last split-RoPE
    #   call and returns them while the key matches, until a call with
    #   another key replaces them.
    # Effect: LTX-2 T2V 704x1280x121, 40 steps, on TPU v7x: 0.770 -> 0.482
    #   s/step (-37%).
    # Scope:
    #   - T stays at dim 2, where the split-RoPE sequence-parallel (SP) plan
    #     shards the tables.
    #   - TP: _slice_rope_for_tp slices 4-D tables by heads on dim 1 and so
    #     passes these (B, 1, T, H * D) tables through whole; the rotation
    #     takes the lanes of the local heads.
    #   - Cache key: the storage of coords (its data pointer; Tensor.is_set_to
    #     is not implemented for TPU), the layout of coords on it (storage
    #     offset, shape, stride), its dtype, device and version counter, and
    #     the device and out_dtype arguments. Fresh views with the same layout,
    #     such as the cross-attention ropes' coords[:, 0:1, :] each step, hit
    #     in and out of inference mode. The cache holds the coords view, which
    #     keeps the storage alive, so its address is not reused while cached.
    #     Inference tensors have no version counter, so in-place edits of
    #     coords in inference mode are not seen (vLLM-Omni does not edit
    #     coords after building them).
    #   - Interleaved RoPE and (B, H, T, D/2) tables use the original
    #     functions.
    #   - synchronize_tensors materializes the cached tables once.
    #   - SP hands each rank a dim-2 slice of the tables, and TP the lanes of
    #     its heads. torch_tpu passes a program the buffer behind each
    #     argument, so a rotation that reads a slice takes the full table as
    #     an input and slices it again, in every attention layer. The
    #     rotation reads fresh_buffer copies with a buffer of the slice's
    #     shape instead, built once per table and slice layout and released
    #     when the rope module replaces the tables; a slice whose storage has
    #     no data pointer is rotated as before.
    #   - The tables are not edited after the rope module builds them
    #     (vLLM-Omni only slices them): the rotation caches below are keyed
    #     on storage and layout, not on a version counter.
    #   - The torch.compiler.is_compiling() branches keep these caches and
    #     their storage queries out of a traced graph. No caller compiles the
    #     LTX-2 blocks today, so they are only reached by the unit tests.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer
    from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

    from vllm_omni_torchtpu.rope import (
        apply_split_rotary_half_swap,
        split_rope_natural_tables,
    )

    rope_cls = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed
    if getattr(rope_cls.forward, _SPLIT_ROPE_PATCHED, False):
        return
    orig_rope_forward = rope_cls.forward
    orig_apply_split_rotary_emb = ltx2_transformer.apply_split_rotary_emb

    # (cos, sin) the rotation is handed -> the tables it reads.
    #   - tables_by_view finds them by identity. The same tensor objects reach
    #     every attention layer of a forward, so the hot path makes no storage
    #     or layout query. An entry holds the pair by weak reference (a dead
    #     reference never matches a tensor that reuses its id), the q/k width
    #     and TP layout its lanes were cut for, and the owned tables, or None
    #     for a pair that is rotated as is (never one whose lanes were cut).
    #   - owned_tables holds one pair of owned tables per slice of a pair of
    #     tables: the key is the storage of cos and sin (data pointers) and
    #     the slice layout on it (storage offset, shape and stride of cos and
    #     sin, dtype, device). Each forward slices fresh views of the same
    #     tables, which hit here. Tables of the same shape (under TP the audio
    #     and audio cross-attention tables) have separate entries. An entry
    #     holds its slices, which keeps the storage alive, so a data pointer
    #     is not reused while it is a key.
    tables_by_view: dict[
        int,
        tuple[
            weakref.ref[torch.Tensor],
            weakref.ref[torch.Tensor],
            int,
            tuple[int, int] | None,
            tuple[torch.Tensor, torch.Tensor] | None,
        ],
    ] = {}
    owned_tables: dict[
        tuple[Any, ...],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}

    def _release_owned_tables(tables: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Drops the slices and owned copies cut from ``tables``."""
        storage = (
            tables[0].untyped_storage().data_ptr(),
            tables[1].untyped_storage().data_ptr(),
        )
        stale = [key for key in owned_tables if key[:2] == storage]
        if not stale:
            return
        for key in stale:
            del owned_tables[key]
        tables_by_view.clear()  # its entries hold the owned copies too

    def _cached_rope_forward(
        self: Any,
        coords: torch.Tensor,
        device: str | torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_type != "split":
            return orig_rope_forward(self, coords, device, out_dtype)
        storage = coords.untyped_storage().data_ptr()
        version = None if coords.is_inference() else coords._version
        key = (
            version,
            coords.storage_offset(),
            tuple(coords.shape),
            coords.stride(),
            coords.dtype,
            coords.device,
            device,
            out_dtype,
        )
        cached = getattr(self, "_tpu_split_rope_cache", None)
        if (
            cached is not None
            and storage != 0
            and cached[0].untyped_storage().data_ptr() == storage
            and cached[1] == key
        ):
            return cached[2]
        if cached is not None:
            # Release the replaced tables before the new ones are built: a
            # call with another token count (two-stage refinement, another
            # resolution) slices the new tables in another layout, which
            # would otherwise leave the old slices pinning the old tables.
            self._tpu_split_rope_cache = None
            _release_owned_tables(cached[2])
            del cached
        tables = orig_rope_forward(self, coords, device, out_dtype)
        if tables[0].ndim != 4:
            return tables
        tables = split_rope_natural_tables(*tables)
        synchronize_tensors(list(tables), wait=False)
        self._tpu_split_rope_cache = (coords, key, tables)
        return tables

    def _is_view(t: torch.Tensor) -> bool:
        return (
            not t.is_contiguous()
            or t.storage_offset() != 0
            or t.untyped_storage().nbytes() != t.numel() * t.element_size()
        )

    def _tp_layout() -> tuple[int, int]:
        return (
            ltx2_transformer.get_tensor_model_parallel_world_size(),
            ltx2_transformer.get_tensor_model_parallel_rank(),
        )

    def _local_lanes(
        cos: torch.Tensor, sin: torch.Tensor, width: int, tp_layout: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_size, tp_rank = tp_layout
        if cos.shape[-1] != width * tp_size:
            raise ValueError(
                f"LTX-2 split RoPE tables of width {cos.shape[-1]} do not "
                f"match q/k of width {width} on {tp_size} tensor-parallel "
                "ranks; the tables must cover all heads of all ranks."
            )
        start = tp_rank * width
        return cos[..., start : start + width], sin[..., start : start + width]

    def _rotation_tables(
        cos: torch.Tensor, sin: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tables for the rotation of ``width`` channels with ``(cos, sin)``."""
        entry = tables_by_view.get(id(cos))
        if (
            entry is not None
            and entry[0]() is cos
            and entry[1]() is sin
            and entry[2] == width
            and (entry[3] is None or entry[3] == _tp_layout())
        ):
            return entry[4] or (cos, sin)
        tp_layout = None if cos.shape[-1] == width else _tp_layout()
        local = (
            (cos, sin)
            if tp_layout is None
            else _local_lanes(cos, sin, width, tp_layout)
        )
        owned = None
        if _is_view(local[0]) or _is_view(local[1]):
            cos_ptr = local[0].untyped_storage().data_ptr()
            sin_ptr = local[1].untyped_storage().data_ptr()
            if cos_ptr == 0 or sin_ptr == 0:
                return local  # Unknown storage: rotate with the views.
            key = (
                cos_ptr,
                sin_ptr,
                local[0].storage_offset(),
                tuple(local[0].shape),
                local[0].stride(),
                local[1].storage_offset(),
                tuple(local[1].shape),
                local[1].stride(),
                local[0].dtype,
                local[0].device,
            )
            cached = owned_tables.get(key)
            if cached is None:
                owned = (fresh_buffer(local[0]), fresh_buffer(local[1]))
                synchronize_tensors(list(owned), wait=False)
                if len(owned_tables) >= _MAX_OWNED_ROPE_LAYOUTS:
                    owned_tables.clear()
                    tables_by_view.clear()
                # The entry keeps the views: their storage, and with it the
                # data pointers in the key, stays valid while it is cached.
                owned_tables[key] = (*local, *owned)
            else:
                owned = (cached[2], cached[3])
        if owned is None and tp_layout is not None:
            return local  # Lanes without an owned copy are cut again next time.
        if len(tables_by_view) >= _MAX_ROPE_VIEW_ENTRIES:
            tables_by_view.clear()
        tables_by_view[id(cos)] = (
            weakref.ref(cos),
            weakref.ref(sin),
            width,
            tp_layout,
            owned,
        )
        return owned or local

    def _apply_split_rotary_emb(
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        *,
        head_dim: int,
    ) -> torch.Tensor:
        cos, sin = freqs
        if cos.ndim != 4 or cos.shape[1] != 1 or cos.shape[-1] == head_dim // 2:
            return orig_apply_split_rotary_emb(x, freqs, head_dim=head_dim)
        width = x.shape[-1]
        if not torch.compiler.is_compiling():
            cos, sin = _rotation_tables(cos, sin, width)
        elif cos.shape[-1] != width:
            cos, sin = _local_lanes(cos, sin, width, _tp_layout())
        return apply_split_rotary_half_swap(x, cos, sin, head_dim)

    setattr(_cached_rope_forward, _SPLIT_ROPE_PATCHED, True)
    rope_cls.forward = _cached_rope_forward
    ltx2_transformer.apply_split_rotary_emb = _apply_split_rotary_emb
    logger.info("Applied TPU patch: cached LTX-2 split RoPE on the q/k layout.")


def _patch_ltx2_per_batch_timestep() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTXRuntime._denoise_timestep_kwargs
    # stops expanding unconditioned timesteps for sequence parallelism.
    #
    # Problem: In vllm-omni 0.32.0rc1
    #   (vllm_omni/diffusion/models/ltx2/ltx2_runtime.py::
    #   LTXRuntime._denoise_timestep_kwargs), the method discards forward_ctx
    #   and denoise_ctx and calls guidance_executor.timestep_kwargs with
    #   expand_for_sequence_parallel=True. That expands each sample's scalar
    #   timestep to (B, T_video) and (B, T_audio), causing time_embed and every
    #   transformer block to project and materialize per-token AdaLN modulation
    #   tables even though all tokens share the same timestep.
    # Fix: Replace LTXRuntime._denoise_timestep_kwargs (copied from vllm-omni
    #   0.32.0rc1, since wrapping cannot override the hardcoded keyword
    #   argument) to pass expand_for_sequence_parallel=False. Unconditioned
    #   timesteps stay 1-D (B,), keeping temb at (B, 1, C) so it broadcasts
    #   into each block's norm modulation.
    # Scope:
    #   - LTX2VideoTransformer3DModel._build_sp_plan declares
    #     SequenceParallelInput(split_dim=1, expected_dims=2, split_output=False)
    #     on timestep, so 1-D (B,) tensors skip sequence sharding and stay
    #     replicated across SP ranks. They take the dimension-mismatch path of
    #     SequenceParallelSplitHook._prepare_sp_input
    #     (vllm_omni/diffusion/hooks/sequence_parallel.py), which logs
    #     "Expected tensor with 2 dims, got 1. Skipping split." through
    #     warning_once: under SP every rank prints it once for T2V, and it is
    #     expected.
    #   - Conditioned I2V requests with a conditioning_mask still override
    #     timestep with the 2-D (B, T_video) masked tensor in
    #     LTXI2VConditioningMixin._denoise_timestep_kwargs (which is sharded
    #     by the SP plan), while keeping audio_timestep 1-D (B,).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

    if getattr(
        LTXRuntime._denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        False,
    ):
        return

    def _per_batch_denoise_timestep_kwargs(
        self: Any,
        ts: torch.Tensor,
        forward_ctx: Any,
        denoise_ctx: Any,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return self.guidance_executor.timestep_kwargs(
            ts,
            video_token_count=video_token_count,
            audio_token_count=audio_token_count,
            expand_for_sequence_parallel=False,
        )

    setattr(
        _per_batch_denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        True,
    )
    LTXRuntime._denoise_timestep_kwargs = _per_batch_denoise_timestep_kwargs
    logger.info("Applied TPU patch: per-batch LTX-2 unconditioned timestep.")


def _patch_ltx2_guidance_sigma_scalar() -> None:
    """Pre-materialize LTX-2 scheduler sigmas on CPU to avoid per-step D2H syncs.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_guidance.py``
    (``LTXGuidanceExecutor.predict_noise``, ``predict_parallel_guidance``, and
    ``_guide_modality``) and ``ltx2_denoise.py`` (``prepare_scheduler_stage``).

    Upstream ``LTXGuidanceExecutor.predict_noise`` (and
    ``predict_parallel_guidance``, which ``predict_noise`` dispatches to when
    ``forward_ctx.guidance_parallel_ready`` is True) calls
    ``self._guide_modality(..., sigma=scheduler.sigmas[index])`` twice per
    denoise step (video and audio) without passing ``sigma_scalar``, so
    ``velocity_from_x0`` executes ``sigma.to(torch.float32).item()`` on a TPU
    tensor twice every step. In TorchTPU lazy execution mode, each ``.item()``
    forces a blocking ``TransferFromDevice`` that drains the in-flight
    execution queue and starves TensorCore between steps. Caching
    ``scheduler.sigmas`` as CPU ``float32`` scalars once per stage and
    supplying ``sigma_scalar`` eliminates all per-step D2H host syncs while
    preserving the exact IEEE-754 ``float32`` scalar value.

    The cache lives in this closure, keyed by the sigma tensor's identity (a
    weakref) and version counter, never by storage address and never on the
    scheduler: ``prepare_scheduler_stage`` deep-copies ``pipeline.scheduler``
    into ``audio_scheduler``, and a freed schedule's address can be reused by
    the next one. Each scheduler stage also refreshes it unconditionally.

    Limitations: ``combine_cfg_velocity`` (``ltx2_guidance.py``) still calls
    ``.item()`` per call. It is reached only through the generic
    ``combine_cfg_noise`` CFG-parallel hook, not through ``predict_noise``.
    Removing these reads also removes the per-step graph cut they made; the
    step flush in ``_patch_ltx2_async_guidance_step`` replaces it.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni pre-materializes scheduler sigmas on CPU and passes
    ``sigma_scalar`` from ``LTXGuidanceExecutor.predict_noise`` /
    ``predict_parallel_guidance`` into ``_guide_modality``.
    """
    from vllm_omni.diffusion.models.ltx2 import ltx2_denoise, ltx2_guidance

    executor_cls = ltx2_guidance.LTXGuidanceExecutor
    if getattr(
        executor_cls.predict_noise,
        _LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED,
        False,
    ):
        return

    orig_prepare_scheduler_stage = ltx2_denoise.prepare_scheduler_stage
    orig_predict_noise = executor_cls.predict_noise
    orig_guide_modality = executor_cls._guide_modality
    active_step_sigma_map: dict[int, float | None] = {}
    # id(sigmas) -> (weakref to sigmas, version counter or None, scalars).
    sigma_cache: dict[int, tuple[Any, int | None, tuple[float, ...]]] = {}

    def _ensure_sigma_scalars(
        scheduler: Any, *, refresh: bool = False
    ) -> tuple[float, ...] | None:
        sigmas_t = getattr(scheduler, "sigmas", None)
        if (
            sigmas_t is None
            or not isinstance(sigmas_t, torch.Tensor)
            or sigmas_t.ndim != 1
        ):
            return None
        key = id(sigmas_t)
        # Inference tensors have no version counter; the per-stage refresh
        # covers schedules rebuilt in inference mode.
        version = None if sigmas_t.is_inference() else sigmas_t._version
        entry = sigma_cache.get(key)
        if (
            not refresh
            and entry is not None
            and entry[0]() is sigmas_t
            and entry[1] == version
        ):
            return entry[2]
        scalars = tuple(
            float(x)
            for x in sigmas_t.detach().to(device="cpu", dtype=torch.float32).tolist()
        )
        if len(sigma_cache) >= _MAX_SIGMA_CACHE_ENTRIES:
            for dead in [k for k, e in sigma_cache.items() if e[0]() is None]:
                del sigma_cache[dead]
            if len(sigma_cache) >= _MAX_SIGMA_CACHE_ENTRIES:
                sigma_cache.clear()
        sigma_cache[key] = (weakref.ref(sigmas_t), version, scalars)
        return scalars

    def _cached_prepare_scheduler_stage(
        pipeline: Any,
        request_inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, torch.Tensor]:
        audio_scheduler, video_audio_step_adapter, timesteps_tensor = (
            orig_prepare_scheduler_stage(pipeline, request_inputs, *args, **kwargs)
        )
        _ensure_sigma_scalars(getattr(pipeline, "scheduler", None), refresh=True)
        _ensure_sigma_scalars(audio_scheduler, refresh=True)
        return audio_scheduler, video_audio_step_adapter, timesteps_tensor

    def _predict_noise_with_cached_sigma(
        self: Any,
        pipeline: Any,
        plan: Any,
        index: int,
        timestep: torch.Tensor,
        state: Any,
        forward_ctx: Any,
        denoise_ctx: Any,
        preserve_positive_velocity: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_scalars = _ensure_sigma_scalars(getattr(pipeline, "scheduler", None))
        a_scalars = _ensure_sigma_scalars(getattr(forward_ctx, "audio_scheduler", None))
        active_step_sigma_map[id(state.video)] = (
            v_scalars[index]
            if v_scalars is not None and 0 <= index < len(v_scalars)
            else None
        )
        active_step_sigma_map[id(state.audio)] = (
            a_scalars[index]
            if a_scalars is not None and 0 <= index < len(a_scalars)
            else None
        )
        try:
            return orig_predict_noise(
                self,
                pipeline,
                plan,
                index,
                timestep,
                state,
                forward_ctx,
                denoise_ctx,
                preserve_positive_velocity=preserve_positive_velocity,
            )
        finally:
            active_step_sigma_map.clear()

    @staticmethod
    def _guide_modality_with_cached_sigma(
        sample: torch.Tensor,
        splits: dict[str, torch.Tensor],
        sigma: torch.Tensor,
        guidance: Any,
        *,
        model_sigma: torch.Tensor | None = None,
        sigma_scalar: float | None = None,
        preserve_positive_velocity: bool = False,
        rescale_token_count: int | None = None,
    ) -> torch.Tensor:
        if sigma_scalar is None:
            sigma_scalar = active_step_sigma_map.get(id(sample))
        return orig_guide_modality(
            sample,
            splits,
            sigma,
            guidance,
            model_sigma=model_sigma,
            sigma_scalar=sigma_scalar,
            preserve_positive_velocity=preserve_positive_velocity,
            rescale_token_count=rescale_token_count,
        )

    setattr(
        _predict_noise_with_cached_sigma,
        _LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED,
        True,
    )
    ltx2_denoise.prepare_scheduler_stage = _cached_prepare_scheduler_stage
    executor_cls.predict_noise = _predict_noise_with_cached_sigma
    executor_cls._guide_modality = _guide_modality_with_cached_sigma
    logger.info(
        "Applied TPU patch: pre-materialized LTX-2 guidance sigma scalars on CPU."
    )


def _patch_ltx2_async_guidance_step() -> None:
    """Replace CUDA-only per-step device synchronize() with non-blocking flush.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_runtime.py``
    (``LTXRuntime._synchronize_guidance_parallel_step_output``).

    Upstream materializes contiguous latents and calls
    ``current_omni_platform.synchronize()`` at the end of every CFG-parallel
    denoise step to avoid CUDA async numerical drift. On TPU, that blocking
    sync stalls the host between steps and cannot be wrapped without invoking
    the sync itself, so this patch replaces the 5-line method body to keep the
    same CFG-world-size guard and ``.contiguous()`` materialization while
    flushing via ``synchronize_tensors(list(latents), wait=False)``.

    Every other step (``cfg_parallel_size == 1``, or a single-pass phase such
    as the two-stage refine) is flushed too, without the copy. Upstream skips
    the sync there, but its ``velocity_from_x0`` ``.item()`` cut the lazy graph
    every step; ``_patch_ltx2_guidance_sigma_scalar`` removes that read, so
    without this flush the whole loop and the decoder would trace into one
    TPU program. This runs in the host denoise loop, never under
    ``torch.compile``.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni gates the blocking ``synchronize()`` in
    ``LTXRuntime._synchronize_guidance_parallel_step_output`` to CUDA (or
    uses a non-blocking flush on TPU).
    """
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime
    from vllm_torchtpu.utils import synchronize_tensors

    if getattr(
        LTXRuntime._synchronize_guidance_parallel_step_output,
        _LTX2_ASYNC_STEP_PATCHED,
        False,
    ):
        return

    def _async_guidance_step_output(
        self: Any,
        latents: tuple[torch.Tensor, torch.Tensor],
        guidance_parallel_ready: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self
        if (
            guidance_parallel_ready
            and parallel_state.get_classifier_free_guidance_world_size() > 1
        ):
            latents = tuple(tensor.contiguous() for tensor in latents)
        synchronize_tensors(list(latents), wait=False)
        return latents

    setattr(_async_guidance_step_output, _LTX2_ASYNC_STEP_PATCHED, True)
    LTXRuntime._synchronize_guidance_parallel_step_output = _async_guidance_step_output
    logger.info(
        "Applied TPU patch: non-blocking async guidance-parallel step boundary."
    )
