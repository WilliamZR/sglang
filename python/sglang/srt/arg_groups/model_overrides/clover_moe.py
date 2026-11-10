"""Config-time override declarations for Clover."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.utils.common import get_device_sm

logger = logging.getLogger(__name__)


@_register_for("CloverMoeModelLM")
def _clover_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if (
        cfg.speculative_algorithm is None
        or cfg.speculative_algorithm.upper() != "CLOVER"
    ):
        raise ValueError("CloverMoeModelLM requires --speculative-algorithm clover")
    if cfg.dllm_algorithm is not None:
        raise ValueError(
            "Clover does not support --dllm-algorithm; use --speculative-algorithm clover"
        )
    if cfg.device != "cuda" or cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError(
            "Clover preview kernels require CUDA; serving dtypes other than BF16 are unvalidated"
        )
    if get_device_sm() // 10 not in (9, 10):
        raise ValueError(
            "Clover preview kernels require compute capability 9.x or 10.x"
        )
    if (
        cfg.linear_attn_backend not in ("triton", "flashinfer")
        or cfg.linear_attn_prefill_backend not in (None, "triton", "flashinfer")
        or cfg.linear_attn_decode_backend not in (None, "triton", "flashinfer")
    ):
        raise ValueError(
            "Clover GDN backends other than Triton and FlashInfer are unvalidated"
        )
    if any(
        backend not in (None, "flashinfer")
        for backend in (
            cfg.attention_backend,
            cfg.prefill_attention_backend,
            cfg.decode_attention_backend,
        )
    ):
        raise ValueError(
            "Clover full-attention backends other than FlashInfer are unvalidated"
        )
    if cfg.dcp_size != 1 or cfg.attn_cp_size != 1 or cfg.attn_dp_size != 1:
        raise ValueError(
            "Clover with DCP or attention DP/CP greater than 1 is unvalidated"
        )
    if cfg.pp_size != 1:
        raise ValueError("Clover requires pipeline parallel size 1")
    # A2A backends can expand EP during later argument resolution.
    use_deepep = cfg.moe_a2a_backend == "deepep"
    if cfg.moe_a2a_backend not in ("none", "deepep"):
        raise ValueError("Clover requires --moe-a2a-backend none or deepep")
    if use_deepep and cfg.moe_runner_backend not in ("auto", "deep_gemm"):
        raise ValueError("Clover DeepEP requires --moe-runner-backend deep_gemm")
    # Auto dispatches prefill in normal mode and the preview and verify steps,
    # which run as decode batches, in low-latency mode; forcing low latency would
    # send whole prompts through its per-rank token cap.
    if use_deepep and cfg.deepep_mode not in ("auto", "normal"):
        raise ValueError("Clover DeepEP requires --deepep-mode auto or normal")
    # The expert-distribution recorder has no gatherer for auto DeepEP mode.
    if (
        use_deepep
        and cfg.deepep_mode == "auto"
        and (cfg.enable_eplb or cfg.expert_distribution_recorder_mode is not None)
    ):
        raise ValueError(
            "Clover EPLB or expert recording with DeepEP requires --deepep-mode normal"
        )
    if (
        not use_deepep
        and cfg.ep_size > 1
        and cfg.moe_runner_backend not in ("auto", "triton")
    ):
        raise ValueError("Clover EP requires --moe-runner-backend triton")
    if cfg.quantization is not None:
        raise ValueError("Clover supports only unquantized models")
    if cfg.disaggregation_mode != "null" or cfg.enable_hierarchical_cache:
        raise ValueError("Clover with disaggregation or HiCache is unvalidated")
    strategy = "no_buffer" if cfg.disable_radix_cache else "extra_buffer"
    if cfg.mamba_radix_cache_strategy not in ("auto", strategy):
        logger.warning(
            "Clover requires mamba_radix_cache_strategy=%s, overriding %s.",
            strategy,
            cfg.mamba_radix_cache_strategy,
        )
    overrides = {
        "attention_backend": "flashinfer",
        "mamba_radix_cache_strategy": strategy,
    }
    if use_deepep:
        overrides["moe_runner_backend"] = "deep_gemm"
    elif cfg.ep_size > 1:
        overrides["moe_runner_backend"] = "triton"
    return overrides
