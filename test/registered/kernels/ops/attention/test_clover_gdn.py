"""Clover GDN preview and native chunk checkpoint numerical contracts."""

import math

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("heads", [(2, 4), (16, 32)])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
def test_preview_kernel_parity_readonly_and_graph(block_size, heads, activation_dtype):
    from sglang.kernels.ops.attention.clover_gdn_triton import (
        clover_local_memory_triton,
    )

    torch.manual_seed(123)
    q_heads, v_heads = heads
    q = torch.randn(
        1, block_size * 2, q_heads, 128, device="cuda", dtype=activation_dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, block_size * 2, v_heads, 128, device="cuda", dtype=activation_dtype
    )
    g = -torch.rand(1, block_size * 2, v_heads, device="cuda")
    state = torch.randn(3, v_heads, 128, 128, device="cuda") * 0.05
    indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    kw = dict(
        clean_ssm_states=state,
        cache_indices=indices,
        active_valid_mask=torch.ones(block_size * 2, device="cuda", dtype=torch.bool),
        scale=128**-0.5,
        softmax_logit_scale=torch.full((v_heads,), 10.0, device="cuda"),
        softmax_history_bias=torch.full((v_heads,), 5.0, device="cuda"),
        softmax_history_c=torch.ones(v_heads, device="cuda"),
    )
    kw["active_valid_mask"][block_size:] = False
    before = state.clone()
    # Only the first row is live. Combine its local softmax with the history bucket.
    qf = q[0, :block_size].float().repeat_interleave(v_heads // q_heads, dim=1)
    kf = k[0, :block_size].float().repeat_interleave(v_heads // q_heads, dim=1)
    qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    local_logits = torch.einsum("lhk,shk->hls", qf, kf) * 10
    history_logits = torch.full_like(
        local_logits[..., :1], 5 + math.log(block_size / 32)
    )
    weights = torch.cat((local_logits, history_logits), dim=-1).softmax(-1)
    local = torch.einsum("hls,shv->lhv", weights[..., :-1], v[0, :block_size].float())
    history = torch.einsum("lhk,hvk->lhv", qf * 128**-0.5, state[1])
    history = history * g[0, :block_size].exp()[..., None]
    expected = torch.zeros_like(v)
    expected[0, :block_size] = (local + weights[..., -1].T[..., None] * history).to(
        v.dtype
    )
    actual = clover_local_memory_triton(q, k, v, g, **kw)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.04, rtol=0.04)
    # Reduced-precision and non-contiguous scalar views must match FP32 inputs.
    for dtype in (torch.bfloat16, torch.float16):
        for name in (
            "softmax_logit_scale",
            "softmax_history_bias",
            "softmax_history_c",
        ):
            kw[name] = torch.randn(2 * v_heads, device="cuda", dtype=dtype)[::2]
        fp32_kw = {
            name: value.float() if name.startswith("softmax_") else value
            for name, value in kw.items()
        }
        torch.testing.assert_close(
            clover_local_memory_triton(q, k, v, g, **kw),
            clover_local_memory_triton(q, k, v, g, **fp32_kw),
            rtol=0,
            atol=0,
        )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = clover_local_memory_triton(q, k, v, g, **kw)
    q.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, clover_local_memory_triton(q, k, v, g, **kw))
    torch.testing.assert_close(state, before, atol=0, rtol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
