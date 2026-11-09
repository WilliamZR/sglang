"""Clover's preview kernels admit validated CUDA targets without widening SM8x."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.clover_gdn_triton import (
    supports_clover_local_memory_triton,
)
from sglang.kernels.ops.mamba.clover_conv_triton import (
    supports_clover_local_memory_conv,
)
from sglang.kernels.ops.moe.clover_coreset_triton import (
    supports_clover_block_coreset_triton,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _supported_shapes(is_cuda=True):
    common = dict(is_cuda=is_cuda, dtype=torch.bfloat16, device=torch.device("cuda"))
    query = SimpleNamespace(**common, ndim=4, shape=(1, 32, 16, 128))
    value = SimpleNamespace(**common, ndim=4, shape=(1, 32, 32, 128))
    mixed = SimpleNamespace(**common, ndim=2, shape=(32, 8192))
    router = SimpleNamespace(**common, ndim=2, shape=(32, 256))
    weights = SimpleNamespace(numel=lambda: 8192 * 4)
    return (
        supports_clover_local_memory_triton(query, query, value, block_size=32),
        supports_clover_local_memory_conv(
            mixed, conv_weights=weights, block_size=32, activation="silu"
        ),
        supports_clover_block_coreset_triton(router, top_k=8, block_size=32),
    )


@pytest.mark.parametrize(
    "capability,supported",
    [
        ((7, 5), False),
        ((8, 0), False),
        ((8, 6), True),
        ((8, 9), False),
        ((9, 0), True),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 0), False),
    ],
)
def test_preview_capability_boundary(monkeypatch, capability, supported):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert _supported_shapes() == (supported,) * 3


def test_cpu_input_does_not_query_cuda(monkeypatch):
    def unexpected_query(device):
        raise AssertionError("CPU inputs must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    assert _supported_shapes(is_cuda=False) == (False,) * 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
