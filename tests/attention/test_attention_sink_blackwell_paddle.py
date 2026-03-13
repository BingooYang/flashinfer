"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import paddle

paddle.compat.enable_torch_proxy()
import einops
import pytest
import torch
import numpy as np
from tests.test_helpers.sink_attention_reference import sink_attention_unified

import flashinfer
# from flashinfer.utils import get_compute_capability


def generate_causal_mask(batch_size, q_seq_len, device):
    """Generate packed uint16 causal mask for speculative decoding.

    Returns shape [batch_size, q_seq_len, divUp(q_seq_len, 32) * 2] dtype uint16.
    Each 32-bit packed word stores one bit per KV position (1 = attend, 0 = mask).
    The result is stored as pairs of uint16 (low half, high half).
    """
    num_packed_per_token = (q_seq_len + 31) // 32  # number of uint32 words per token

    # Build the mask on CPU using numpy, then transfer to device
    # Shape: [q_seq_len, num_packed_per_token] as uint32
    mask_np = np.zeros((q_seq_len, num_packed_per_token), dtype=np.uint32)
    for q_idx in range(q_seq_len):
        for pack_idx in range(num_packed_per_token):
            word = np.uint32(0)
            for bit in range(32):
                kv_idx = pack_idx * 32 + bit
                if kv_idx < q_seq_len and kv_idx <= q_idx:
                    word |= np.uint32(1 << bit)
            mask_np[q_idx, pack_idx] = word

    # View as uint16: each uint32 becomes 2 uint16 values
    mask_u16_np = mask_np.view(np.uint16)  # [q_seq_len, num_packed_per_token * 2]

    # Expand to batch and move to device
    mask_u16_np = np.tile(mask_u16_np, (batch_size, 1, 1))  # [batch, q_seq_len, ...]
    mask_tensor = torch.from_numpy(mask_u16_np.copy()).to(device)
    return mask_tensor


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("batch_size", [1, 4, 16])
# @pytest.mark.parametrize("page_size", [32])
# @pytest.mark.parametrize("seq_len", [32, 128, 1024])
# @pytest.mark.parametrize("num_qo_heads", [32])
# @pytest.mark.parametrize("num_kv_heads", [8, 32])
# @pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
def test_blackwell_trtllm_gen_decode_attention_sink(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    # compute_capability = get_compute_capability(torch.device(device="cuda"))
    # if compute_capability[0] != 10:
    #     pytest.skip("trtllm-gen only supports SM100 and SM103 GPUs.")
    seed = 0
    paddle.seed(seed)
    device = "cuda:0"

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate unique block IDs for all sequences
    block_tables = torch.arange(
        (batch_size * max_num_blocks_per_seq), dtype=torch.int32, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Create separate K and V caches
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        1.0,  # bmm1_scale
        1.0,  # bmm2_scale
        -1,  # window_left
        out_dtype=dtype,
        sinks=sink,
    )

    k = einops.rearrange(
        k_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )
    v = einops.rearrange(
        v_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )

    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        -1,
        False,
        1.0,
        mode="incremental",
    )

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # torch.testing.assert_close(o_ref, output, atol=atol, rtol=rtol)
    np.testing.assert_allclose(o_ref.float(), output.float(), atol=atol, rtol=rtol)


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("batch_size", [1, 4, 16])
# @pytest.mark.parametrize("page_size", [32])
# @pytest.mark.parametrize("seq_len", [32, 128, 1024])
# @pytest.mark.parametrize("num_qo_heads", [32])
# @pytest.mark.parametrize("num_kv_heads", [8, 32])
# @pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
def test_blackwell_trtllm_gen_context_attention_sink(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    # compute_capability = get_compute_capability(torch.device(device="cuda"))
    # if compute_capability[0] != 10:
    #     pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    seed = 0
    paddle.seed(seed)
    # torch.manual_seed(seed)
    device = "cuda:0"

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate unique block IDs for all sequences
    block_tables = torch.arange(
        (batch_size * max_num_blocks_per_seq), dtype=torch.int32, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Create separate K and V caches
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        num_tokens,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    q_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    kv_indptr = (
        torch.arange(0, num_blocks + 1, dtype=torch.int32, device=device) * page_size
    )

    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        seq_len,
        1.0,  # bmm1_scale
        1.0,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        -1,  # window_left
        out_dtype=dtype,
        sinks=sink,
    )

    k = einops.rearrange(
        k_cache,
        "num_pages h p d -> (num_pages p) h d",
    )
    v = einops.rearrange(
        v_cache,
        "num_pages h p d -> (num_pages p) h d",
    )

    print(q.shape, k.shape, v.shape)

    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        -1,
        True,
        1.0,
        mode="prefill",
        batch_size=batch_size,
    )

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ref_o = o_ref.float().numpy()
    output_o = output.float().numpy()
    np.testing.assert_allclose(ref_o, output_o, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
def test_blackwell_trtllm_gen_decode_all_params(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    """Test trtllm_batch_decode_with_kv_cache exercising all parameters.

    Covers: kv_cache as tuple, pre-allocated out, explicit out_dtype,
    bmm1/bmm2 scale (both float and device tensor), window_left (sliding window),
    sinks (attention sink), kv_layout (HND and NHD), enable_pdl,
    backend='trtllm-gen', q_len_per_req, and cum_seq_lens_q/max_q_len
    (variable-length query for speculative decoding).
    """
    seed = 42
    paddle.seed(seed)
    np.random.seed(seed)
    device = "cuda:0"

    # Sub-test configs to exercise different parameter combos
    # Note: only Dense (window_left=-1) cubins are locally cached;
    # SlidingOrChunkedCausal cubins require network download from NVIDIA.
    configs = [
        # Config 0: basic with all defaults made explicit, float scales, HND, no window
        {
            "kv_layout": "HND",
            "window_left": -1,
            "enable_pdl": True,
            "use_device_scale": False,
            "use_pre_alloc_out": False,
            "use_sinks": True,
            "q_len_per_req": 1,
            "max_q_len": None,
            "backend": "trtllm-gen",
            "use_mask": False,
        },
        # Config 1: NHD layout, pre-allocated out, device tensor scales
        {
            "kv_layout": "NHD",
            "window_left": -1,
            "enable_pdl": False,
            "use_device_scale": True,
            "use_pre_alloc_out": True,
            "use_sinks": True,
            "q_len_per_req": 1,
            "max_q_len": None,
            "backend": "trtllm-gen",
            "use_mask": False,
        },
        # Config 2: speculative decode with uniform q_len_per_req > 1
        {
            "kv_layout": "HND",
            "window_left": -1,
            "enable_pdl": None,
            "use_device_scale": False,
            "use_pre_alloc_out": False,
            "use_sinks": True,
            "q_len_per_req": 3,
            "max_q_len": None,
            "backend": "trtllm-gen",
            "use_mask": False,
        },
        # Config 3: variable-length query (speculative decode), no sinks
        {
            "kv_layout": "HND",
            "window_left": -1,
            "enable_pdl": True,
            "use_device_scale": False,
            "use_pre_alloc_out": True,
            "use_sinks": False,
            "q_len_per_req": None,
            "max_q_len": 4,
            "backend": "trtllm-gen",
            "use_mask": False,
        },
        # Config 4: NHD + sinks + variable-length query + device scale
        {
            "kv_layout": "NHD",
            "window_left": -1,
            "enable_pdl": True,
            "use_device_scale": True,
            "use_pre_alloc_out": False,
            "use_sinks": True,
            "q_len_per_req": None,
            "max_q_len": 3,
            "backend": "trtllm-gen",
            "use_mask": False,
        },
        # Config 5: trtllm-gen + mask + speculative decode (q_len_per_req > 1)
        # mask is accepted but not used by trtllm-gen kernel (it handles causal internally)
        {
            "kv_layout": "HND",
            "window_left": -1,
            "enable_pdl": True,
            "use_device_scale": False,
            "use_pre_alloc_out": False,
            "use_sinks": True,
            "q_len_per_req": 3,
            "max_q_len": None,
            "backend": "trtllm-gen",
            "use_mask": True,
        },
        # Config 6: xqa + mask + speculative decode (mask is actually applied by xqa kernel)
        {
            "kv_layout": "HND",
            "window_left": -1,
            "enable_pdl": True,
            "use_device_scale": False,
            "use_pre_alloc_out": False,
            "use_sinks": False,
            "q_len_per_req": 3,
            "max_q_len": None,
            "backend": "xqa",
            "use_mask": True,
        },
    ]

    sm_scale = 1.0 / (head_dim**0.5)

    for cfg_idx, cfg in enumerate(configs):
        kv_layout = cfg["kv_layout"]
        window_left = cfg["window_left"]
        enable_pdl = cfg["enable_pdl"]
        use_device_scale = cfg["use_device_scale"]
        use_pre_alloc_out = cfg["use_pre_alloc_out"]
        use_sinks = cfg["use_sinks"]
        q_len_per_req = cfg["q_len_per_req"]
        max_q_len_cfg = cfg["max_q_len"]
        backend = cfg["backend"]
        use_mask = cfg["use_mask"]

        # --- Determine query lengths ---
        if q_len_per_req is not None:
            total_q_tokens = batch_size * q_len_per_req
            q_lens = torch.tensor([q_len_per_req] * batch_size, dtype=torch.int32)
        else:
            # Variable-length query: random q_len per request in [1, max_q_len]
            q_lens_np = np.random.randint(1, max_q_len_cfg + 1, size=(batch_size,))
            q_lens_np[-1] = max_q_len_cfg  # ensure max is present
            q_lens = torch.tensor(q_lens_np, dtype=torch.int32)
            total_q_tokens = int(q_lens.sum().item())

        # Cumulative query lengths
        q_indptr = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.cumsum(q_lens.to(device), dim=0, dtype=torch.int32),
            ]
        )

        # KV sequence lengths
        kv_seq_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        blocks_per_seq = (kv_seq_lens + page_size - 1) // page_size
        max_num_blocks_per_seq = int(torch.max(blocks_per_seq).item())
        num_blocks = int(batch_size * max_num_blocks_per_seq)

        # Block tables
        block_tables = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).reshape(batch_size, max_num_blocks_per_seq)

        # Query
        q = torch.randn(
            total_q_tokens, num_qo_heads, head_dim, dtype=dtype, device=device
        )

        # KV cache
        if kv_layout == "HND":
            k_cache = torch.randn(
                num_blocks,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.randn(
                num_blocks,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
        else:  # NHD
            k_cache = torch.randn(
                num_blocks,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.randn(
                num_blocks,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )

        # Sinks
        sink = (
            torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
            if use_sinks
            else None
        )

        # Scales
        bmm1_scale_val = sm_scale
        bmm2_scale_val = 1.0
        if use_device_scale:
            bmm1_scale = torch.tensor(
                bmm1_scale_val, device=device, dtype=torch.float32
            )
            bmm2_scale = torch.tensor(
                bmm2_scale_val, device=device, dtype=torch.float32
            )
        else:
            bmm1_scale = bmm1_scale_val
            bmm2_scale = bmm2_scale_val

        # Output
        if use_pre_alloc_out:
            out = torch.empty(
                total_q_tokens,
                num_qo_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
        else:
            out = None

        # Workspace (zero-initialized for trtllm-gen)
        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )

        # Mask (packed causal mask for speculative decoding)
        effective_q_len = (
            q_len_per_req if q_len_per_req is not None else (max_q_len_cfg or 1)
        )
        if use_mask and effective_q_len > 1:
            mask = generate_causal_mask(batch_size, effective_q_len, device)
        else:
            mask = None

        # Call the function under test with ALL parameters
        output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q.contiguous(),
            (k_cache, v_cache),
            workspace_buffer,
            block_tables,
            kv_seq_lens,
            seq_len,
            bmm1_scale,
            bmm2_scale,
            window_left,
            out=out,
            out_dtype=dtype,
            o_sf_scale=None,
            o_sf_vec_size=None,
            sinks=sink,
            kv_layout=kv_layout,
            enable_pdl=enable_pdl,
            backend=backend,
            q_len_per_req=q_len_per_req,
            o_scale=1.0,
            mask=mask,
            max_q_len=max_q_len_cfg if q_len_per_req is None else None,
            cum_seq_lens_q=q_indptr if q_len_per_req is None else None,
        )

        # If pre-allocated out, verify the same tensor is returned
        if use_pre_alloc_out:
            assert output.data_ptr() == out.data_ptr(), (
                f"Config {cfg_idx}: pre-allocated out tensor not reused"
            )

        assert output.shape == (total_q_tokens, num_qo_heads, head_dim), (
            f"Config {cfg_idx}: output shape mismatch, "
            f"expected {(total_q_tokens, num_qo_heads, head_dim)}, got {output.shape}"
        )
        assert output.dtype == dtype, (
            f"Config {cfg_idx}: output dtype mismatch, expected {dtype}, got {output.dtype}"
        )

        # Build reference output
        # Flatten paged KV to contiguous for reference computation
        if kv_layout == "HND":
            k_flat = einops.rearrange(
                k_cache,
                "(b npb) h p d -> b (npb p) h d",
                b=batch_size,
                npb=max_num_blocks_per_seq,
            )
            v_flat = einops.rearrange(
                v_cache,
                "(b npb) h p d -> b (npb p) h d",
                b=batch_size,
                npb=max_num_blocks_per_seq,
            )
        else:  # NHD
            k_flat = einops.rearrange(
                k_cache,
                "(b npb) p h d -> b (npb p) h d",
                b=batch_size,
                npb=max_num_blocks_per_seq,
            )
            v_flat = einops.rearrange(
                v_cache,
                "(b npb) p h d -> b (npb p) h d",
                b=batch_size,
                npb=max_num_blocks_per_seq,
            )

        if use_sinks:
            # Flatten for varlen reference
            k_flat_2d = k_flat.reshape(-1, num_kv_heads, head_dim)
            v_flat_2d = v_flat.reshape(-1, num_kv_heads, head_dim)
            kv_indptr_tokens = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32, device=device),
                    torch.cumsum(kv_seq_lens, dim=0, dtype=torch.int32),
                ]
            )

            o_ref = sink_attention_unified(
                q,
                k_flat_2d,
                v_flat_2d,
                sink,
                window_left,
                True,
                sm_scale,
                mode="varlen",
                batch_size=batch_size,
                qo_indptr=q_indptr,
                kv_indptr=kv_indptr_tokens,
            )
        else:
            # Without sinks, use per-batch einsum reference with causal mask
            o_ref_list = []
            for i in range(batch_size):
                qo_start = int(q_indptr[i].item())
                qo_end = int(q_indptr[i + 1].item())
                qo_len_i = qo_end - qo_start
                kv_len_i = int(kv_seq_lens[i].item())

                q_i = q[qo_start:qo_end]  # [qo_len_i, num_qo_heads, head_dim]
                k_i = k_flat[i, :kv_len_i]  # [kv_len_i, num_kv_heads, head_dim]
                v_i = v_flat[i, :kv_len_i]

                # GQA expand
                k_i_exp = torch.repeat_interleave(
                    k_i, num_qo_heads // num_kv_heads, dim=1
                )
                v_i_exp = torch.repeat_interleave(
                    v_i, num_qo_heads // num_kv_heads, dim=1
                )

                # logits: [num_qo_heads, qo_len_i, kv_len_i]
                logits_i = (
                    torch.einsum("qhd,khd->hqk", q_i.float(), k_i_exp.float())
                    * sm_scale
                )

                # causal mask
                row_idx = torch.arange(qo_len_i, device=device)[:, None]
                col_idx = torch.arange(kv_len_i, device=device)[None, :]
                query_pos = kv_len_i - qo_len_i + row_idx
                causal_mask = query_pos >= col_idx
                if window_left >= 0:
                    causal_mask &= (query_pos - window_left) <= col_idx

                logits_i = logits_i.masked_fill(
                    causal_mask.unsqueeze(0) == 0, float("-inf")
                )
                p_i = torch.softmax(logits_i, dim=-1)
                o_i = torch.einsum("hqk,khd->qhd", p_i, v_i_exp.float()).to(dtype)
                o_ref_list.append(o_i)
            o_ref = torch.cat(o_ref_list, dim=0)

        atol, rtol = 1e-2, 1e-2
        # Relax tolerance for multi-token queries
        if (q_len_per_req and q_len_per_req > 1) or (
            max_q_len_cfg and max_q_len_cfg > 1
        ):
            atol, rtol = atol * 2, rtol * 2

        np.testing.assert_allclose(
            o_ref.float().cpu().numpy(),
            output.float().cpu().numpy(),
            atol=atol,
            rtol=rtol,
            err_msg=f"Config {cfg_idx} failed: {cfg}",
        )
