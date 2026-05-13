#!/bin/bash
set -e

python -m pytest -rs tests/attention/test_attention_sink_blackwell.py -k test_blackwell_trtllm_gen_context_attention_sink
python -m pytest -rs tests/attention/test_attention_sink_blackwell.py -k test_blackwell_trtllm_gen_decode_attention_sink
python -m pytest -rs tests/moe/test_trtllm_gen_fused_moe.py::test_fp8_block_scale_routed_activation_type_relu2_smoke
python -m pytest -rs "tests/comm/test_trtllm_allreduce_fusion.py::test_trtllm_allreduce_fusion[True-1024-dtype0-2]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_renormalize_routing[...FP8_Block_DeepSeek-1024-1024-8-RandomHiddenStates]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_sigmoid_routing[...FP8_Block_DeepSeek-1024-1024-8]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_dyn_block_kernel_routing[...FP8_Block_DeepSeek...]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_tier_1024_experts_routing[...FP8_Block_DeepSeek...]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_deepseek_ngroup1_block_per_token_routing[...FP8_Block_DeepSeek...]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_routing_dtype_flexibility[...FP8_Block_DeepSeek...]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_mxfp8_block_scale_moe_relu2_non_gated[...Shuffled E32_K4]"
# python -m pytest -rs tests/moe/test_trtllm_gen_fused_moe.py::test_mxfp8_block_scale_moe_relu2_deepseekv3_topk22
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_fp8_block_scale_autotune_valid_configs[...MxFp8_Relu2]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_fp8_per_tensor_autotune_valid_configs_nonefp8[...PerTensor_Swiglu]"
# python -m pytest -rs "tests/moe/test_trtllm_gen_fused_moe.py::test_llama4_routing[...FP8_Tensor-1024-1024-8]"
# python -m pytest -rs tests/moe/test_trtllm_gen_fused_moe.py::test_deepseekv3_routing
# python -m pytest -rs tests/moe/test_trtllm_gen_fused_moe.py::test_nvfp4_moe_gemm_bias
python -m pytest -rs tests/norm/test_fused_rmsnorm_silu.py
