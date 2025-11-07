"""
Copyright (c) 2024 by FlashInfer team.

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

import ctypes
import functools
import os

# Re-export
from . import cubin_loader
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import gen_cudnn_fmha_module as gen_cudnn_fmha_module
from .attention import gen_batch_attention_module as gen_batch_attention_module
from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_attention_uri as get_batch_attention_uri
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .attention import gen_trtllm_gen_fmha_module as gen_trtllm_gen_fmha_module
from .core import JitSpec as JitSpec
from .core import JitSpecStatus as JitSpecStatus
from .core import JitSpecRegistry as JitSpecRegistry
from .core import jit_spec_registry as jit_spec_registry
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import gen_jit_spec as gen_jit_spec
from .core import MissingJITCacheError as MissingJITCacheError
from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
from .core import sm100a_nvcc_flags as sm100a_nvcc_flags
from .core import sm100f_nvcc_flags as sm100f_nvcc_flags
from .core import sm103a_nvcc_flags as sm103a_nvcc_flags
from .core import sm110a_nvcc_flags as sm110a_nvcc_flags
from .core import sm120a_nvcc_flags as sm120a_nvcc_flags
from .core import sm121a_nvcc_flags as sm121a_nvcc_flags
from .core import current_compilation_context as current_compilation_context
from .cubin_loader import setup_cubin_loader
from .comm import gen_comm_alltoall_module as gen_comm_alltoall_module
from .comm import gen_trtllm_mnnvl_comm_module as gen_trtllm_mnnvl_comm_module
from .comm import gen_trtllm_comm_module as gen_trtllm_comm_module
from .comm import gen_vllm_comm_module as gen_vllm_comm_module
from .comm import gen_nvshmem_module as gen_nvshmem_module
from typing import Optional


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}"
    )
    return path


from .dsv3_optimizations import (
    gen_dsv3_router_gemm_module as gen_dsv3_router_gemm_module,
)


cuda_lib_path = os.environ.get(
    "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
)
process_cudart_path = find_loaded_library("libcudart")
if process_cudart_path is not None:
    ctypes.CDLL(process_cudart_path, mode=ctypes.RTLD_GLOBAL)
elif os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
    ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
