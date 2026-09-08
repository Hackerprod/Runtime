import ctypes

from native_cpu.tools.native import NativeRuntime, _MmRuntimeStats


def test_stats_abi_is_stable_and_exposed():
    names = [name for name, _ in _MmRuntimeStats._fields_]
    assert names[0] == "lm_head_calls"
    assert "qkv_ns" in names
    assert "attention_qk_ns" not in names
    assert "controller_wait_ns" in names
    assert ctypes.sizeof(_MmRuntimeStats) == 1136
    assert _MmRuntimeStats.output_projection_ns.offset == 72
    assert _MmRuntimeStats.participant_compute_ns.offset == 104
    assert _MmRuntimeStats.participant_compute_calls.offset == 624
    assert hasattr(NativeRuntime, "stats")
    assert hasattr(NativeRuntime, "configure_profile")
    assert hasattr(NativeRuntime, "reset_stats")
    assert hasattr(NativeRuntime, "configure_selective_logits")
    assert hasattr(NativeRuntime, "selective_logits")
    assert hasattr(NativeRuntime, "configure_ffn_row4")
    assert hasattr(NativeRuntime, "ffn_row4")
    assert hasattr(NativeRuntime, "configure_gqa_k_shared")
    assert hasattr(NativeRuntime, "gqa_k_shared")
    assert hasattr(NativeRuntime, "attention_qk_ns")
    assert hasattr(NativeRuntime, "stats_abi_incompatible")
    assert hasattr(NativeRuntime, "configure_gqa_v_shared")
    assert hasattr(NativeRuntime, "gqa_v_shared")
    assert hasattr(NativeRuntime, "attention_v_ns")
    assert hasattr(NativeRuntime, "gqa_v_shared_fallbacks")
    assert hasattr(NativeRuntime, "attention_f16_storage")
    assert hasattr(NativeRuntime, "attention_f16_storage_bytes")
    assert hasattr(NativeRuntime, "attention_f16_prepare_ns")
