from native_cpu.tools.native import NativeRuntime, _MmRuntimeStats


def test_stats_abi_is_stable_and_exposed():
    names = [name for name, _ in _MmRuntimeStats._fields_]
    assert names[0] == "lm_head_calls"
    assert "qkv_ns" in names
    assert "attention_qk_ns" in names
    assert "controller_wait_ns" in names
    assert hasattr(NativeRuntime, "stats")
    assert hasattr(NativeRuntime, "configure_profile")
    assert hasattr(NativeRuntime, "reset_stats")
    assert hasattr(NativeRuntime, "configure_selective_logits")
    assert hasattr(NativeRuntime, "selective_logits")
    assert hasattr(NativeRuntime, "configure_ffn_row4")
    assert hasattr(NativeRuntime, "ffn_row4")
    assert hasattr(NativeRuntime, "configure_gqa_k_shared")
    assert hasattr(NativeRuntime, "gqa_k_shared")
