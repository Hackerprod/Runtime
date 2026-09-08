"""Capture live Windows group-0 physical topology and power without changing them."""
import argparse
import ctypes
import importlib.util
import json
import platform
import struct
import winreg
from pathlib import Path
import psutil


def capture():
    spec = importlib.util.spec_from_file_location("cpu_r1_bench_env", Path(__file__).with_name("benchmark.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    result = bench.environment()
    size = ctypes.c_ulong(0)
    api = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
    api.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    api.restype = ctypes.c_int
    api(0, None, ctypes.byref(size))  # RelationProcessorCore
    if not size.value: raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not api(0, buffer, ctypes.byref(size)): raise ctypes.WinError(ctypes.get_last_error())
    data, offset, cores = buffer.raw, 0, []
    while offset < size.value:
        relation, length = struct.unpack_from("<II", data, offset)
        if relation != 0 or length < 48: raise ValueError("unexpected topology record")
        groups = []
        for i in range(struct.unpack_from("<H", data, offset+30)[0]):
            mask, group = struct.unpack_from("<QH", data, offset+32+i*16)
            groups.append({"group":group,"logical_cpus":[n for n in range(64) if mask & (1 << n)]})
        cores.append({"efficiency_class":data[offset+9],"groups":groups})
        offset += length
    selected = [0,2,4,6]
    physical = [next(i for i,c in enumerate(cores) if any(g["group"]==0 and n in g["logical_cpus"] for g in c["groups"])) for n in selected]
    if len(set(physical)) != 4: raise ValueError("P4 configuration aliases physical cores")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        processor = winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    result.update(platform=platform.platform(),processor=processor,cores=cores,
        topology_source="GetLogicalProcessorInformationEx(RelationProcessorCore)",
        selected_profiles=bench.PROFILES,physical_cpu_count=psutil.cpu_count(logical=False),
        logical_cpu_count=psutil.cpu_count(),memory_total_bytes=psutil.virtual_memory().total,
        limitations=["No temperature, energy, cache-miss or actual memory-traffic counters.",
                    "Desktop activity is observed, not controlled; no unrelated processes terminated."])
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=capture()
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
        stream.write("\n")
