"""Reproducible configure/build/test helper for the native MMCPU runtime."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native_cpu"

class BuildError(RuntimeError): pass

def _find_cmake(explicit=None):
    if explicit: return Path(explicit)
    found = shutil.which("cmake")
    if found: return Path(found)
    candidates=[]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            candidates.extend(Path(base).glob("Microsoft Visual Studio/*/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"))
    if candidates: return sorted(candidates)[-1]
    raise BuildError("cmake was not found on PATH or in Visual Studio BuildTools")

def _env():
    env = dict(os.environ)
    if os.name == "nt":
        normalized = {}
        for key, value in env.items(): normalized[key.upper()] = value
        env = normalized
    return env

def _run(cmd, cwd, env):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=True)

def _cache(path):
    values={}
    if not path.is_file(): return values
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line or line.startswith("//") or line.startswith("#"): continue
        key, value = line.split("=", 1); values[key.split(":",1)[0]] = value
    return values

def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def _source_hashes():
    out={}
    roots = [NATIVE / "src", NATIVE / "include", NATIVE / "tests", NATIVE / "CMakeLists.txt"]
    files = [roots[-1]] + [p for base in roots[:-1] for p in base.rglob("*")]
    for path in sorted(files):
        if not path.is_file(): continue
        if path.suffix.lower() in {".cpp", ".cc", ".cxx", ".h", ".hpp", ".cmake"} or path.name == "CMakeLists.txt":
            out[path.relative_to(ROOT).as_posix()] = _sha(path)
    return out

def _find_library(build):
    names=("minimind_cpu.dll", "libminimind_cpu.so", "libminimind_cpu.dylib", "minimind_cpu.so")
    release = [build / "Release" / name for name in names if (build / "Release" / name).is_file()]
    if len(release) == 1: return release[0]
    if len(release) > 1: raise BuildError("multiple Release native libraries found")
    candidates=[p for p in (build / name for name in names) if p.is_file()]
    if len(candidates) == 1: return candidates[0]
    if not candidates: raise BuildError(f"built native library not found under {build}")
    raise BuildError("ambiguous native libraries; expected one Release artifact")

def _compiler_metadata(build):
    files = sorted((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if not files: raise BuildError("CMake compiler metadata not found")
    vals={}
    for line in files[-1].read_text(errors="replace").splitlines():
        if line.startswith("set(CMAKE_CXX_") and '"' in line:
            key=line.split("set(",1)[1].split(" ",1)[0]
            value=line.split('"',2)[1]
            vals[key]=value
    if not all(vals.get(k) for k in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION")):
        raise BuildError("CMake compiler metadata is incomplete")
    return vals

def build(*, cmake=None, generator=None, build_dir=None):
    cmake = _find_cmake(cmake); build_dir = Path(build_dir) if build_dir else NATIVE / "build"
    build_dir.mkdir(parents=True, exist_ok=True); env = _env()
    gen = generator
    if gen is None and os.name == "nt" and "Visual Studio" in str(cmake): gen = "Visual Studio 18 2026"
    configure=[cmake, "-S", NATIVE, "-B", build_dir, "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    if gen: configure += ["-G", gen]
    if os.name == "nt" and gen and "Visual Studio" in gen: configure += ["-A", "x64"]
    _run(configure, ROOT, env)
    build_cmd=[cmake, "--build", build_dir, "--config", "Release", "--parallel", "1"]
    if os.name == "nt" and gen and "Visual Studio" in gen: build_cmd += ["--", "/nr:false"]
    _run(build_cmd, ROOT, env)
    ctest=cmake.with_name("ctest.exe" if os.name == "nt" else "ctest")
    if not ctest.is_file(): ctest=Path(shutil.which("ctest") or ctest)
    _run([ctest, "--test-dir", build_dir, "-C", "Release", "--output-on-failure"], ROOT, env)
    library=_find_library(build_dir); cache=_cache(build_dir/"CMakeCache.txt"); compiler=_compiler_metadata(build_dir)
    compiler_path=compiler["CMAKE_CXX_COMPILER"]
    flags = [cache.get("CMAKE_CXX_FLAGS", ""), cache.get("CMAKE_CXX_FLAGS_RELEASE", "")]
    if compiler["CMAKE_CXX_COMPILER_ID"] == "MSVC": flags.append("/W4 /permissive- /EHsc /fp:precise /std:c++20")
    else: flags.append("-Wall -Wextra -Wpedantic -std=c++20")
    info={"schema_version":1,"build_type":"Release","native_threads":1,
          "compiler":{"id":compiler["CMAKE_CXX_COMPILER_ID"],"version":compiler["CMAKE_CXX_COMPILER_VERSION"],"path":str(Path(compiler_path).resolve())},
          "compile_flags":" ".join(x for x in flags if x),
          "generator":gen or cache.get("CMAKE_GENERATOR", ""), "cmake":str(cmake.resolve()),
          "library":str(library.resolve()), "library_sha256":_sha(library), "native_sources_sha256":_source_hashes()}
    info_path=build_dir/"build-info.json"; info_path.write_text(json.dumps(info,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(info,indent=2,sort_keys=True)); return info

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--cmake"); p.add_argument("--generator"); p.add_argument("--build-dir",type=Path)
    args=p.parse_args(argv)
    try: build(cmake=args.cmake,generator=args.generator,build_dir=args.build_dir)
    except (BuildError, OSError, subprocess.CalledProcessError) as exc: print(f"error: {exc}",file=sys.stderr); return 1
    return 0
if __name__ == "__main__": raise SystemExit(main())
