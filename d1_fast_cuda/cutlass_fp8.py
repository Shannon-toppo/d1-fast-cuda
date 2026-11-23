"""The CUTLASS fp8 GEMM (csrc/fp8_cutlass.cu) as a shared library loaded through ctypes.

`build()` compiles it once with nvcc (C++17; on Windows with MSVC, found through
vcvars64.bat when cl.exe is not on PATH) against the CUTLASS headers (`D1_CUTLASS`,
default third_party/cutlass), into `D1_CUTLASS_BUILD` (default ~/.cache/d1_fast_cuda).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("csrc") / "fp8_cutlass.cu"
_lib = None


def _msvc_env() -> None:
    """On Windows, import vcvars64.bat's environment into this process if cl.exe is missing."""
    if sys.platform != "win32" or shutil.which("cl"):
        return
    roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
             Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio"]
    found = sorted((p for r in roots if r.exists() for p in r.glob("*/*/VC/Auxiliary/Build/vcvars64.bat")),
                   key=lambda p: ("2022" not in str(p), str(p)))
    if not found:
        raise RuntimeError("MSVC not found: build from a 'x64 Native Tools Command Prompt'")
    out = subprocess.run(f'"{found[0]}" >nul && set', shell=True, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            os.environ[key] = value


def _nvcc() -> str:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        for name in ("nvcc.exe", "nvcc"):
            if (Path(home) / "bin" / name).exists():
                return str(Path(home) / "bin" / name)
    found = shutil.which("nvcc")
    if not found:
        raise RuntimeError("nvcc not found (set CUDA_HOME)")
    return found


def build(verbose: bool = False) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    cutlass = Path(os.environ.get("D1_CUTLASS", ROOT / "third_party" / "cutlass"))
    if not (cutlass / "include" / "cutlass" / "cutlass.h").exists():
        raise RuntimeError(f"CUTLASS headers not found at {cutlass} (set D1_CUTLASS)")
    out_dir = Path(os.environ.get("D1_CUTLASS_BUILD", Path.home() / ".cache" / "d1_fast_cuda"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(SOURCE.read_bytes() + str(cutlass.resolve()).encode()).hexdigest()[:12]
    lib_path = out_dir / (f"d1_fp8_cutlass_{tag}" + (".dll" if sys.platform == "win32" else ".so"))
    if not lib_path.exists():
        _msvc_env()
        cmd = [_nvcc(), "-shared", "-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr",
               "-gencode=arch=compute_89,code=sm_89", "-I", str(cutlass / "include"),
               str(SOURCE), "-o", str(lib_path)]
        if sys.platform == "win32":
            cmd += ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/MD"]
        else:
            cmd += ["-Xcompiler", "-fPIC"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if verbose or r.returncode:
            print(r.stdout[-4000:], r.stderr[-4000:])
        if r.returncode:
            raise RuntimeError("building the CUTLASS fp8 GEMM failed")
    lib = ctypes.CDLL(str(lib_path))
    lib.d1_fp8_count.restype = ctypes.c_int
    lib.d1_fp8_name.restype = ctypes.c_char_p
    lib.d1_fp8_name.argtypes = [ctypes.c_int]
    lib.d1_fp8_workspace.restype = ctypes.c_size_t
    lib.d1_fp8_workspace.argtypes = [ctypes.c_int] * 5
    lib.d1_fp8_mm.restype = ctypes.c_int
    lib.d1_fp8_mm.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
    _lib = lib
    return lib


def configs() -> list[str]:
    lib = build()
    return [lib.d1_fp8_name(i).decode() for i in range(lib.d1_fp8_count())]


_SMS: dict = {}


def mm(x8: torch.Tensor, w8: torch.Tensor, config: int, sms: int | None = None) -> torch.Tensor:
    """float32 (M, N) = x8 (M, K) @ w8 (N, K).T, both e4m3, unscaled."""
    lib = build()
    m, k = x8.shape
    n = w8.shape[0]
    assert x8.is_contiguous() and w8.is_contiguous() and w8.shape[1] == k
    if sms is None:
        dev = x8.device.index
        sms = _SMS.get(dev) or _SMS.setdefault(dev, torch.cuda.get_device_properties(x8.device).multi_processor_count)
    out = torch.empty((m, n), device=x8.device, dtype=torch.float32)
    ws_size = lib.d1_fp8_workspace(config, m, n, k, sms)
    ws = torch.empty(max(ws_size, 1), device=x8.device, dtype=torch.uint8)
    stream = torch.cuda.current_stream(x8.device).cuda_stream
    rc = lib.d1_fp8_mm(config, m, n, k, x8.data_ptr(), w8.data_ptr(), out.data_ptr(), ws.data_ptr(), sms, stream)
    if rc:
        raise RuntimeError(f"CUTLASS fp8 GEMM {lib.d1_fp8_name(config).decode()} failed ({rc}) for {m}x{n}x{k}")
    return out
