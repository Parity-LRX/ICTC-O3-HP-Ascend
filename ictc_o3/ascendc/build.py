"""Build and optionally install the PyTorch extension with CANN's ASC CMake API."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess


def _toolkit(path=None):
    selected = path
    if selected is None:
        selected = next((os.environ[name] for name in (
            "ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"
        ) if os.environ.get(name)), None)
    if not selected:
        raise RuntimeError("Source CANN set_env.sh or pass --cann-path to the toolkit root")
    root = Path(selected).expanduser().resolve(strict=True)
    module = root / "compiler/tikcpp/ascendc_kernel_cmake/asc_modules/FindASC.cmake"
    if not module.is_file():
        raise RuntimeError(f"CANN ASC CMake modules not found: {module}; install the CANN compiler package")
    return root


def _cmake():
    command = os.environ.get("ICTC_ASCENDC_CMAKE") or shutil.which("cmake") or shutil.which("cmake3")
    if not command:
        raise RuntimeError("CMake >= 3.20 is required to build Ascend C kernels")
    version = subprocess.run([command, "--version"], check=True, capture_output=True, text=True)
    match = re.search(r"cmake version (\d+)\.(\d+)", version.stdout)
    if not match or tuple(map(int, match.groups())) < (3, 20):
        raise RuntimeError("CMake >= 3.20 is required to build Ascend C kernels")
    return command


def _host_includes():
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("A host C++ compiler is required to build Ascend C kernels")
    probe = subprocess.run(
        compiler + ["-E", "-x", "c++", "-v", "-"], input="", text=True,
        capture_output=True, check=True, env={**os.environ, "LC_ALL": "C"},
    )
    includes = []
    collecting = False
    for line in probe.stderr.splitlines():
        if "#include <...> search starts here:" in line:
            collecting = True
        elif "End of search list." in line:
            collecting = False
        elif collecting and Path(line.strip()).is_dir():
            includes.append(line.strip())
    # Bisheng supplies Clang intrinsics; GCC arm_neon.h is incompatible.
    includes = [path for path in includes if "/lib/gcc/" not in path]
    if not includes:
        raise RuntimeError("C++ standard headers not found: " + probe.stderr)
    return includes


def _check_cache(output, expected):
    cache = output / "CMakeCache.txt"
    if not cache.is_file():
        return
    existing = {}
    for line in cache.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            existing[key.split(":", 1)[0]] = value
    for key, value in expected.items():
        if key in existing and existing[key] != value:
            raise RuntimeError(f"Build directory has a different {key}; use a new --output directory")


def build(output, *, soc=None, jobs=4, cann_path=None, install_prefix=None, build_type="Release"):
    """Return the built .so, or the installed .so when install_prefix is given.

    An explicit soc (or SOC_VERSION) allows compilation without NPU discovery.
    The caller must provide the site's matching CANN, PyTorch and torch_npu.
    """
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if build_type not in ("Release", "Debug", "RelWithDebInfo"):
        raise ValueError("build_type must be Release, Debug or RelWithDebInfo")
    if platform.system() != "Linux":
        raise RuntimeError("Build Ascend C kernels in the target Linux compute environment")
    source = Path(__file__).resolve().parent
    output = Path(output).expanduser().resolve()
    if output == source or output in source.parents:
        raise ValueError("The output directory must not contain the Ascend C source directory")
    cann = _toolkit(cann_path)
    cmake = _cmake()

    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    torch_release = re.match(r"(\d+)\.(\d+)", torch.__version__)
    npu_release = re.match(r"(\d+)\.(\d+)", torch_npu.__version__)
    if not torch_release or not npu_release or torch_release.groups() != npu_release.groups():
        raise RuntimeError("PyTorch and torch_npu major/minor versions must match; use the official compatibility matrix")
    soc = soc or os.environ.get("SOC_VERSION")
    if soc is None:
        soc = torch.npu.get_device_name(torch.npu.current_device())
    if not isinstance(soc, str) or not re.fullmatch(r"Ascend[A-Za-z0-9_]+", soc):
        raise ValueError("soc must be an Ascend chip name such as Ascend910B3")
    npu = Path(torch_npu.__file__).resolve().parent
    abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    _check_cache(output, {
        "CMAKE_HOME_DIRECTORY": str(source), "ASCEND_CANN_PACKAGE_PATH": str(cann),
        "SOC_VERSION": soc, "TORCH_NPU_PATH": str(npu), "ICTC_CXX11_ABI": abi,
    })
    includes = _host_includes()
    command = [
        cmake, "-S", str(source), "-B", str(output),
        "-DCMAKE_BUILD_TYPE=" + build_type, "-DASCEND_CANN_PACKAGE_PATH=" + str(cann),
        "-DSOC_VERSION=" + soc, "-DTORCH_NPU_PATH=" + str(npu),
        "-DTORCH_INCLUDE_DIRS=" + ";".join(include_paths()),
        "-DTORCH_LIBRARY_DIRS=" + ";".join(library_paths()),
        "-DICTC_CXX11_ABI=" + abi, "-DICTC_HOST_CXX_INCLUDES=" + ";".join(includes),
        "-DCMAKE_INSTALL_LIBDIR=lib",
    ]
    prefix = Path(install_prefix).expanduser().resolve() if install_prefix is not None else None
    if prefix is not None:
        if prefix == source or prefix in source.parents:
            raise ValueError("The install prefix must not contain the Ascend C source directory")
        command.append("-DCMAKE_INSTALL_PREFIX=" + str(prefix))
    build_env = {
        **os.environ, "ASCEND_HOME_PATH": str(cann),
        "CPLUS_INCLUDE_PATH": os.pathsep.join(includes + [os.environ.get("CPLUS_INCLUDE_PATH", "")]),
    }
    print(f"CANN ASC build: soc={soc}, torch={torch.__version__}, torch_npu={torch_npu.__version__}, ABI={abi}", flush=True)
    subprocess.run(command, check=True, env=build_env)
    subprocess.run([cmake, "--build", str(output), "--parallel", str(jobs)], check=True, env=build_env)
    library = output / "libictc_ascendc.so"
    if not library.is_file():
        raise RuntimeError(f"Build completed without the expected native library: {library}")
    metadata = dict(
        build_api="CANN ASC", cann_path=str(cann), soc=soc, build_type=build_type,
        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
        cxx11_abi=int(abi), host_architecture=platform.machine(),
    )
    (output / "ictc_ascendc_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if prefix is not None:
        subprocess.run([cmake, "--install", str(output), "--component", "ictc_runtime"], check=True, env=build_env)
        library = prefix / "lib/libictc_ascendc.so"
        if not library.is_file():
            raise RuntimeError(f"Install completed without the expected native library: {library}")
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Out-of-source CMake build directory")
    parser.add_argument("--soc", help="Target chip name; default: SOC_VERSION or current NPU")
    parser.add_argument("--cann-path", help="CANN toolkit root; default: sourced CANN environment")
    parser.add_argument("--install-prefix", help="Optionally install the runtime library under PREFIX/lib")
    parser.add_argument("--build-type", choices=("Release", "Debug", "RelWithDebInfo"), default="Release")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    print(build(args.output, soc=args.soc, jobs=args.jobs, cann_path=args.cann_path,
                install_prefix=args.install_prefix, build_type=args.build_type))


if __name__ == "__main__":
    main()
