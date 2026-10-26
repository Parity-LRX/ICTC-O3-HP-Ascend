"""Build native kernels inside a configured Ascend compute environment."""
import argparse
from pathlib import Path
import os
import subprocess
import shutil


def build(output, *, soc=None, jobs=4):
    if not isinstance(jobs,int) or isinstance(jobs,bool) or jobs<1:
        raise ValueError('jobs must be a positive integer')
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_INSTALL_PATH")
    if not cann:
        raise RuntimeError("Source the CANN environment before building")
    soc = soc or torch.npu.get_device_name(torch.npu.current_device())
    npu = Path(torch_npu.__file__).resolve().parent
    cmake = os.environ.get("ICTC_ASCENDC_CMAKE") or shutil.which("cmake") or shutil.which("cmake3")
    if not cmake:
        raise RuntimeError("CMake >= 3.16 is required to build Ascend C kernels")
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if not compiler:
        raise RuntimeError('A host C++ compiler is required to build Ascend C kernels')
    probe = subprocess.run([compiler,"-E","-x","c++","-v","-"],input="",text=True,capture_output=True,check=True,env={**os.environ,"LC_ALL":"C"})
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
        raise RuntimeError("C++ standard headers not found: "+probe.stderr)
    print("Native C++ include paths:", includes, flush=True)
    command = [cmake, "-S", str(source), "-B", str(output),
               "-DCMAKE_BUILD_TYPE=Release", "-DASCEND_CANN_PACKAGE_PATH="+cann,
               "-DSOC_VERSION="+soc, "-DTORCH_NPU_PATH="+str(npu),
               "-DTORCH_INCLUDE_DIRS="+";".join(include_paths()),
               "-DTORCH_LIBRARY_DIRS="+";".join(library_paths()),
               "-DICTC_CXX11_ABI="+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
               "-DICTC_HOST_CXX_INCLUDES="+";".join(includes)]
    build_env = {**os.environ,"CPLUS_INCLUDE_PATH":os.pathsep.join(includes+[os.environ.get("CPLUS_INCLUDE_PATH","")])}
    subprocess.run(command, check=True,env=build_env)
    subprocess.run([cmake, "--build", str(output), "--parallel", str(jobs)], check=True,env=build_env)
    return output/"libictc_ascendc.so"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soc")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    print(build(args.output, soc=args.soc, jobs=args.jobs))


if __name__ == "__main__":
    main()
