"""Install the plain Ascend mff/torch pair style into a clean LAMMPS source tree."""
import argparse
from pathlib import Path
import shutil


def install(root):
    root = Path(root).resolve()
    source = Path(__file__).resolve().parent
    cmake = root / 'cmake/CMakeLists.txt'
    if not cmake.is_file() or not (root/'src/pair.h').is_file():
        raise ValueError('Expected the root of a LAMMPS source tree')
    package = root/'src/USER-MFFTORCH'
    marker = package/'.ictc_ascend'
    if package.exists() and any(package.iterdir()) and not marker.is_file():
        raise ValueError('USER-MFFTORCH already exists; use a fresh LAMMPS tree for the Ascend build')
    package.mkdir(exist_ok=True)
    for name in ('pair_mff_torch.cpp', 'pair_mff_torch.h'):
        shutil.copy2(source/'src/ASCEND'/name, package/name)
    shutil.copy2(source/'src/ASCEND/mff_periodic_table.h', package/'mff_periodic_table.h')
    marker.write_text('Ascend eager plain pair style\n')
    packages = root/'cmake/Modules/Packages'
    if not packages.is_dir():
        packages = root/'cmake/Packages'
    packages.mkdir(exist_ok=True, parents=True)
    target = packages/'USER-MFFTORCH.cmake'
    shutil.copy2(source/'cmake/Ascend.cmake', target)
    text = cmake.read_text()
    if 'set(STANDARD_PACKAGES\n  USER-MFFTORCH\n' not in text:
        if text.count('set(STANDARD_PACKAGES\n') != 1:
            raise ValueError('Unsupported LAMMPS CMake package registration layout')
        text = text.replace('set(STANDARD_PACKAGES\n', 'set(STANDARD_PACKAGES\n  USER-MFFTORCH\n', 1)
    if 'foreach(PKG_WITH_INCL USER-MFFTORCH ' not in text:
        if 'foreach(PKG_WITH_INCL ' not in text:
            raise ValueError('Unsupported LAMMPS CMake package include layout')
        text = text.replace('foreach(PKG_WITH_INCL ', 'foreach(PKG_WITH_INCL USER-MFFTORCH ', 1)
    cmake.write_text(text)
    print(f'Installed Ascend mff/torch into {root}; enable PKG_USER-MFFTORCH and BUILD_MPI')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lammps_source')
    install(parser.parse_args().lammps_source)
