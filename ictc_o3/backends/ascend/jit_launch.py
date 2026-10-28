# SPDX-License-Identifier: MIT
"""Serialize cold Triton-Ascend compilation into a shared filesystem cache.

Concurrent compiler writers can publish metadata before the NPU binary is
visible on NFS. Hot launches use the existing per-device JIT cache directly;
the lock never synchronizes device work or caches live numerical inputs.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import threading

_BINDER_LOCK = threading.RLock()


def launch(kernel, grid, *args, **options):
    from triton.runtime import driver
    from triton.compiler import make_backend
    from triton.runtime.cache import default_cache_dir

    device = driver.active.get_current_device()
    options['debug'] = options.get('debug', False) or os.environ.get('TRITON_DEBUG', '0') == '1'
    with _BINDER_LOCK:
        if kernel.binder is None:
            kernel.create_binder(make_backend(driver.active.get_current_target()))
    # Use the qualified JIT's exact type/alignment/constant signature, including
    # dynamic-N specialization. A cache hit does not open a file or take flock.
    _, signature, constants, _, extra = kernel.binder(*args, **options)
    key = ''.join(signature) + str((constants, extra))
    if key in kernel.cache[device]:
        return kernel.run(*args, grid=grid, warmup=False, **options)
    directory = Path(os.environ.get('TRITON_CACHE_DIR') or default_cache_dir()) / '.ictc-compile-locks'
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((kernel.cache_key + key).encode()).hexdigest() + '.lock'
    with (directory / name).open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return kernel.run(*args, grid=grid, warmup=False, **options)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
