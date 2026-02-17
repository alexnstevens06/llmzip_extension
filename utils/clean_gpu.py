#!/usr/bin/env python3
"""Kill any processes holding the GPU and free VRAM.

Run this before starting a benchmark if a previous run was interrupted.
Usage: python clean_gpu.py
"""

import os
import subprocess
import time

MY_PID = os.getpid()


def get_gpu_pids():
    """Get PIDs of processes using the AMD GPU via fuser."""
    try:
        result = subprocess.run(
            ["fuser", "/dev/kfd"],
            capture_output=True, text=True, timeout=5
        )
        # fuser outputs PIDs to stderr
        output = (result.stdout + " " + result.stderr).strip()
        pids = set()
        for token in output.split():
            token = token.strip().rstrip("m")  # fuser may append 'm'
            if token.isdigit():
                pid = int(token)
                if pid != MY_PID:
                    pids.add(pid)
        return pids
    except Exception as e:
        print(f"Warning: Could not query GPU processes: {e}")
        return set()


def get_vram_usage():
    """Get VRAM usage percentage from rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if line.strip().startswith("0"):
                parts = line.split()
                for part in parts:
                    if part.endswith("%") and parts.index(part) > 5:
                        return part
        return "unknown"
    except Exception:
        return "unknown"


def main():
    gpu_pids = get_gpu_pids()

    if not gpu_pids:
        print(f"No GPU processes found. VRAM usage: {get_vram_usage()}")
        return

    print(f"Found {len(gpu_pids)} process(es) using the GPU: {gpu_pids}")

    for pid in gpu_pids:
        try:
            os.kill(pid, 9)  # SIGKILL
            print(f"  Killed PID {pid}")
        except ProcessLookupError:
            print(f"  PID {pid} already gone")
        except PermissionError:
            print(f"  No permission to kill PID {pid}")

    time.sleep(2)

    remaining = get_gpu_pids()
    if remaining:
        print(f"WARNING: {len(remaining)} process(es) still using GPU: {remaining}")
    else:
        print(f"GPU cleared. VRAM usage: {get_vram_usage()}")


if __name__ == "__main__":
    main()
