#!/usr/bin/env python3
"""GPU Memory Monitoring Utility"""

import jax
import jax.numpy as jnp
import psutil
import os

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3, info.total / 1024**3
    except ImportError:
        return None, None

def get_system_memory_usage():
    """Get current system memory usage in GB"""
    memory = psutil.virtual_memory()
    return memory.used / 1024**3, memory.total / 1024**3

def print_memory_usage():
    """Print current memory usage"""
    gpu_used, gpu_total = get_gpu_memory_usage()
    sys_used, sys_total = get_system_memory_usage()
    
    print(f"System Memory: {sys_used:.2f}GB / {sys_total:.2f}GB ({sys_used/sys_total*100:.1f}%)")
    if gpu_used is not None:
        print(f"GPU Memory: {gpu_used:.2f}GB / {gpu_total:.2f}GB ({gpu_used/gpu_total*100:.1f}%)")
    
    # JAX device info
    devices = jax.devices()
    print(f"JAX Devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")

def clear_jax_caches():
    """Clear JAX caches to free memory"""
    jax.clear_caches()
    print("Cleared JAX caches")

if __name__ == "__main__":
    print_memory_usage() 