# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Third Party
import weakref
import torch
import logging
# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: weakref.WeakValueDictionary[int, torch.Tensor] = weakref.WeakValueDictionary()
logger = logging.getLogger(__name__)

def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    Note: NUMA and pinned memory are not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    logger.info(f"[*] Warning: NUMA-aware pinned memory allocation is not supported on non-CUDA. Proceeding with standard pinned memory allocation.")
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)
    logger.info(f"[*] Allocating NUMA-aware pinned memory of size {size} bytes on NUMA node {numa_id}")

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)
    logger.info(f"[*] First-touch initialization completed")  
    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()
    logger.info(f"[*] Obtained data pointer: {ptr}")
    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. Note: Pinned memory is not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    logger.info(f"[*] emptying pinned memory allocation of size {size} bytes")
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)
    logger.info(f"[*] Allocating pinned memory of size {size} bytes")
    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)
    logger.info(f"[*] First-touch initialization completed")
    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()
    logger.info(f"[*] Obtained data pointer: {ptr}")
    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)
