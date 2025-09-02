"""
Device utilities for cross-platform PyTorch support.
Provides automatic device detection with fallback hierarchy: CUDA -> MPS -> CPU
"""
import torch
import os
from typing import Optional


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_device_for_distributed_training() -> None:
    """
    Set the appropriate device for distributed training.
    For CUDA, sets the local device based on LOCAL_RANK.
    For MPS/CPU, this is a no-op since they don't support multi-GPU.
    """
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_map_location() -> Optional[str]:
    """
    Get the appropriate map_location string for torch.load().
    
    Returns:
        Optional[str]: Device string for map_location, or None for automatic detection
    """
    device = get_device()
    if device.type == "cuda":
        return "cuda"
    elif device.type == "mps":
        return "mps"
    else:
        return "cpu"


def move_batch_to_device(batch: dict, device: Optional[torch.device] = None) -> dict:
    """
    Move a batch of tensors to the specified device.
    
    Args:
        batch: Dictionary of tensors
        device: Target device (if None, uses get_device())
        
    Returns:
        dict: Batch with tensors moved to device
    """
    if device is None:
        device = get_device()
    
    return {k: v.to(device) for k, v in batch.items()}


def create_device_context(device: Optional[torch.device] = None):
    """
    Create a device context for tensor creation.
    
    Args:
        device: Target device (if None, uses get_device())
        
    Returns:
        torch.device context manager
    """
    if device is None:
        device = get_device()
    
    return torch.device(device)