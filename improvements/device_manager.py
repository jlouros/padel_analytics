"""
Improved device compatibility and resource management.
"""

import torch
import logging
from typing import Optional

class DeviceManager:
    """
    Robust device management for PyTorch operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._device = None
        self._device_capabilities = {}
        
    @property 
    def device(self) -> str:
        """Get the best available device."""
        if self._device is None:
            self._device = self._detect_best_device()
        return self._device
        
    def _detect_best_device(self) -> str:
        """Detect and return the best available device."""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                self.logger.info(f"CUDA available with {device_count} device(s)")
                
                # Check memory and compute capability
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    self.logger.info(f"CUDA device {i}: {props.name}, {memory_gb:.1f}GB")
                    
                    if memory_gb >= 2.0:  # Minimum memory requirement
                        return f"cuda:{i}"
                        
                return "cuda:0"  # Fallback to default CUDA device
                
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                self.logger.info("Using MPS (Apple Silicon) acceleration")
                return "mps"
                
            else:
                self.logger.info("Using CPU device")
                return "cpu"
                
        except Exception as e:
            self.logger.warning(f"Error detecting device: {e}, falling back to CPU")
            return "cpu"
            
    def move_to_device(self, tensor_or_model, device: Optional[str] = None):
        """Safely move tensor or model to device."""
        target_device = device or self.device
        
        try:
            return tensor_or_model.to(target_device)
        except Exception as e:
            self.logger.warning(f"Failed to move to {target_device}: {e}")
            # Fallback to CPU
            return tensor_or_model.to("cpu")
            
    def clear_cache(self):
        """Clear device cache if applicable."""
        try:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache")
            elif self.device == "mps":
                torch.mps.empty_cache()
                self.logger.debug("Cleared MPS cache")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")

# Global device manager instance
device_manager = DeviceManager()
