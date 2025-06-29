import torch
from comfy import model_management
from collections import namedtuple

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index

class SafeToGPU:
    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device,safe_scale=1.3):
        if is_same_device(device, 'cpu'):
            obj.to(device)
        else:
            if is_same_device(obj.device, 'cpu'):  # cpu to gpu
                model_management.free_memory(self.size * safe_scale, device)
                if model_management.get_free_memory(device) > self.size * safe_scale:
                    try:
                        obj.to(device)
                    except:
                        print(f"WARN: The model is not moved to the '{device}' due to insufficient memory. [1]")
                else:
                    print(f"WARN: The model is not moved to the '{device}' due to insufficient memory. [2]")
