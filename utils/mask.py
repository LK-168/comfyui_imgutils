import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from scipy import ndimage

class MaskMorphologyNode:
    """Mask morphology operations: Dilate, Erode, Opening, Closing"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (["dilate", "erode", "opening", "closing"],),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 51, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "kernel_shape": (["ellipse", "rectangle", "cross"], {"default": "ellipse"}),
            }
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "apply_morphology"

    def apply_morphology(self, mask, operation, kernel_size, iterations, kernel_shape="ellipse"):
        batch_size = mask.shape[0]
        results = []
        
        # Create kernel
        if kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        
        for i in range(batch_size):
            current_mask = mask[i].squeeze().cpu().numpy()
            
            # Convert to uint8 for cv2
            mask_uint8 = (current_mask * 255).astype(np.uint8)
            
            # Apply morphological operation
            if operation == "dilate":
                result = cv2.dilate(mask_uint8, kernel, iterations=iterations)
            elif operation == "erode":
                result = cv2.erode(mask_uint8, kernel, iterations=iterations)
            elif operation == "opening":
                result = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == "closing":
                result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Convert back to float tensor
            result_float = result.astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_float))
        
        return (torch.stack(results),)


class MaskEdgeNode:
    """Mask edge operations: expand/contract mask boundaries"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (["expand", "contract", "expand_contract", "contract_expand"],),
                "pixels": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
            "optional": {
                "feather": ("INT", {"default": 0, "min": 0, "max": 50, "tooltip": "Soften edges after operation"}),
                "preserve_original": ("BOOLEAN", {"default": False, "tooltip": "Keep original mask areas"}),
            }
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "adjust_edges"

    def adjust_edges(self, mask, operation, pixels, feather=0, preserve_original=False):
        batch_size = mask.shape[0]
        results = []
        
        # Create circular kernel for smooth expansion/contraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        
        for i in range(batch_size):
            current_mask = mask[i].squeeze().cpu().numpy()
            original_mask = current_mask.copy()
            
            # Convert to uint8 for cv2
            mask_uint8 = (current_mask * 255).astype(np.uint8)
            
            # Apply edge operation
            if operation == "expand":
                result = cv2.dilate(mask_uint8, kernel, iterations=1)
            elif operation == "contract":
                result = cv2.erode(mask_uint8, kernel, iterations=1)
            elif operation == "expand_contract":
                # Expand then contract
                temp = cv2.dilate(mask_uint8, kernel, iterations=1)
                result = cv2.erode(temp, kernel, iterations=1)
            elif operation == "contract_expand":
                # Contract then expand
                temp = cv2.erode(mask_uint8, kernel, iterations=1)
                result = cv2.dilate(temp, kernel, iterations=1)
            
            # Convert back to float
            result_float = result.astype(np.float32) / 255.0
            
            # Preserve original mask areas if requested
            if preserve_original:
                result_float = np.maximum(result_float, original_mask)
            
            # Apply feathering if requested
            if feather > 0:
                result_float = self._apply_feather(result_float, feather)
            
            results.append(torch.from_numpy(result_float))
        
        return (torch.stack(results),)
    
    def _apply_feather(self, mask, feather_pixels):
        """Apply gaussian blur for feathering effect"""
        # Use scipy for gaussian filter
        sigma = feather_pixels / 3.0  # Convert pixels to sigma
        return ndimage.gaussian_filter(mask, sigma=sigma)


class MaskAttributeNode:
    """Mask attribute adjustments: binarization, thresholding, smoothing"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for binarization"}),
                "smooth_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Gaussian blur radius for smoothing"}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "tooltip": "Adjust mask contrast"}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Adjust mask brightness"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Invert mask"}),
                "binarize": ("BOOLEAN", {"default": False, "tooltip": "Convert to binary mask"}),
            }
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "adjust_attributes"

    def adjust_attributes(self, mask, threshold=0.5, smooth_radius=0.0, contrast=1.0, 
                         brightness=0.0, invert=False, binarize=False):
        batch_size = mask.shape[0]
        results = []
        
        for i in range(batch_size):
            current_mask = mask[i].squeeze().cpu().numpy().astype(np.float32)
            
            # Apply brightness adjustment
            if brightness != 0.0:
                current_mask = np.clip(current_mask + brightness, 0.0, 1.0)
            
            # Apply contrast adjustment
            if contrast != 1.0:
                # Contrast around 0.5 midpoint
                current_mask = np.clip(((current_mask - 0.5) * contrast) + 0.5, 0.0, 1.0)
            
            # Apply smoothing (feathering)
            if smooth_radius > 0:
                sigma = smooth_radius / 3.0
                current_mask = ndimage.gaussian_filter(current_mask, sigma=sigma)
                current_mask = np.clip(current_mask, 0.0, 1.0)
            
            # Apply binarization
            if binarize:
                current_mask = (current_mask >= threshold).astype(np.float32)
            
            # Apply inversion
            if invert:
                current_mask = 1.0 - current_mask
            
            results.append(torch.from_numpy(current_mask))
        
        return (torch.stack(results),)


class MaskCombineNode:
    """Combine multiple masks with various operations"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "operation": (["add", "subtract", "multiply", "divide", "max", "min", "xor"],),
            },
            "optional": {
                "mask1_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask2_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "clamp_result": ("BOOLEAN", {"default": True, "tooltip": "Clamp result to [0,1] range"}),
            }
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine_masks"

    def combine_masks(self, mask1, mask2, operation, mask1_weight=1.0, mask2_weight=1.0, clamp_result=True):
        # Handle different batch sizes
        batch_size = max(mask1.shape[0], mask2.shape[0])
        results = []
        
        for i in range(batch_size):
            # Get current masks
            current_mask1 = mask1[i] if i < mask1.shape[0] else mask1[0]
            current_mask2 = mask2[i] if i < mask2.shape[0] else mask2[0]
            
            # Convert to numpy and apply weights
            m1 = current_mask1.squeeze().cpu().numpy() * mask1_weight
            m2 = current_mask2.squeeze().cpu().numpy() * mask2_weight
            
            # Ensure same shape
            if m1.shape != m2.shape:
                # Resize smaller mask to match larger one
                if m1.size < m2.size:
                    m1 = cv2.resize(m1, (m2.shape[1], m2.shape[0]))
                else:
                    m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]))
            
            # Perform operation
            if operation == "add":
                result = m1 + m2
            elif operation == "subtract":
                result = m1 - m2
            elif operation == "multiply":
                result = m1 * m2
            elif operation == "divide":
                result = np.divide(m1, m2 + 1e-8)  # Avoid division by zero
            elif operation == "max":
                result = np.maximum(m1, m2)
            elif operation == "min":
                result = np.minimum(m1, m2)
            elif operation == "xor":
                # XOR operation: (A + B) - 2*(A * B)
                result = (m1 + m2) - 2 * (m1 * m2)
            
            # Clamp result to valid range
            if clamp_result:
                result = np.clip(result, 0.0, 1.0)
            
            results.append(torch.from_numpy(result.astype(np.float32)))
        
        return (torch.stack(results),)


class MaskInfoNode:
    """Display mask information and statistics"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "get_mask_info"

    def get_mask_info(self, mask):
        batch_size = mask.shape[0]
        info_list = []
        
        for i in range(batch_size):
            current_mask = mask[i].squeeze().cpu().numpy()
            
            # Calculate statistics
            shape = current_mask.shape
            total_pixels = current_mask.size
            white_pixels = np.sum(current_mask > 0.5)
            coverage = (white_pixels / total_pixels) * 100
            
            min_val = np.min(current_mask)
            max_val = np.max(current_mask)
            mean_val = np.mean(current_mask)
            
            info = f"Batch {i}: Shape={shape}, Coverage={coverage:.1f}%, "
            info += f"Range=[{min_val:.3f}, {max_val:.3f}], Mean={mean_val:.3f}"
            info_list.append(info)
        
        combined_info = "\n".join(info_list)
        return (mask, combined_info)

class MaskHelperLK:
    """Helper class for above mask operations"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "language": (["en", "zh"], {"default": "en", "tooltip": "Select language for node descriptions"}),
            },        
        }
    
    CATEGORY = "imgutils/mask"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("readme",)
    
    FUNCTION = "print"

    def print(self, language):
        readme = ""
        if language == "zh":
            readme += "这是一个mask处理的工具包，包含了多种常用的mask操作节点。\n"
            readme += "它可以帮助你进行mask的形态学操作(Mask Morphology)、边缘处理(Mask Edge Operations)、属性调整(Mask Attributes)以及多mask合并(Mask Combine)。\n"
            readme += "每个节点的功能如下：\n"
            readme += "- Mask Morphology: 形态学操作，如膨胀、腐蚀、开运算和闭运算。\n"
            readme += "- Mask Edge Operations: 边缘处理操作，如扩展、收缩、扩展收缩和收缩扩展。\n"
            readme += "- Mask Attributes: 属性调整，如二值化、阈值处理、平滑处理、对比度和亮度调整、反转和二值化。\n"
            readme += "- Mask Combine: 多mask合并操作，如加法、减法、乘法、除法、最大值、最小值和异或操作。\n"
            readme += "此外，还有一个Mask Info节点，可以显示mask的统计信息，如形状、覆盖率、范围和均值。\n"
            readme += "由于写的功能太杂了，所以又写了这个节点方便查看。\n"
            readme += "\n"
            readme += "我不知道写什么了就先卖个萌吧(｡•̀ᴗ-)✧\n"

        else:
            readme += "This is a mask processing toolkit that includes various commonly used mask operation nodes."
            readme += "It can help you perform mask morphology operations, edge processing, attribute adjustments, and multi-mask combinations.\n"
            readme += "Each node has the following functions:\n"
            readme += "- Mask Morphology: Morphological operations such as dilation, erosion, opening, and closing.\n"
            readme += "- Mask Edge Operations: Edge processing operations such as expand, contract, expand_contract, and contract_expand.\n"
            readme += "- Mask Attributes: Attribute adjustments such as binarization, thresholding, smoothing, contrast and brightness adjustments, inversion, and binarization.\n"
            readme += "- Mask Combine: Multi-mask combination operations such as addition, subtraction, multiplication, division, max, min, and xor operations.\n"
            readme += "Additionally, there is a Mask Info node that displays mask statistics such as shape, coverage, range, and mean value.\n"
            readme += "Since the functionality is too diverse, this node is provided for easy viewing.\n"
        
        return (readme,)

# Register all nodes
NODE_CLASS_MAPPINGS = {
    "MaskMorphologyNodeLK": MaskMorphologyNode,
    "MaskEdgeNodeLK": MaskEdgeNode,
    "MaskAttributeNodeLK": MaskAttributeNode,
    "MaskCombineNodeLK": MaskCombineNode,
    "MaskInfoNodeLK": MaskInfoNode,
    "MaskHelperLK": MaskHelperLK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMorphologyNodeLK": "Mask Morphology",
    "MaskEdgeNodeLK": "Mask Edge Operations",
    "MaskAttributeNodeLK": "Mask Attributes",
    "MaskCombineNodeLK": "Mask Combine",
    "MaskInfoNodeLK": "Mask Info",
    "MaskHelperLK": "Mask Helper LK",
}