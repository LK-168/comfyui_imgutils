import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2

class CensorWithMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "censor_mode": (["blur", "pixelate", "color"],),
            },
            "optional": {
                # Blur mode parameters
                "blur_radius": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "tooltip": "Blur radius for blur mode"}),
                
                # Pixelate mode parameters
                "pixelate_size": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Pixel block size for pixelate mode"}),
                
                # Color mode parameters
                "color_hex": ("STRING", {"default": "#000000", "tooltip": "Hex color for color mode (e.g., #FF0000 for red)"}),
                "color_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "Opacity/transparency for color mode"}),
            }
        }
    
    CATEGORY = "imgutils/censor"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("censored_image",)
    FUNCTION = "censor_with_mask"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (0, 0, 0)  # Default to black if invalid

    def censor_with_mask(self, image, mask, censor_mode, blur_radius=5.0, pixelate_size=10, color_hex="#000000", color_opacity=1.0):
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        results = []
        
        for i in range(batch_size):
            # Handle different batch sizes for image and mask
            current_image = image[i]
            current_mask = mask[i] if mask_batch_size > 1 else mask[0]
            
            # Convert to numpy arrays
            image_np_rgb = (current_image.cpu().numpy() * 255.0).astype(np.uint8)
            mask_np = current_mask.squeeze().cpu().numpy()
            
            # Convert to PIL
            mask_pil = Image.fromarray((mask_np * 255.0).astype(np.uint8), 'L')
            image_pil = Image.fromarray(image_np_rgb, 'RGB')
            
            # Apply censoring based on mode
            if censor_mode == "blur":
                censored_image_pil = self._apply_blur(image_pil, mask_pil, blur_radius)
            elif censor_mode == "pixelate":
                censored_image_pil = self._apply_pixelate(image_pil, mask_pil, pixelate_size)
            elif censor_mode == "color":
                censored_image_pil = self._apply_color(image_pil, mask_pil, color_opacity, color_hex)
            else:
                censored_image_pil = image_pil  # Fallback
            
            # Convert back to tensor
            censored_image_np = np.array(censored_image_pil).astype(np.float32) / 255.0
            censored_image_tensor = torch.from_numpy(censored_image_np)
            results.append(censored_image_tensor)
        
        # Stack all results
        final_result = torch.stack(results)
        return (final_result,)

    def _apply_blur(self, image_pil, mask_pil, blur_radius):
        """Apply gaussian blur with mask"""
        radius = max(1, int(blur_radius))
        blurred_image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius))
        return Image.composite(blurred_image_pil, image_pil, mask_pil)

    def _apply_pixelate(self, image_pil, mask_pil, pixelate_size):
        """Apply pixelation with mask"""
        block_size = max(1, int(pixelate_size))
        width, height = image_pil.size
        
        # Calculate new dimensions
        small_width = max(1, width // block_size)
        small_height = max(1, height // block_size)
        
        # Create pixelated version
        pixelated_image_pil = image_pil.resize((small_width, small_height), Image.NEAREST)
        pixelated_image_pil = pixelated_image_pil.resize((width, height), Image.NEAREST)
        
        return Image.composite(pixelated_image_pil, image_pil, mask_pil)

    def _apply_color(self, image_pil, mask_pil, color_opacity, color_hex):
        """Apply solid color fill with mask"""
        color_rgb = self.hex_to_rgb(color_hex)
        
        color_layer = Image.new('RGB', image_pil.size, color_rgb)
        
        # Use color_opacity directly as it's already normalized (0.0-1.0)
        alpha_intensity = np.clip(color_opacity, 0.0, 1.0)
        mask_array = np.array(mask_pil) * alpha_intensity
        adjusted_mask = Image.fromarray(mask_array.astype(np.uint8), 'L')
        
        return Image.composite(color_layer, image_pil, adjusted_mask)

NODE_CLASS_MAPPINGS = {
    "CensorWithMask": CensorWithMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CensorWithMask": "Censor with Mask",
}