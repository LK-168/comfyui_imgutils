import torch
import numpy as np
from PIL import Image
# import cv2

from imgutils.segment import (
    segment_rgba_with_isnetis,
    segment_with_isnetis,
    get_isnetis_mask
)


class ImgutilsAutoSegmenter(object):
    """
    对输入的图像进行自动分割，输出前景图像和其蒙版。
    """
    CATEGORY = "imgutils/segmentation"

    @classmethod
    def INPUT_TYPES(cls):
        INPUT_TYPES = {
            "required": {
                "image": ("IMAGE",),
                "segment_mode": (["rgba_transparent", "rgb_white_bg", "rgb_black_bg"],),
                "scale": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 128}),
            },
        }
        return INPUT_TYPES

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented_image", "segment_mask")
    FUNCTION = "segment"

    def segment(self, image, segment_mode, scale):
        img_tensor = image[0]
        img_np_255 = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np_255).convert("RGB")

        output_image_pil = None
        output_mask_np = None

        try:
            if segment_mode == "rgba_transparent":
                seg_mask_np, seg_image_pil = segment_rgba_with_isnetis(img_pil, scale=scale)
                output_image_pil = seg_image_pil
                output_mask_np = seg_mask_np
            elif segment_mode == "rgb_white_bg":
                seg_mask_np, seg_image_pil = segment_with_isnetis(img_pil, scale=scale, background='white')
                output_image_pil = seg_image_pil
                output_mask_np = seg_mask_np
            elif segment_mode == "rgb_black_bg":
                seg_mask_np, seg_image_pil = segment_with_isnetis(img_pil, scale=scale, background='black')
                output_image_pil = seg_image_pil
                output_mask_np = seg_mask_np

        except Exception as e:
            print(f"Error during imgutils segmentation: {e}")
            output_image_pil = img_pil # Return original image on error
            # Ensure mask is correctly shaped if error occurs
            output_mask_np = np.zeros((img_pil.height, img_pil.width), dtype=np.float32)

        # Format output
        if output_image_pil:
            output_image_np_255 = np.array(output_image_pil.convert("RGB"))
            output_image_tensor = torch.from_numpy(output_image_np_255.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            output_image_tensor = image

        if output_mask_np is not None:
            if output_mask_np.dtype == np.uint8:
                output_mask_np = output_mask_np.astype(np.float32) / 255.0
            
            output_mask_tensor = torch.from_numpy(output_mask_np).unsqueeze(0)
        else:
            output_mask_tensor = torch.zeros((1, img_pil.height, img_pil.width), dtype=torch.float32)

        return (output_image_tensor, output_mask_tensor)

class ImgutilsBBoxSegmenter(object):
    """
    使用输入 BBox mask 来指导 imgutils 进行分割，并输出叠加结果。
    """
    CATEGORY = "imgutils/segmentation"

    @classmethod
    def INPUT_TYPES(cls):
        INPUT_TYPES = {
            "required": {
                "image": ("IMAGE",),
                "bbox_mask": ("MASK",),
                "segment_mode": (["rgba_transparent", "rgb_white_bg", "rgb_black_bg"],),
                "scale": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 128}),
            },
        }
        return INPUT_TYPES

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented_image_with_bbox", "segment_mask_from_bbox")
    FUNCTION = "segment_with_bbox_mask"

    def segment_with_bbox_mask(self, image, bbox_mask, segment_mode, scale):
        img_tensor = image[0]
        img_np_255 = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        original_image_pil = Image.fromarray(img_np_255).convert("RGB")

        bbox_mask_np_0_1 = bbox_mask[0].cpu().numpy() # MASK is [B, H, W], values 0-1
        bbox_mask_np = (bbox_mask_np_0_1 > 0.01).astype(np.float32)

        # If bbox_mask is empty, return original image and empty mask
        if not np.any(bbox_mask_np):
            print("BBox mask is empty. Returning original image and empty mask.")
            empty_mask_tensor = torch.zeros((1, original_image_pil.height, original_image_pil.width), dtype=torch.float32)
            return (image, empty_mask_tensor)

        # Find bounding box coordinates from the mask
        bbox_indices = np.where(bbox_mask_np > 0)
        if len(bbox_indices[0]) == 0:
            print("BBox mask has no valid pixels. Returning original image and empty mask.")
            empty_mask_tensor = torch.zeros((1, original_image_pil.height, original_image_pil.width), dtype=torch.float32)
            return (image, empty_mask_tensor)
        
        min_y, max_y = bbox_indices[0].min(), bbox_indices[0].max() + 1
        min_x, max_x = bbox_indices[1].min(), bbox_indices[1].max() + 1

        # Get the main subject's mask from imgutils (full image)
        seg_mask_np_full = get_isnetis_mask(original_image_pil, scale=scale)
        # Normalize mask to 0-1 float32
        if seg_mask_np_full.dtype == np.uint8:
            seg_mask_np_full = seg_mask_np_full.astype(np.float32) / 255.0
        # Ensure it's 0 or 1 for multiplication
        seg_mask_np_full = (seg_mask_np_full > 0.5).astype(np.float32)

        # Combine imgutils mask with provided bbox_mask
        final_seg_mask_np = seg_mask_np_full * bbox_mask_np

        # Get the segmented image based on mode (full image)
        if segment_mode == "rgba_transparent":
            _, seg_image_pil_full = segment_rgba_with_isnetis(original_image_pil, scale=scale)
            if seg_image_pil_full.mode != 'RGBA':
                seg_image_pil_full = seg_image_pil_full.convert('RGBA')
            
            # Apply the combined mask to the alpha channel of the segmented image
            alpha_from_mask = Image.fromarray((final_seg_mask_np * 255).astype(np.uint8), mode='L')
            seg_image_pil_full.putalpha(alpha_from_mask)
            full_output_image_pil = seg_image_pil_full

        else: # RGB modes
            bg_color_val = 0 if segment_mode == "rgb_black_bg" else 255
            background_image = Image.new('RGB', original_image_pil.size, (bg_color_val, bg_color_val, bg_color_val))
            
            # Get the segmented image with imgutils' background handling
            if segment_mode == "rgb_white_bg":
                _, seg_image_pil_full = segment_with_isnetis(original_image_pil, scale=scale, background='white')
            else: # rgb_black_bg
                _, seg_image_pil_full = segment_with_isnetis(original_image_pil, scale=scale, background='black')
            
            if seg_image_pil_full.mode != 'RGB':
                seg_image_pil_full = seg_image_pil_full.convert('RGB')
            
            # Paste the imgutils segmented image onto the background using the combined mask
            paste_mask_pil = Image.fromarray((final_seg_mask_np * 255).astype(np.uint8), mode='L')
            background_image.paste(seg_image_pil_full, (0, 0), paste_mask_pil)
            full_output_image_pil = background_image

        # Crop the segmented image to the bbox region
        cropped_output_image_pil = full_output_image_pil.crop((min_x, min_y, max_x, max_y))
        
        # Crop the mask to the bbox region as well
        cropped_seg_mask_np = final_seg_mask_np[min_y:max_y, min_x:max_x]

        # Format output
        if cropped_output_image_pil:
            output_image_np_255 = np.array(cropped_output_image_pil.convert("RGB"))
            output_image_tensor = torch.from_numpy(output_image_np_255.astype(np.float32) / 255.0).unsqueeze(0)
        else: # Fallback
            output_image_tensor = image

        # Output the cropped mask
        output_mask_tensor = torch.from_numpy(cropped_seg_mask_np).unsqueeze(0)

        return (output_image_tensor, output_mask_tensor)

NODE_CLASS_MAPPINGS = {
    "ImgutilsAutoSegmenter": ImgutilsAutoSegmenter,
    "ImgutilsBBoxSegmenter": ImgutilsBBoxSegmenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImgutilsAutoSegmenter": "Imgutils Auto Segmenter",
    "ImgutilsBBoxSegmenter": "Imgutils BBox Segmenter",
}