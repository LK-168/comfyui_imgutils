import folder_paths
from segment_anything import sam_model_registry
from .util import SafeToGPU
from comfy import model_management
from segment_anything import SamPredictor
import numpy as np
import torch
import os
from .detect import BBOX
from .detect import BBOX_imagutils
from .util import SEG


model_path = folder_paths.models_dir

def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


class SAMWrapper:
    def __init__(self, model, safe_to_gpu,is_auto_mode):
        self.model = model
        self.safe_to_gpu = safe_to_gpu 
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        if self.is_auto_mode:
            device = model_management.get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)

class SAMLoaderLK:
    @classmethod
    def INPUT_TYPES(cls):
        models = [x for x in folder_paths.get_filename_list("sams")]


        return {
            "required": {
                "model_name": (models, {"tooltip": "The detection accuracy varies depending on the SAM model. ESAM can only be used if ComfyUI-YoloWorld-EfficientSAM is installed."}),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"], {"tooltip": "AUTO: Only applicable when a GPU is available. It temporarily loads the SAM_MODEL into VRAM only when the detection function is used.\n"
                                                                           "Prefer GPU: Tries to keep the SAM_MODEL on the GPU whenever possible. This can be used when there is sufficient VRAM available.\n"
                                                                           "CPU: Always loads only on the CPU."}),
            }
        }

    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "imgutils/sam"

    DESCRIPTION = "Load the SAM (Segment Anything) model."

    def load_model(self, model_name, device_mode="auto"):
        modelname = folder_paths.get_full_path("sams", model_name)

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint=modelname)

        size = os.path.getsize(modelname)
        safe_to = SafeToGPU(size)

        device = model_management.get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            safe_to.to_device(sam, "cuda")

        is_auto_mode = device_mode == "AUTO"

        sam_obj = SAMWrapper(sam,safe_to_gpu=safe_to,is_auto_mode=is_auto_mode)
        sam.sam_wrapper = sam_obj

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam, )


class SAMPredictorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",), # Input from SAMLoaderLK
                "image": ("IMAGE",),          # ComfyUI Image tensor (BCHW, float 0-1)
                "threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold to include masks"}),
                "bbox": (BBOX, {"optional": True, "tooltip": "Optional bounding box [x_min, y_min, x_max, y_max] (pixel space). Connect a MaskToBBoxNode output or similar."}),
                "points_method":(["None", "center-1","vertical-2", "horizontal-2", "rectangle-4", "center-corner-5", "diamond-4"],
                                    {"default": "None", "tooltip": "Method to generate points for the SAM model. 'None' means no points"}),
                "merge_options": (["Merge All","BBox Merge", "No Merge"], {"default": "Merge All", "tooltip": "How to merge masks if multiple are generated. 'Merge All' combines all masks, 'BBox Merge' merges masks within the same bbox, 'No Merge' returns all masks separately."}),
            },
            "optional": {
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1, "tooltip": "Factor to crop the image for SEG"}),
            }
        }

    RETURN_TYPES = ("MASK","SEGS") # Output is a ComfyUI Mask tensor (BH'W', float 0-1)

    FUNCTION = "predict_with_sam"
    CATEGORY = "imgutils/sam"
    DESCRIPTION = "Predicts segmentation masks using the SAM model based on prompts."

    def get_points(self, image,bbox, points_method="None"):
        if points_method == "None":
            return None, None

        points_coords = None

        x_min, y_min, x_max, y_max = bbox[:4]
        d_x = x_max - x_min
        d_y = y_max - y_min

        if points_method == "center-1":
            points_coords = [[(x_min + x_max) / 2, (y_min + y_max) / 2]]
            points_labels = [1]
        if points_method == "vertical-2":
            points_coords = [[(x_min + x_max) / 2, y_min + d_y // 3], [(x_min + x_max) / 2, y_max - d_y // 3]]
            points_labels = [1, 1]
        elif points_method == "horizontal-2":
            points_coords = [[x_min + d_x // 3, (y_min + y_max) / 2], [x_max - d_x // 3, (y_min + y_max) / 2]]
            points_labels = [1, 1]
        elif points_method == "rectangle-4":
            points_coords = [
                [x_min + d_x // 3, y_min + d_y // 3],
                [x_max - d_x // 3, y_min + d_y // 3],
                [x_max - d_x // 3, y_max - d_y // 3],
                [x_min + d_x // 3, y_max - d_y // 3]
            ]
            points_labels = [1, 1, 1, 1]
        elif points_method == "center-corner-5":
            points_coords = [
                [(x_min + x_max) / 2, (y_min + y_max) / 2],  # Center point
                [x_min + d_x // 3, y_min + d_y // 3],        # Top-left corner
                [x_max - d_x // 3, y_min + d_y // 3],        # Top-right corner
                [x_max - d_x // 3, y_max - d_y // 3],        # Bottom-right corner
                [x_min + d_x // 3, y_max - d_y // 3]         # Bottom-left corner
            ]
            points_labels = [1, 0, 0, 0, 0]
        elif points_method == "diamond-4":
            points_coords = [
                [(x_min + x_max) / 2, y_min + d_y // 3],  # Top point
                [x_max - d_x // 3, (y_min + y_max) / 2],  # Right point
                [(x_min + x_max) / 2, y_max - d_y // 3],  # Bottom point
                [x_min + d_x // 3, (y_min + y_max) / 2]   # Left point
            ]
            points_labels = [1, 1, 1, 1]
            
        return points_coords, points_labels
        

    def predict_with_sam(self, sam_model, image, threshold=0.4, 
                         bbox=None,points_method="None",merge_options="Merge All",
                         crop_factor=3.0,
                         ):
        # Input 'sam_model' is the original model object with the wrapper attached by SAMLoaderLK
        sam_wrapper = getattr(sam_model, 'sam_wrapper', None)
        if sam_wrapper is None or not isinstance(sam_wrapper, SAMWrapper):
             raise TypeError("Input 'sam_model' does not contain a valid SAMWrapper. Please ensure it comes from a compatible SAMLoader node.")

        # Image is a torch.Tensor [B, C, H, W], float [0, 1]
        # SAM Predictor expects numpy HWC [H, W, C], uint8 [0, 255], RGB
        # Assuming batch size B=1 for the image input

        print(f"Debug: Input image shape: {image.shape}")
        
        if image.shape[0] > 1:
             print(f"Warning: SAMPredictorNode received a batch of images ({image.shape[0]}). Processing the first image only.")

        # Take the first image from batch and ensure it's in the correct format
        single_image = image[0]  # Shape should be [C, H, W]
        # print(f"Debug: Single image shape after batch selection: {single_image.shape}")
        
        # Ensure we have 3 channels (RGB)
        if single_image.shape[-1] == 3:
            # Already RGB, convert to uint8 numpy HWC
            image_np_rgb = (single_image * 255.0).cpu().numpy().astype(np.uint8)
        elif single_image.shape[-1] == 1:
            # Grayscale [H, W, 1], convert to RGB [H, W, 3] and uint8 numpy
            single_image_rgb = single_image.repeat(1, 1, 3) # Repeat last dimension
            image_np_rgb = (single_image_rgb * 255.0).cpu().numpy().astype(np.uint8)
        else:
            # Unexpected number of channels, try to handle it
            # print(f"Warning: Unexpected number of channels in last dimension: {single_image.shape[-1]}. Trying to use first 3 channels.")
            if single_image.shape[-1] >= 3:
                 # Assuming first 3 are RGB, take them
                 single_image_rgb = single_image[:, :, :3]
                 image_np_rgb = (single_image_rgb * 255.0).cpu().numpy().astype(np.uint8)
            else:
                 raise ValueError(f"Cannot handle image with {single_image.shape[-1]} channels in last dimension")

        # print(f"Debug: Final image_np_rgb shape for SamPredictor: {image_np_rgb.shape}")

        img_w, img_h = image_np_rgb.shape[1], image_np_rgb.shape[0]

        all_masks_np = []
        bbox_input_list = []

        if bbox is not None:
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], BBOX_imagutils):
                    # bbox_input_list = [b.get_bbox() for b in bbox]
                    bbox_input_list = bbox
                    
            elif isinstance(bbox, BBOX_imagutils):
                # bbox_input_list = [bbox.get_bbox()]
                bbox_input_list = [bbox]
    
        sam_wrapper.prepare_device()  
        SEG_list = []

        try:
            # Only bboxes provided
            for single_bbox in bbox_input_list:

                single_bbox_loc = single_bbox.get_bbox() if isinstance(single_bbox, BBOX_imagutils) else single_bbox[:4]
                
                points_coords, points_labels = self.get_points(image_np_rgb, single_bbox_loc, points_method)

                mask_list_np = sam_wrapper.predict(
                    image_np_rgb,
                    points=points_coords,
                    plabs=points_labels,
                    bbox=single_bbox_loc,
                    threshold=threshold
                )
                x_min, y_min, x_max, y_max = single_bbox_loc[:4]
                d_x = x_max - x_min
                d_y = y_max - y_min
                
                cropped_region = [
                    max(0, int(x_min - d_x * (crop_factor - 1) / 2)),
                    max(0, int(y_min - d_y * (crop_factor - 1) / 2)),
                    min(img_w, int(x_max + d_x * (crop_factor - 1) / 2)),
                    min(img_h, int(y_max + d_y * (crop_factor - 1) / 2))
                ]
                cropped_image = image_np_rgb[
                    cropped_region[1]:cropped_region[3],
                    cropped_region[0]:cropped_region[2]
                ]

                if merge_options == "BBox Merge" or merge_options == "Merge All":
                    
                    if len(mask_list_np) > 0:
                        merged_mask = np.logical_or.reduce(mask_list_np)
                        if merge_options == "BBox Merge":
                            all_masks_np.append(merged_mask.astype(np.float32))
                        else: 
                            all_masks_np.extend(mask_list_np)
                        
                        seg = SEG(
                            cropped_image=cropped_image,  
                            cropped_mask=merged_mask.astype(np.float32),
                            confidence=single_bbox.confidence if isinstance(single_bbox, BBOX_imagutils) else threshold,
                            crop_region=cropped_region,
                            bbox=single_bbox_loc,
                            label=single_bbox.label if isinstance(single_bbox, BBOX_imagutils) else None,
                            control_net_wrapper=None
                        )
                        SEG_list.append(seg)
                        
                else:
                    all_masks_np.extend(mask_list_np)
                    for mask_np in mask_list_np:
                        # Create SEG object for each mask
                        seg = SEG(
                            cropped_image=cropped_image,  
                            cropped_mask=mask_np,
                            confidence=single_bbox.confidence if isinstance(single_bbox, BBOX_imagutils) else threshold,
                            crop_region=cropped_region,
                            bbox=single_bbox_loc,
                            label=single_bbox.label if isinstance(single_bbox, BBOX_imagutils) else None,
                            control_net_wrapper=None
                        )
                        SEG_list.append(seg)
        except Exception as e:
            print(f"Error during SAM prediction: {e}")
            all_masks_np = []
        finally:
            sam_wrapper.release_device()

        if not all_masks_np:
            original_h, original_w = image_np_rgb.shape[:2]
            return (torch.zeros((1, original_h, original_w), dtype=torch.float32),SEG_list)

        if merge_options == "Merge All":
            # Merge all masks into one
            merged_mask = np.logical_or.reduce(all_masks_np)
            masks_tensor = torch.from_numpy(merged_mask.astype(np.float32)).unsqueeze(0)
        
        else:
            # Convert list of masks to tensor
            masks_np = np.stack(all_masks_np, axis=0)
            masks_tensor = torch.from_numpy(masks_np.astype(np.float32))

        return (masks_tensor, SEG_list)




NODE_CLASS_MAPPINGS = {
    "SAMLoaderLK": SAMLoaderLK,
    "SAMPredictorNode": SAMPredictorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMLoaderLK": "SAM Loader for SAMPredictorNode",
    "SAMPredictorNode": "SAM Predictor",
}


