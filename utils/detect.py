import torch
import numpy as np
from PIL import Image
import cv2

from imgutils.detect import (
    detect_person, detect_faces, detect_heads, detect_halfbody,
    detect_hands, detect_eyes, detect_with_booru_yolo, detect_with_nudenet
)
from imgutils.detect.censor import detect_censors

from imgutils.detect.visual import detection_visualize


BBOX = "BBOX"
class BBOX_imagutils:
    """
    A class to represent bounding boxes in imgutils detection.
    This is a placeholder for future use if needed.
    """
    def __init__(self, x0, y0, x1, y1,img_w,img_h, label="", confidence=0.0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.img_w = img_w
        self.img_h = img_h
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return f"BBOX({self.x0}, {self.y0}, {self.x1}, {self.y1}, {self.img_w}, {self.img_h}, '{self.label}', {self.confidence})"

    def get_bbox(self):
        return (self.x0, self.y0, self.x1, self.y1) 
    
    def to_json(self):
        """Convert BBOX_imagutils to JSON serializable format"""
        return {
            "type": "BBOX_imagutils",
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "img_w": self.img_w,
            "img_h": self.img_h,
            "label": self.label,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_json(cls, data):
        """Create BBOX_imagutils from JSON data"""
        if isinstance(data, dict) and data.get("type") == "BBOX_imagutils":
            return cls(
                x0=data["x0"],
                y0=data["y0"],
                x1=data["x1"],
                y1=data["y1"],
                img_w=data.get("img_w", 0),
                img_h=data.get("img_h", 0),
                label=data.get("label", ""),
                confidence=data.get("confidence", 0.0)
            )
        return data
    
    def __dict__(self):
        """Support for dict() conversion"""
        return self.to_json()
       

DETECTION_FUNCTIONS = {
    "Person Detection": detect_person,
    "Face Detection": detect_faces,
    "Head Detection": detect_heads,
    "Halfbody Detection": detect_halfbody,
    "Hand Detection": detect_hands,
    "Eye Detection": detect_eyes,
    "Censor Detection": detect_censors,
    # "Booru YOLO Detection": detect_with_booru_yolo,
    # "Nude Detection": detect_with_nudenet,
}

MODEL_VERSIONS = {
    "Person Detection": ["v0", "v1", "v1.1"],
    "Face Detection": ["v0", "v1", "v1.3", "v1.4"],
    "Head Detection": [],
    "Halfbody Detection": ["v1"],
    "Hand Detection": ["v1"],
    "Eye Detection": ["v1"],
    "Censor Detection": ["v1"],
}

class ImgutilsGenericDetector:
    """
    A generic imgutils detector node that unifies various detection functions.
    """
    CATEGORY = "imgutils/detection"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_type": (list(DETECTION_FUNCTIONS.keys()),),
                "conf_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "draw_boxes": ("BOOLEAN", {"default": True}),
                # Model specific parameters, always present for simplicity in this generic node
                "level": ("STRING", {"default": "s", "options": ["n", "s"]}),
                "version": ("STRING", {"default": "v1.1", "options": ["v0", "v1", "v1.1"]}),
                # "model_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",BBOX, )
    RETURN_NAMES = ("image_with_boxes", "detection_mask","detection_results",)
    FUNCTION = "detect"

    def detect(self, image, detection_type, conf_threshold, 
               iou_threshold, draw_boxes, level, version, model_name=None):
        # Convert ComfyUI image tensor to PIL Image
        # image is in format [B, H, W, C] with values 0-1
        img_tensor = image[0]  # Get first image from batch
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # Convert to 0-255 range
        img_pil = Image.fromarray(img_np)
        
        # For visualization and mask generation, we still need numpy array
        img_np_rgb = np.array(img_pil.convert("RGB"))
        
        selected_detect_func = DETECTION_FUNCTIONS.get(detection_type)
        detection_results_formatted = []
        output_image_tensor = image  # Default to original image tensor
        output_mask_np = np.zeros((img_np_rgb.shape[0], img_np_rgb.shape[1]), dtype=np.float32)
        output_string = ""

        if not selected_detect_func:
            output_string = f"Error: Detection type '{detection_type}' not supported."
            output_mask_tensor = torch.from_numpy(output_mask_np).unsqueeze(0)  # Add batch dimension
            return (output_image_tensor, output_string, output_mask_tensor)

        func_params = {
            "image": img_pil,  # Use PIL Image directly for imgutils functions
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        }
        
        func_params["level"] = level

        if version in MODEL_VERSIONS[detection_type]:
            func_params["version"] = version
        if model_name is None or model_name == "":
            pass
        else:
            func_params["model_name"] = model_name

        # if 1 + 1 == 2:  
        try:
            # print(f"Running {detection_type} with parameters:")
            # print(f"  Original image type: {type(img_pil)}")
            # print(f"  Original image mode: {img_pil.mode}")
            # print(f"  Original image size: {img_pil.size}")
            
            # Ensure image is in RGB mode
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            print(f"Using detection function: {selected_detect_func.__name__}")
            
            # # Try different calling patterns based on the function
            # if detection_type in ["Face Detection", "Person Detection"]:
            #     # These functions might have different parameter requirements
            #     if detection_type == "Face Detection":
            #         # Try with minimal parameters first
            #         raw_detections = selected_detect_func(
            #             image=img_pil,
            #             conf_threshold=conf_threshold
            #         )
            #     else:  # Person Detection
            #         raw_detections = selected_detect_func(
            #             image=img_pil,
            #             level=level,
            #             version=version if version in MODEL_VERSIONS[detection_type] else "v1.1",
            #             conf_threshold=conf_threshold
            #         )
            # else:
            #     # For other detection types, use the standard parameters
            #     raw_detections = selected_detect_func(
            #         image=img_pil,
            #         level=level,
            #         conf_threshold=conf_threshold
            #     )

            if "version" in func_params:
                raw_detections = selected_detect_func(
                    image=img_pil,
                    level=level,
                    version=func_params["version"],
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
            else:
                raw_detections = selected_detect_func(
                    image=img_pil,
                    level=level,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
            img_w, img_h = img_pil.size

            for item in raw_detections:
                if isinstance(item, tuple) and len(item) == 3:
                    box, label, confidence = item
                    if isinstance(box, tuple) and len(box) == 4:
                        detection_results_formatted.append({
                            'box': [int(b) for b in box],
                            'label': str(label),
                            'confidence': float(confidence)
                        })
            bbox_list = []

            if detection_results_formatted:
                # output_string = f"{detection_type} Results:\n"
                # output_string += "\n".join([
                #     f"- {res['label']}: Confidence={res['confidence']:.2f}, Box={res['box']}"
                #     for res in detection_results_formatted if res['confidence'] >= conf_threshold # Apply conf_threshold for string output too
                # ])
                # print(f"Detection results: {output_string}")
                for res in detection_results_formatted:
                    if res['confidence'] >= conf_threshold:
                        bbox = BBOX_imagutils(
                            x0=res['box'][0],
                            y0=res['box'][1],
                            x1=res['box'][2],
                            y1=res['box'][3],
                            img_w=img_w,
                            img_h=img_h,
                            label=res['label'],
                            confidence=res['confidence']
                        )
                        bbox_list.append(bbox)
            else:
                print(f"No {detection_type} detected with confidence >= {conf_threshold}.")
                # output_string = f"No {detection_type} detected with confidence >= {conf_threshold}."


            if draw_boxes and detection_results_formatted:
                vis_input_for_detection_visualize = []
                for res in detection_results_formatted:
                    if res['confidence'] >= conf_threshold: 
                        vis_input_for_detection_visualize.append(
                            (tuple(res['box']), res['label'], res['confidence'])
                        )
                
                if vis_input_for_detection_visualize: # Only visualize if there are results after filtering
                    output_image_pil = detection_visualize(
                        image=img_pil,
                        detection=vis_input_for_detection_visualize,
                        fontsize=12,
                        no_label=False
                    )
                    # Convert PIL back to ComfyUI tensor format
                    output_image_np = np.array(output_image_pil.convert("RGB"))
                    output_image_tensor = torch.from_numpy(output_image_np.astype(np.float32) / 255.0).unsqueeze(0)
                else:
                    output_image_tensor = image # Use original image tensor

            else: # If not drawing or no results, use original image tensor
                output_image_tensor = image            # Generate mask
            for res in detection_results_formatted:
                if res['confidence'] >= conf_threshold:
                    box = res['box']
                    cv2.rectangle(output_mask_np, (box[0], box[1]), (box[2], box[3]), 1.0, -1)

            # Convert mask to ComfyUI tensor format [B, H, W]
            output_mask_tensor = torch.from_numpy(output_mask_np).unsqueeze(0)

            return (
                output_image_tensor,
                output_mask_tensor,
                bbox_list if bbox_list else [],
            )

        except Exception as e:
            print(f"Error during {detection_type} detection: {e}")
            # Error handling - return original image, error message, and empty mask
            # output_string = f"Error during '{detection_type}' detection: {e}"
            empty_mask = torch.zeros((1, img_np_rgb.shape[0], img_np_rgb.shape[1]), dtype=torch.float32)
            empty_bbox_list = []            
            return (image, empty_mask, empty_bbox_list)
        

class MaskToBBoxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold to binarize the mask before finding bbox"}),
                "combine": ("BOOLEAN", {"default": False, "tooltip": "If True, combine all mask regions into one bbox. If False, output separate bboxes for each connected component."}),
                "nms_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "IoU threshold for Non-Maximum Suppression. Higher values merge more overlapping boxes."})
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "mask_to_bbox"
    CATEGORY = "imgutils/bbox"
    DESCRIPTION = "Calculates the bounding box(es) of a mask. Can combine all regions or separate them with NMS."

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            box1, box2: [x_min, y_min, x_max, y_max]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def merge_boxes(self, box1, box2):
        """
        Merge two bounding boxes by taking the outer bounds.
        
        Args:
            box1, box2: [x_min, y_min, x_max, y_max]
            
        Returns:
            Merged bounding box [x_min, y_min, x_max, y_max]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        return [
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max)
        ]

    def apply_nms(self, bbox_list, nms_threshold):
        """
        Apply Non-Maximum Suppression to merge overlapping bounding boxes.
        
        Args:
            bbox_list: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
            nms_threshold: IoU threshold for merging
            
        Returns:
            List of merged bounding boxes
        """
        if len(bbox_list) <= 1:
            return bbox_list
        
        merged_boxes = []
        used = [False] * len(bbox_list)
        img_w = bbox_list[0].img_w if isinstance(bbox_list[0], BBOX_imagutils) else 1
        img_h = bbox_list[0].img_h if isinstance(bbox_list[0], BBOX_imagutils) else 1
        
        for i in range(len(bbox_list)):
            if used[i]:
                continue
                
            current_box = bbox_list[i]
            boxes_to_merge = [current_box]
            used[i] = True
            
            # Find all boxes that overlap with current box above threshold
            # for j in range(i + 1, len(bbox_list)):
            #     if used[j]:
            #         continue
                    
            #     iou = self.calculate_iou(current_box, bbox_list[j])
            #     if iou >= nms_threshold:
            #         boxes_to_merge.append(bbox_list[j])
            #         used[j] = True
            
            # # Merge all overlapping boxes
            # if len(boxes_to_merge) == 1:
            #     merged_boxes.append(current_box)
            # else:
            #     merged_box = boxes_to_merge[0]
            #     for box in boxes_to_merge[1:]:
            #         merged_box = self.merge_boxes(merged_box, box)
            #     merged_boxes.append(merged_box)

            for j in range(i + 1, len(bbox_list)):
                if used[j]:
                    continue

                # 取出坐标用于IoU计算
                iou = self.calculate_iou(current_box.get_bbox(), bbox_list[j].get_bbox())
                if iou >= nms_threshold:
                    boxes_to_merge.append(bbox_list[j])
                    used[j] = True

            # Merge all overlapping boxes
            if len(boxes_to_merge) == 1:
                merged_boxes.append(current_box)
            else:
                # 合并所有box，取最大外接框，label和confidence可自定义合并策略
                merged_box_coords = boxes_to_merge[0].get_bbox()
                label = boxes_to_merge[0].label
                confidence = max(box.confidence for box in boxes_to_merge)
                for box in boxes_to_merge[1:]:
                    merged_box_coords = self.merge_boxes(merged_box_coords, box.get_bbox())
                merged_box = BBOX_imagutils(
                    x0=merged_box_coords[0],
                    y0=merged_box_coords[1],
                    x1=merged_box_coords[2],
                    y1=merged_box_coords[3],
                    img_w=img_w,
                    img_h=img_h,
                    label=label,
                    confidence=confidence
                )
                merged_boxes.append(merged_box)
        
        return merged_boxes

    def mask_to_bbox(self, mask, threshold=0.5, combine=True, nms_threshold=0.5):
        if mask.shape[0] > 1:
            print(f"Warning: MaskToBBoxNode received a batch of masks ({mask.shape[0]}). Calculating bbox for the first mask only.")

        # Convert to boolean numpy array (H, W)
        mask_np = mask[0].squeeze(0).cpu().numpy() > threshold
        mask_w, mask_h = mask_np.shape

        if combine:
            # Original behavior - combine all mask regions into one bbox
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                print("Warning: Mask is empty or fully transparent, cannot calculate bbox.")
                bbox = [0, 0, 0, 0]
            else:
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            print(f"Calculated combined BBox: {bbox}")
            return (bbox,)
        
        else:
            from scipy import ndimage
            
            # Label connected components
            labeled_mask, num_components = ndimage.label(mask_np)
            
            if num_components == 0:
                print("Warning: Mask is empty or fully transparent, cannot calculate bbox.")
                bbox_list = []
            else:
                bbox_list = []
                
                for component_id in range(1, num_components + 1):
                    # Get mask for this component
                    component_mask = (labeled_mask == component_id)
                    
                    # Find bounding box for this component
                    rows = np.any(component_mask, axis=1)
                    cols = np.any(component_mask, axis=0)
                    
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        # bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                        # bbox = [int(x_min), int(y_min), int(x_max), int(y_max),""]
                        bbox = BBOX_imagutils(
                            x0=int(x_min),
                            y0=int(y_min),
                            x1=int(x_max),
                            y1=int(y_max),
                            img_w=mask_w,
                            img_h=mask_h,
                            label="",  # No label for generic bbox
                            confidence=1.0  # Confidence is 1.0 for mask-based bboxes
                        )
                        bbox_list.append(bbox)
                
                # Apply NMS to merge overlapping boxes
                if bbox_list and nms_threshold < 1.0:
                    bbox_list = self.apply_nms(bbox_list, nms_threshold)
                    print(f"After NMS (threshold={nms_threshold}): {len(bbox_list)} BBoxes remain")
            
            print(f"Final BBoxes: {bbox_list}")
            
            return (bbox_list,)

class BBoxToMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "image_shape": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "Height and width of the output mask"}),
                "merge_options": ("BOOLEAN", {"default": False, "tooltip": "If True, merge all bounding boxes into one mask. If False, create separate masks for each bbox."}),
            }
        }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "bbox_to_mask"

    CATEGORY = "imgutils/bbox"
    DESCRIPTION = "Converts bounding boxes to a binary mask."
    
    def bbox_to_mask(self, bbox, merge_options=False, image_shape=512):
        
        mask_list = []
        bbox_input_list = []
        if bbox is not None:
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], BBOX_imagutils):
                    bbox_input_list = [b.get_bbox() for b in bbox]
        elif isinstance(bbox, BBOX_imagutils):
            bbox_input_list = [bbox.get_bbox()]
        
        if len(bbox_input_list) == 0:
            print("Warning: No bounding boxes provided, returning empty mask.")
            return (torch.zeros((1, image_shape, image_shape), dtype=torch.float32),)
        
        mask_w, mask_h = image_shape, image_shape

        if merge_options:
            # make sure all boxes are in same image shape
            img_w, img_h = bbox_input_list[0].img_w, bbox_input_list[0].img_h
            for box in bbox_input_list:
                if box.img_w != img_w or box.img_h != img_h:
                    
                    print(f"Warning: Bounding boxes have different image shapes ({box.img_w}x{box.img_h}), using first box shape ({img_w}x{img_h}) for mask.")
                    break
            mask_w, mask_h = img_w, img_h
            
        for box in bbox_input_list:
            x0, y0, x1, y1 = box.get_bbox()
            # Ensure coordinates are within bounds
            x0 = max(0, min(x0, mask_w - 1))
            y0 = max(0, min(y0, mask_h - 1))
            x1 = max(0, min(x1, mask_w - 1))
            y1 = max(0, min(y1, mask_h - 1))

            mask = torch.zeros((1, mask_h, mask_w), dtype=torch.float32)
            mask[0, y0:y1, x0:x1] = 1.
            mask_list.append(mask)    

        if merge_options:
            # Merge all masks into one
            merged_mask = torch.zeros((1, mask_h, mask_w), dtype=torch.float32)
            for mask in mask_list:
                merged_mask += mask
            merged_mask = (merged_mask > 0).float()

            return (merged_mask,)
        else:
            return (mask_list,)

class BBoxFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": (BBOX,),                
                "labels": ("STRING", {"default": "", "tooltip": "Comma-separated list of labels to filter by. Leave empty to keep all labels."}),
                "include_labels": ("BOOLEAN", {"default": True, "tooltip": "If True, only keep bounding boxes with specified labels. If False, exclude these labels."}),
                "min_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000000.0, "step": 1.0, "tooltip": "Minimum area of bounding boxes to keep"}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum confidence of bounding boxes to keep"}),
            }
        }
    RETURN_TYPES = (BBOX,)
    RETURN_NAMES = ("filtered_bboxes",)
    FUNCTION = "filter_bboxes"

    CATEGORY = "imgutils/bbox"
    DESCRIPTION = "Filters bounding boxes based on area, confidence, and labels."
    def filter_bboxes(self, bboxes,include_labels, min_area=0.0, min_confidence=0.0, labels=""):
        
        filtered_bboxes = []
        # Labels 应该是一个逗号分隔的字符串，如果 labels 为空或者仅有空格，则表示不过滤标签
        if labels:
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        else:
            labels = None

        for bbox in bboxes:
            if isinstance(bbox, BBOX_imagutils):
                area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
                if area < min_area or bbox.confidence < min_confidence:
                    continue
                if include_labels:
                    if labels and bbox.label not in labels:
                        continue
                else:  # Exclude labels
                    if labels and bbox.label in labels:
                        continue
            
                filtered_bboxes.append(bbox)
            else:
                print(f"Warning: Skipping non-BBOX_imagutils object: {bbox}")
        
        return (filtered_bboxes,)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ImgutilsGenericDetector": ImgutilsGenericDetector,
    "MaskToBBoxNode": MaskToBBoxNode,
    "BBoxToMaskNode": BBoxToMaskNode,
    "BBoxFilter": BBoxFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImgutilsGenericDetector": "Imgutils Generic Detector",
    "MaskToBBoxNode": "Mask to BBox",
    "BBoxToMaskNode": "BBox to Mask",
    "BBoxFilter": "BBox Filter",
}


