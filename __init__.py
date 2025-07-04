from .utils.detect import NODE_CLASS_MAPPINGS as DETECT_NODE_CLASS_MAPPINGS
from .utils.detect import NODE_DISPLAY_NAME_MAPPINGS as DETECT_NODE_DISPLAY_NAME_MAPPINGS

from .utils.segment import NODE_CLASS_MAPPINGS as SEGMENT_NODE_CLASS_MAPPINGS
from .utils.segment import NODE_DISPLAY_NAME_MAPPINGS as SEGMENT_NODE_DISPLAY_NAME_MAPPINGS

from .utils.sam import NODE_CLASS_MAPPINGS as SAM_NODE_CLASS_MAPPINGS
from .utils.sam import NODE_DISPLAY_NAME_MAPPINGS as SAM_NODE_DISPLAY_NAME_MAPPINGS

from .utils.censor import NODE_CLASS_MAPPINGS as CENSOR_NODE_CLASS_MAPPINGS
from .utils.censor import NODE_DISPLAY_NAME_MAPPINGS as CENSOR_NODE_DISPLAY_NAME_MAPPINGS

from .utils.mask import NODE_CLASS_MAPPINGS as MASK_NODE_CLASS_MAPPINGS
from .utils.mask import NODE_DISPLAY_NAME_MAPPINGS as MASK_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **DETECT_NODE_CLASS_MAPPINGS,
    **SEGMENT_NODE_CLASS_MAPPINGS,
    **SAM_NODE_CLASS_MAPPINGS,
    **CENSOR_NODE_CLASS_MAPPINGS,
    **MASK_NODE_CLASS_MAPPINGS,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DETECT_NODE_DISPLAY_NAME_MAPPINGS,
    **SEGMENT_NODE_DISPLAY_NAME_MAPPINGS,
    **SAM_NODE_DISPLAY_NAME_MAPPINGS,
    **CENSOR_NODE_DISPLAY_NAME_MAPPINGS,
    **MASK_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']