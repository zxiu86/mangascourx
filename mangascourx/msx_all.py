"""
msx_all.py — كل أدوات mangascourx في مكان واحد

الاستخدام:
    import mangascourx.msx_all as msx

    result = msx.TextRemovePipeline().run(image)
    mask   = msx.detect_bubbles(binary_mask)
    boxes  = msx.detect_text_regions(image)
    out    = msx.TeleaInpainter(radius=5).run(image, mask)
    clean  = msx.clean_noise(mask, open_kernel_size=3)
    pq     = msx.PriorityQueue(capacity=1000)
    ...

كل شيء في المكتبة يمكن استدعاؤه مباشرة من هنا.
"""

# ─── الأنابيب الرئيسية ───────────────────────────────────────────────────────
from .manga_clean import MangaCleanPipeline
from .text_remove import TextRemovePipeline

# ─── Detection ───────────────────────────────────────────────────────────────
from .detection import DetectionOrchestrator
from .detection.detection import Detector          # backward-compat alias

# mask merging / processing
from .detection.mask import (
    merge_labeled,
    merge_binary,
    cleanup_mask,
    process_masks,
)

# bubble tools
from .detection.bubbles.contours import (
    find_contours,
    filter_contours_by_area,
    get_convex_hulls,
    get_bounding_rects,
    get_bounding_circles,
    detect_bubbles,
    draw_contours,
    draw_convex_hulls,
    draw_bounding_rects,
    draw_bounding_circles,
)
from .detection.bubbles.morphology import (
    dilate,
    erode,
    open_mask,
    close_mask,
    clean_noise,
    improve_mask,
    apply_morphology,
)

# text tools
from .detection.text.mser import (
    detect_text_regions,
    create_mser,
    non_max_suppression,
    draw_regions,
)
from .detection.text.craft_adapter import CRAFTDetector
from .detection.text.swt import (
    stroke_width_transform,
    swt_to_mask,
    detect_text_swt,
)

# detector base
from .detection.base import BaseDetector, validate_image

# ─── Inpainting ──────────────────────────────────────────────────────────────
from .inpainting import PatchMatchInpainter, TeleaInpainter, CoherenceTransport
from .inpainting.base import Inpainter

# ─── Core math ───────────────────────────────────────────────────────────────
from .core import (
    euclidean_distance_transform,
    connected_components,
    structure_tensor,
    perona_malik_diffusion,
    curvature_diffusion,
    PriorityQueue,
)

# ─── Version ─────────────────────────────────────────────────────────────────
from ._version import __version__

# ─── __all__ (everything exported) ───────────────────────────────────────────
__all__ = [
    # pipelines
    "MangaCleanPipeline",
    "TextRemovePipeline",

    # detection orchestration
    "DetectionOrchestrator",
    "Detector",

    # mask ops
    "merge_labeled",
    "merge_binary",
    "cleanup_mask",
    "process_masks",

    # bubble contours
    "find_contours",
    "filter_contours_by_area",
    "get_convex_hulls",
    "get_bounding_rects",
    "get_bounding_circles",
    "detect_bubbles",
    "draw_contours",
    "draw_convex_hulls",
    "draw_bounding_rects",
    "draw_bounding_circles",

    # morphology
    "dilate",
    "erode",
    "open_mask",
    "close_mask",
    "clean_noise",
    "improve_mask",
    "apply_morphology",

    # text detection
    "detect_text_regions",
    "create_mser",
    "non_max_suppression",
    "draw_regions",
    "CRAFTDetector",
    "stroke_width_transform",
    "swt_to_mask",
    "detect_text_swt",

    # detector base
    "BaseDetector",
    "validate_image",

    # inpainting
    "Inpainter",
    "PatchMatchInpainter",
    "TeleaInpainter",
    "CoherenceTransport",

    # core math
    "euclidean_distance_transform",
    "connected_components",
    "structure_tensor",
    "perona_malik_diffusion",
    "curvature_diffusion",
    "PriorityQueue",

    # meta
    "__version__",
]
