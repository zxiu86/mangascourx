# __init__.py for inpainting module
from .base import Inpainter
from .telea import TeleaInpainter
from .patchmatch.engine import PatchMatchInpainter

__all__ = [
    "Inpainter",
    "TeleaInpainter",
    "PatchMatchInpainter"
]