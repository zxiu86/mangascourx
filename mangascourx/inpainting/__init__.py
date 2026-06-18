"""Inpainting package — PatchMatch, Telea, CoherenceTransport."""
from .patchmatch.engine import PatchMatchInpainter
from .telea import TeleaInpainter
from .coherence import CoherenceTransport

__all__ = ["PatchMatchInpainter", "TeleaInpainter", "CoherenceTransport"]
