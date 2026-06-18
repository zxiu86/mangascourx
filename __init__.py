"""
mangascourx - Advanced Multi-Scale PatchMatch & AI-Powered Text Removal Engine for Manga
"""

from ._version import __version__
from .manga_clean import MangaCleanPipeline
from .text_remove import TextRemovePipeline

__all__ = [
    "__version__",
    "MangaCleanPipeline",
    "TextRemovePipeline",
]
