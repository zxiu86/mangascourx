from __future__ import annotations
import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Dict, Any

from .text_remove import TextRemovePipeline

class MangaCleanPipeline:
    """
    المايسترو الأكبر:
    تنظيف نويز الصفحة -> إزالة النصوص والفقاعات -> تبييض الخلفية وجعلها Production-Ready للطباعة أو القراءة.
    """
    def __init__(
        self,
        inpainting_method: str = "patchmatch",
        patch_size: int = 7,
        denoise_level: int = 0,
        whiten_background: bool = True
    ) -> None:
        self.text_remover = TextRemovePipeline(inpainting_method=inpainting_method, patch_size=patch_size)
        self.denoise_level = denoise_level
        self.whiten_background = whiten_background

    def _apply_adaptive_whitening(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        فيلتر ذكي لتبييض الأوراق الصفراء أو الرمادية بصفحات المانجا القديمة دون تدمير خطوط الرسم الحادة.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # استخدام عتبة تكيفية لمعرفة أين تقع الخلفية البيضاء المتسخة
        smooth = cv2.GaussianBlur(gray, (25, 25), 0)
        division = cv2.divide(gray, smooth, scale=255)
        
        # دمج النتيجة التكيفية لتعزيز بياض الخلفية وثبات سواد الحبر
        result = cv2.cvtColor(division, cv2.COLOR_GRAY2BGR)
        return result

    def run(self, image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        تنفيذ عمليات التطهير والترميم بالتوالي عبر الـ Pipeline.
        """
        current_img = image.copy()

        # المرحلة 1: معالجة أولية (Denoising) إذا طلب المستخدم لتقليل النويز قبل الفحص
        if self.denoise_level > 0:
            current_img = cv2.fastNdrMeansDenoisingColored(
                current_img, None, self.denoise_level, self.denoise_level, 7, 21
            )

        # المرحلة 2: تسليم الراية لإشارة مرور إزالة النصوص (Text Removal)
        text_remove_res = self.text_remover.run(current_img)
        cleaned_img = text_remove_res["result"]

        # المرحلة 3: تبييض وضبط تباين الصفحة النهائي للـ Manga Printing
        if self.whiten_background:
            cleaned_img = self._apply_adaptive_whitening(cleaned_img)

        return {
            "final_page": cleaned_img,
            "mask": text_remove_res["mask"],
            "text_removed": text_remove_res["text_detected"]
        }
