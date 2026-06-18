from __future__ import annotations
import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Dict, Any, Optional

# استيراد شرطي المرور من المجلدات المتقفلة
from .detection.detection import DetectionOrchestrator
from .inpainting.patchmatch.engine import PatchMatchInpainter
from .inpainting.telea import TeleaInpainter

class TextRemovePipeline:
    """
    إشارة المرور الخاصة بحذف النصوص:
    صورة صفحة المانجا -> كشف النصوص/الفقاعات -> إنتاج ماسك ثنائي -> ترميم الفراغات.
    """
    def __init__(
        self,
        merge_priority: list[str] = ["text", "bubbles"],
        patch_size: int = 7,
        inpainting_method: str = "patchmatch"
    ) -> None:
        # تهيئة فاحص ومحدد النصوص والفقاعات
        self.detector = DetectionOrchestrator(merge_priority=merge_priority)
        self.inpainting_method = inpainting_method.lower()
        self.patch_size = patch_size

    def run(self, image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        تشغيل خط الإنتاج بالترتيب الهندسي الصارم.
        """
        if image is None or image.size == 0:
            raise ValueError("الصورة فارغة أو غير صالحة!")

        # الخطوة 1: الفحص والكشف (كلاس شرطي المرور مالت الـ detection)
        # هذا راح يشغل MSER/CRAFT والـ Bubbles ويدمجهم سوا
        detection_result = self.detector.run(image, enable_text=True, enable_bubbles=True)
        final_mask = detection_result["mask"]

        # الخطوة 2: فحص إذا ماكو أي نص، نرجع الصورة فوراً بدون تضييع وقت المعالج
        if np.sum(final_mask) == 0:
            return {
                "result": image.copy(),
                "mask": final_mask,
                "text_detected": False
            }

        # الخطوة 3: تمرير الإشارة إلى وحوش الـ Inpainting بناءً على اختيار المستخدم
        if self.inpainting_method == "patchmatch":
            inpainter = PatchMatchInpainter(patch_size=self.patch_size, knn=3, iterations=3)
            # الـ PatchMatch مالتنا يحب الصورة float والماسك bool
            reconstructed = inpainter.run(image, final_mask)
        elif self.inpainting_method == "telea":
            inpainter = TeleaInpainter(radius=5)
            reconstructed = inpainter.run(image, final_mask)
        else:
            # Fallback سريع بـ OpenCV لو الطريقة غير مدعومة
            reconstructed = cv2.inpaint(image, final_mask, 3, cv2.INPAINT_TELEA)

        return {
            "result": reconstructed,
            "mask": final_mask,
            "text_detected": True,
            "meta": detection_result
        }