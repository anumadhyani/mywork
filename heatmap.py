import base64
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter

from model import predict_embedding_p_ai_pil

GRID_SIZE = int(os.getenv("AI_HEATMAP_GRID_SIZE", "8"))
MAX_DIM = int(os.getenv("AI_HEATMAP_MAX_DIM", "448"))


def _masked_variant(image: Image.Image, box) -> Image.Image:
    variant = image.copy()
    x0, y0, x1, y1 = box
    patch = variant.crop(box).filter(ImageFilter.GaussianBlur(radius=12))
    variant.paste(patch, box)
    return variant


def generate_ai_heatmap(image_path: str, grid_size: int = GRID_SIZE):
    """Occlusion-sensitivity heatmap.

    Returns a dict with an overlay PNG (base64), the AI probability, and the
    most influential grid cells. Cells show where masking reduced the AI
    score — i.e., regions driving the prediction. Not proof of edits.
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM))
    w, h = img.size

    baseline = predict_embedding_p_ai_pil(img)
    if baseline is None:
        return None

    cell_w, cell_h = w / grid_size, h / grid_size
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)
    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (
                int(c * cell_w),
                int(r * cell_h),
                int((c + 1) * cell_w),
                int((r + 1) * cell_h),
            )
            p_masked = predict_embedding_p_ai_pil(_masked_variant(img, box))
            importance = max(0.0, baseline - (p_masked if p_masked is not None else baseline))
            heat[r, c] = importance
            cells.append({"row": r, "col": c, "box": list(box), "importance": round(float(importance), 4)})

    peak = float(heat.max())
    heat_norm = heat / peak if peak > 0 else heat
    heat_img = cv2.resize((heat_norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.55, colored, 0.45, 0)
    ok, png = cv2.imencode(".png", overlay)

    cells.sort(key=lambda c: c["importance"], reverse=True)
    return {
        "p_ai": baseline,
        "grid_size": grid_size,
        "width": w,
        "height": h,
        "peak_drop": round(peak, 4),
        "overlay_png_b64": base64.b64encode(png.tobytes()).decode() if ok else None,
        "top_regions": [c for c in cells if c["importance"] > 0][:8],
    }
