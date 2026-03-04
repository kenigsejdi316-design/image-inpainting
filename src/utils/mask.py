import random
import numpy as np


def _random_irregular_mask(h: int, w: int, num_strokes: int = 8) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_strokes):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        length = random.randint(20, 80)
        angle = random.uniform(0, 2 * np.pi)
        brush_w = random.randint(8, 24)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        rr = np.linspace(y1, y2, num=100).astype(np.int32)
        cc = np.linspace(x1, x2, num=100).astype(np.int32)
        for r, c in zip(rr, cc):
            r1 = max(0, r - brush_w // 2)
            r2 = min(h, r + brush_w // 2)
            c1 = max(0, c - brush_w // 2)
            c2 = min(w, c + brush_w // 2)
            mask[r1:r2, c1:c2] = 1
    return mask


def _random_rectangle_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    num_rects = random.randint(1, 3)
    for _ in range(num_rects):
        rect_w = random.randint(max(16, w // 10), max(24, w // 3))
        rect_h = random.randint(max(16, h // 10), max(24, h // 3))
        x1 = random.randint(0, max(0, w - rect_w))
        y1 = random.randint(0, max(0, h - rect_h))
        mask[y1:y1 + rect_h, x1:x1 + rect_w] = 1
    return mask


def _random_horizontal_band_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    band_h = random.randint(max(10, h // 20), max(16, h // 7))
    band_w = random.randint(max(w // 5, 20), max(w * 4 // 5, 24))
    x1 = random.randint(0, max(0, w - band_w))
    y1 = random.randint(0, max(0, h - band_h))
    mask[y1:y1 + band_h, x1:x1 + band_w] = 1
    return mask


def _random_text_like_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    rows = random.randint(1, 2)
    for _ in range(rows):
        y = random.randint(0, max(0, h - max(10, h // 8)))
        num_blocks = random.randint(5, 14)
        cursor = random.randint(0, max(0, w // 6))
        block_h = random.randint(max(8, h // 28), max(12, h // 14))
        for _ in range(num_blocks):
            bw = random.randint(max(6, w // 40), max(10, w // 18))
            if cursor + bw >= w:
                break
            mask[y:y + block_h, cursor:cursor + bw] = 1
            cursor += bw + random.randint(2, 8)
    return mask


def random_mask(h: int, w: int, num_strokes: int = 8, mode: str = "mixed") -> np.ndarray:
    mode = (mode or "mixed").lower()

    if mode in {"random_irregular", "irregular"}:
        return _random_irregular_mask(h, w, num_strokes=num_strokes)
    if mode in {"rectangle", "rect"}:
        return _random_rectangle_mask(h, w)
    if mode in {"horizontal_band", "band"}:
        return _random_horizontal_band_mask(h, w)
    if mode in {"text", "text_like"}:
        return _random_text_like_mask(h, w)

    candidates = [
        _random_irregular_mask(h, w, num_strokes=num_strokes),
        _random_rectangle_mask(h, w),
        _random_horizontal_band_mask(h, w),
        _random_text_like_mask(h, w),
    ]
    probs = [0.45, 0.2, 0.2, 0.15]
    choice = random.choices(range(len(candidates)), weights=probs, k=1)[0]
    return candidates[choice]
