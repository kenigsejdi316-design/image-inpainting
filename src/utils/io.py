from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    images = [p for p in folder_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(images)
