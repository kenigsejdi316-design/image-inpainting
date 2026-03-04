from .lol_dataset import LOLDataset
from .sidd_dataset import SIDDDataset
from .div2k_dataset import DIV2KDataset
from .places365_inpaint_dataset import Places365InpaintDataset
from .faces_dataset import FacesRestorationDataset


def build_dataset(task_name: str, task_cfg: dict, split: str = "train"):
    dataset_name = task_cfg["dataset"]

    if dataset_name == "lol":
        return LOLDataset(task_cfg, split=split)
    if dataset_name == "sidd":
        return SIDDDataset(task_cfg, split=split)
    if dataset_name == "div2k":
        return DIV2KDataset(task_cfg, split=split)
    if dataset_name == "places365":
        return Places365InpaintDataset(task_cfg, split=split)
    if dataset_name == "faces":
        return FacesRestorationDataset(task_cfg, split=split)

    raise ValueError(f"Unsupported dataset: {dataset_name}")
