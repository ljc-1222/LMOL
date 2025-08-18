# -*- coding: utf-8 -*-
# All comments must be in English.

from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from torch.utils.data import Dataset
from PIL import Image

from configs import config
from data.data_utils import PairRecord, read_pairs_csv
from utils.transform import basic_image_loader

@dataclass
class Item:
    img1: Image.Image
    img2: Image.Image
    label: str  # "First." | "Second." | "Similar."

class SCUT_FBP5500_Pairs(Dataset):
    """
    Dataset returning two facial images and a tri-class label for LMOL order learning.
    - Images are loaded as PIL (RGB).
    - Any resizing/normalization is handled later by LLaVA's image_processor in the collator.
    """
    def __init__(self, pairs_csv: str, loader: Optional[Callable] = None):
        super().__init__()
        self.pairs: List[PairRecord] = read_pairs_csv(pairs_csv)
        self.loader = loader if loader is not None else basic_image_loader()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Item:
        p = self.pairs[idx]
        img1 = self.loader(p.img1)
        img2 = self.loader(p.img2)
        return Item(img1=img1, img2=img2, label=p.label)
