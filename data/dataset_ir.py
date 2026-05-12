
import os
from typing import Any, Dict

import torch.utils.data as data
import utils.utils_image as util
import glob


class DatasetIR(data.Dataset):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__()

        self.opt = opt_dataset

        if self.opt['phase'] == 'train':
            self.patch_size = self.opt['H_size']
        self.n_channels = opt_dataset['n_channels']

        self.name: str = os.path.basename(opt_dataset['dataroot_H'])


        if self.opt["real"] == "true":
            img_root = opt_dataset['dataroot_H']

            self.himg_paths = sorted(
                glob.glob(os.path.join(img_root, "*.[jpJP][pnPN]*[gG]")) +
                glob.glob(os.path.join(img_root, "*.[bB][mM][pP]")) +
                glob.glob(os.path.join(img_root, "*.[tT][iI][fF]*"))
            )
            self.himg_paths = [p for p in self.himg_paths if os.path.isfile(p)]
        else:
            self.himg_paths = util.get_subfolder_paths(opt_dataset['dataroot_H'])

        self.count = 0

    def __len__(self):
        return len(self.himg_paths)