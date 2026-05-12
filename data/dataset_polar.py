import random
import numpy as np
import torch
import glob
import os
import cv2
import h5py
import polanalyser as pa
import utils.utils_image as util
from .dataset_ir import DatasetIR
from typing import Dict, Any, Union


class DatasetPolar(DatasetIR):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__(opt_dataset)
    
        self.angle_patterns = {
            0: ['0', '_0', '-0'],
            45: ['45', '_45', '-45'],
            90: ['90', '_90', '-90'],
            135: ['135', '_135', '-135']
        }

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        himg_path = self.himg_paths[index]
        if self.opt['real']=="true":
            cpfa_single = cv2.imread(himg_path , cv2.IMREAD_GRAYSCALE)
            cpfa_single = cpfa_single.astype(np.float32) * (1.0 / 255.0)
            img_L=sig_12channel(cpfa_single)
            img_H=img_L

        else:
            if self.opt['img_type'] == "mat":
                img_H = self._load_mat_file(himg_path)
            else:
                img_H = self._load_image_folder_fast(himg_path)

            img_L = Gen_CPFA_fast(img_H)
        H, W = img_H.shape[:2]

        blur_level = torch.FloatTensor([25 / 255])

        if self.opt['phase'] == 'train':
            self.count += 1
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            rnd_h_end = rnd_h + self.patch_size
            rnd_w_end = rnd_w + self.patch_size

            patch_H = img_H[rnd_h:rnd_h_end, rnd_w:rnd_w_end]
            patch_L = img_L[rnd_h:rnd_h_end, rnd_w:rnd_w_end]

            img_H_tensor = util.single2tensor3(patch_H)
            img_L_tensor = util.single2tensor3(patch_L)
        else:
            img_H_tensor = util.single2tensor3(img_H)
            img_L_tensor = util.single2tensor3(img_L)

        return {
            'y': img_L_tensor,
            'y_gt': img_H_tensor,
            'sigma': blur_level.unsqueeze(1).unsqueeze(1),
            'hpath': himg_path,
        }

    def _load_mat_file(self, file_path: str) -> np.ndarray:
       
        with h5py.File(file_path, 'r') as f:
           
            img = np.concatenate([
                np.transpose(f['RGB_0'][()], (2, 1, 0)),
                np.transpose(f['RGB_45'][()], (2, 1, 0)),
                np.transpose(f['RGB_90'][()], (2, 1, 0)),
                np.transpose(f['RGB_135'][()], (2, 1, 0))
            ], axis=2)
        return img

    def _load_image_folder_fast(self, folder_path: str) -> np.ndarray:
     
        image_paths = glob.glob(os.path.join(folder_path, '*.[jpJP][pnPN]*[gG]'))

        filename_map = {}
        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            filename_map[filename] = img_path

        all_channels = []
        for angle, patterns in self.angle_patterns.items():
            img_path = None
            for pattern in patterns:
      
                if pattern in filename_map:
                    img_path = filename_map[pattern]
                    break
         
                for filename in filename_map:
                    if filename.endswith(pattern):
                        img_path = filename_map[filename]
                        break
                if img_path:
                    break

            if img_path:
         
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                all_channels.extend([r, g, b])

  
        combined_matrix = np.stack(all_channels, axis=-1)
        combined_matrix = combined_matrix.astype(np.float32) * (1.0 / 255.0)

        return combined_matrix


def Gen_CPFA_fast(input_image: np.ndarray) -> np.ndarray:

    H, W, C = input_image.shape
    output = np.zeros_like(input_image)


    pattern = np.array([
        [6, 3, 7, 4],
        [9, 0, 10, 1],
        [7, 4, 8, 5],
        [10, 1, 11, 2]
    ], dtype=np.uint8)

    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pattern_idx = pattern[rows % 4, cols % 4]

    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    pattern_flat = pattern_idx.flatten()

    output[rows_flat, cols_flat, pattern_flat] = input_image[rows_flat, cols_flat, pattern_flat]

    return output

def sig_12channel(I: np.ndarray):

    h, w = I.shape
    out_12 = np.zeros((h, w, 12), dtype=I.dtype)

    out_12[1::4, 1::4, 0] = I[1::4, 1::4]
    out_12[1::4, 3::4, 1] = I[1::4, 3::4]
    out_12[3::4, 1::4, 1] = I[3::4, 1::4]
    out_12[3::4, 3::4, 2] = I[3::4, 3::4]

    out_12[0::4, 1::4, 3] = I[0::4, 1::4]
    out_12[0::4, 3::4, 4] = I[0::4, 3::4]
    out_12[2::4, 1::4, 4] = I[2::4, 1::4]
    out_12[2::4, 3::4, 5] = I[2::4, 3::4]

    out_12[0::4, 0::4, 6] = I[0::4, 0::4]
    out_12[0::4, 2::4, 7] = I[0::4, 2::4]
    out_12[2::4, 0::4, 7] = I[2::4, 0::4]
    out_12[2::4, 2::4, 8] = I[2::4, 2::4]

    out_12[1::4, 0::4, 9] = I[1::4, 0::4]
    out_12[1::4, 2::4, 10] = I[1::4, 2::4]
    out_12[3::4, 0::4, 10] = I[3::4, 0::4]
    out_12[3::4, 2::4, 11] = I[3::4, 2::4]
    return out_12
