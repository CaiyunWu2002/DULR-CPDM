"""
Compute PSNR / SSIM / MAE (AoP angular error) using pyiqa.

The rest of the pipeline, including image loading, polarization parameter
extraction, and Excel export, remains unchanged.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import pyiqa
import torch


# ---------- 1. Initialize pyiqa metrics ---------- #
# Automatically uses CUDA if available, otherwise CPU.
psnr_metric = pyiqa.create_metric("psnr", test_y_channel=False)   # RGB images
ssim_metric = pyiqa.create_metric("ssim", test_y_channel=False)

# pyiqa does not provide an angular-error metric, so AoP MAE is computed manually.
# A custom pyiqa metric can be registered in the future if needed.


# ---------- 2. Utility functions ---------- #
def load_images(cal_folder):
    """Return the normalized 5-D ndarray, folder-to-index mapping, and folder list."""
    image_types = ["0", "45", "90", "135"]
    num_types = len(image_types)

    subfolders = sorted([
        f for f in os.listdir(cal_folder)
        if os.path.isdir(os.path.join(cal_folder, f))
    ])
    folder_to_index = {f: i for i, f in enumerate(subfolders)}

    # Use one sample image to determine the image size.
    sample_path = os.path.join(cal_folder, subfolders[0])
    for t in image_types:
        fp = os.path.join(sample_path, f"{t}.png")
        if os.path.exists(fp):
            with Image.open(fp) as im:
                arr = np.array(im)
                h = arr.shape[0]
                w = arr.shape[1]
                c = arr.shape[2] if arr.ndim == 3 else 1
            break
    else:
        raise ValueError("No valid sample image was found.")

    result_mat = np.zeros(
        (len(subfolders), h, w, c, num_types),
        dtype=np.float32
    )

    for img_idx, subdir in enumerate(subfolders):
        sub_path = os.path.join(cal_folder, subdir)
        for type_idx, t in enumerate(image_types):
            fp = os.path.join(sub_path, f"{t}.png")
            if not os.path.exists(fp):
                print(f"Warning: missing file {fp}")
                continue

            im = Image.open(fp)
            arr = np.asarray(im, dtype=np.uint8)

            if arr.ndim == 2:
                # Convert grayscale images to 3-channel images.
                arr = np.repeat(arr[:, :, None], 3, axis=-1)

            if arr.shape[-1] == 4:
                # Convert RGBA images to RGB images.
                arr = arr[..., :3]

            arr = arr.astype(np.float32) / 255.0
            result_mat[img_idx, :, :, :, type_idx] = arr

    print(f"Image matrix loaded successfully, shape={result_mat.shape}")
    return result_mat, folder_to_index, subfolders


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_img_from_np(input_np: np.ndarray):
    """Extract polarization parameters and return RGB arrays in the range [0, 1]."""
    c = input_np.shape[2]

    I0 = np.clip(input_np[..., :c, 0], 0, 1)
    I45 = np.clip(input_np[..., :c, 1], 0, 1)
    I90 = np.clip(input_np[..., :c, 2], 0, 1)
    I135 = np.clip(input_np[..., :c, 3], 0, 1)

    S0_raw = (I0 + I45 + I90 + I135) / 2.0
    S1 = I0 - I90
    S2 = I45 - I135

    S0_norm = S0_raw / 2.0
    S1_norm = normalize(S1)
    S2_norm = normalize(S2)

    AoP_raw = np.arctan2(S2, S1) / 2.0       # [-pi/2, pi/2]
    AoP_norm = (AoP_raw + np.pi / 2) / np.pi # [0, 1]

    DoP = np.sqrt(S1 ** 2 + S2 ** 2) / (S0_raw + 1e-8)
    DoP = np.clip(DoP, 0, 1)

    return I0, I45, I90, I135, S0_norm, S1_norm, S2_norm, AoP_norm, DoP, AoP_raw


# ---------- 3. Metric computation using pyiqa ---------- #
def _tensorify(img: np.ndarray) -> torch.Tensor:
    """Convert numpy [H, W, 3] float32 image in [0, 1] to torch [1, 3, H, W]."""
    return torch.from_numpy(img.transpose(2, 0, 1)[None, ...])


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return psnr_metric(_tensorify(img1), _tensorify(img2)).item()


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim_metric(_tensorify(img1), _tensorify(img2)).item()


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute AoP angular error in degrees."""
    err = np.minimum(
        np.abs(img1 - img2),
        np.abs(np.abs(img1 - img2) - np.pi)
    )
    return float(np.mean(err) / np.pi * 180.0)


# ---------- 4. Main metric calculation pipeline ---------- #
def cal_metrics(resultmat: np.ndarray, gtmat: np.ndarray, subfolders: list):
    all_metrics_data = []
    n = min(resultmat.shape[0], gtmat.shape[0])

    for idx in range(n):
        proc = resultmat[idx]
        gt = gtmat[idx]
        fname = subfolders[idx]

        gt_params = get_img_from_np(gt)
        out_params = get_img_from_np(proc)

        (
            gI0, gI45, gI90, gI135,
            gS0, gS1, gS2,
            gAoP_norm, gDoP, gAoP_raw
        ) = gt_params

        (
            oI0, oI45, oI90, oI135,
            oS0, oS1, oS2,
            oAoP_norm, oDoP, oAoP_raw
        ) = out_params

        metrics = {
            "image_filename": fname,

            "I0_psnr": calculate_psnr(oI0, gI0),
            "I45_psnr": calculate_psnr(oI45, gI45),
            "I90_psnr": calculate_psnr(oI90, gI90),
            "I135_psnr": calculate_psnr(oI135, gI135),
            "S0_psnr": calculate_psnr(oS0, gS0),
            "S1_psnr": calculate_psnr(oS1, gS1),
            "S2_psnr": calculate_psnr(oS2, gS2),
            "DoP_psnr": calculate_psnr(oDoP, gDoP),
            "AoP_psnr": calculate_psnr(oAoP_norm, gAoP_norm),

            "I0_ssim": calculate_ssim(oI0, gI0),
            "I45_ssim": calculate_ssim(oI45, gI45),
            "I90_ssim": calculate_ssim(oI90, gI90),
            "I135_ssim": calculate_ssim(oI135, gI135),
            "S0_ssim": calculate_ssim(oS0, gS0),
            "S1_ssim": calculate_ssim(oS1, gS1),
            "S2_ssim": calculate_ssim(oS2, gS2),
            "DoP_ssim": calculate_ssim(oDoP, gDoP),
            "AoP_ssim": calculate_ssim(oAoP_norm, gAoP_norm),

            "AoP_mae": calculate_mae(oAoP_raw, gAoP_raw)
        }

        all_metrics_data.append(metrics)
        print(f"Processed {idx + 1}/{n}: {fname}")

    return all_metrics_data


if __name__ == "__main__":
    gt_folder = r"dataset\\test\\PIDSR_aug"
    result_folder = r"debug\denoising\test\images\0"

    save_path = r"metrics"
    savename = "pidsr.xlsx"

    resultmat, _, subfolders = load_images(result_folder)
    gtmat, _, _ = load_images(gt_folder)

    all_metrics = cal_metrics(resultmat, gtmat, subfolders)

    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    excel_path = os.path.join(save_path, savename)
    df.to_excel(excel_path, index=False)

    print(f"\nMetrics saved to: {excel_path}")

    # Print average metrics.
    avg = {
        f"avg_{k}": np.mean([d[k] for d in all_metrics])
        for k in all_metrics[0]
        if k != "image_filename"
    }

    print("\n=== Average Metrics ===")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")
