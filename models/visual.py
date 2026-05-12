import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
 
def save_single_channel_output(
        output_tensor: torch.Tensor,
        save_path: str,
        sample_idx: int = 0,  
        normalize: bool = True
):
   
    ensure_dir(os.path.dirname(save_path))
     
    output_np = tensor_to_numpy(output_tensor)
     
    img = output_np[sample_idx, 0]  # [H, W]

    if normalize:
         
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
     
    img = img.astype(np.uint8)
    cv2.imwrite(save_path, img)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)


def normalize_atom_signed(atom: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    atom = atom.astype(np.float32)
    atom = atom - atom.mean()
    max_abs = np.max(np.abs(atom))
    if max_abs < eps:
        return np.zeros_like(atom)
    return atom / (max_abs + eps)


def make_grid(images: np.ndarray, ncols: int = 8, pad: int = 2, pad_value: float = 0.0):
    """
    images: [N, H, W]
    """
    n, h, w = images.shape
    nrows = math.ceil(n / ncols)
    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad
    grid = np.ones((grid_h, grid_w), dtype=np.float32) * pad_value

    for idx in range(n):
        r = idx // ncols
        c = idx % ncols
        y = r * (h + pad)
        x = c * (w + pad)
        grid[y:y + h, x:x + w] = images[idx]
    return grid


def plot_grid_maps(
        maps: np.ndarray,
        save_path: str,
        title: str = "",
        cmap: str = "gray",
        ncols: int = 4,
        colorbar: bool = False
):
    n_maps = maps.shape[0]
    nrows = math.ceil(n_maps / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_maps:
            im = ax.imshow(maps[i], cmap=cmap)
            if colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
        values: np.ndarray,
        save_path: str,
        title: str,
        bins: int = 100,
        log_y: bool = True,
        xlabel: str = "Coefficient Magnitude",
        ylabel: str = "Frequency"
):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(values.flatten(), bins=bins, color="#6F9FE9")
    if log_y:
        plt.yscale("log")
        ylabel = ylabel + " (log scale)"
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    if title:
        plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binary_grid(
        maps: np.ndarray,
        threshold: float,
        save_path: str,
        title: str = "",
        ncols: int = 4
):

    binary_maps = (np.abs(maps) > threshold).astype(np.float32)
    plot_grid_maps(binary_maps, save_path, title=title, cmap="gray", ncols=ncols)


def visualize_sparse_coefficients(
        Z: torch.Tensor,
        save_dir: str,
        threshold: float = 1e-3,
):
    ensure_dir(save_dir)
    Z_np = tensor_to_numpy(Z)  # [B, C, H, W]
    sample = Z_np[0]  # [C, H, W]

     
    coeff_maps = np.stack([normalize_map(m) for m in sample], axis=0)
    plot_grid_maps(
        coeff_maps,
        os.path.join(save_dir, "Z_maps.png"),
        title="Coefficient maps",
        cmap="gray",
        ncols=8,
        colorbar=False
    )

     
    plot_binary_grid(
        sample,
        threshold,
        os.path.join(save_dir, "Z_binary.png"),
        title=f"Binary activation maps (threshold={threshold})",
        ncols=8
    )

     
    plot_histogram(
        np.abs(Z_np),
        os.path.join(save_dir, "Z_abs_hist.png"),
        title="Histogram of coefficient magnitudes",
        bins=100,
        log_y=True,
        xlabel="Coefficient Magnitude",
        ylabel="Frequency"
    )

     
    sparsity_per_sample = (np.abs(Z_np) < threshold).reshape(Z_np.shape[0], -1).mean(axis=1)
    plot_histogram(
        sparsity_per_sample,
        os.path.join(save_dir, "Z_sparsity_hist.png"),
        title=f"Sparsity distribution over samples (threshold={threshold})",
        bins=30,
        log_y=False,
        xlabel="Sparsity",
        ylabel="Number of Samples"
    )

    sample_sparsity = (np.abs(sample) < threshold).mean()
    mean_sparsity = sparsity_per_sample.mean()

    print(f"[Visualization] Sample-0 sparsity: {sample_sparsity * 100:.2f}%")
    print(f"[Visualization] Mean sparsity over batch: {mean_sparsity * 100:.2f}%")

    return {
        "sample_sparsity": sample_sparsity,
        "mean_sparsity": mean_sparsity
    }


 

def fft_spectrum_2d_padded(kernel_2d: np.ndarray, pad_size: int = 32) -> np.ndarray:

    kernel_2d = kernel_2d.astype(np.float32)
    h, w = kernel_2d.shape
    canvas = np.zeros((pad_size, pad_size), dtype=np.float32)

    y0 = (pad_size - h) // 2
    x0 = (pad_size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = kernel_2d

    fft_map = np.fft.fftshift(np.fft.fft2(canvas))
    mag = np.log(np.abs(fft_map) + 1e-8)
    return normalize_map(mag)


def compute_atom_orientation_hist(atom: np.ndarray, num_bins: int = 12, eps: float = 1e-8):

    atom = atom.astype(np.float32)
    gx = cv2.Sobel(atom, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(atom, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx)
    ori = np.mod(ori, np.pi)  # [0, pi)

    bins = np.linspace(0, np.pi, num_bins + 1)
    hist = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (ori >= bins[i]) & (ori < bins[i + 1])
        hist[i] = mag[mask].sum()

    hist = hist / (hist.sum() + eps)
    return hist


def compute_atom_main_orientation(atom: np.ndarray, num_bins: int = 12):
    hist = compute_atom_orientation_hist(atom, num_bins=num_bins)
    idx = int(np.argmax(hist))
    angle = idx * (180.0 / num_bins)
    strength = float(hist[idx])
    return angle, strength, hist


def plot_orientation_distribution(
        atoms: np.ndarray,
        save_path: str,
        num_bins: int = 20
):

    hists = np.stack([compute_atom_orientation_hist(a, num_bins=num_bins) for a in atoms], axis=0)
    mean_hist = hists.mean(axis=0)

    angles = np.linspace(0, 180, num_bins, endpoint=False)

    plt.figure(figsize=(5, 5))
    plt.bar(angles, mean_hist, width=(180 / num_bins) * 0.85)

     
    ax = plt.gca()
    ax.spines['top'].set_visible(False)   
    ax.spines['right'].set_visible(False)   

    plt.xlabel("Orientation (degrees)", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized energy", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return mean_hist


def plot_dictionary_atoms_spatial_overview(
        atoms: np.ndarray,
        save_path: str,
        n_show: int = 64,
        ncols: int = 8,
        upscale: int = 28
):

    n_show = min(n_show, atoms.shape[0])
    atoms = atoms[:n_show]

    atoms_vis = np.stack([normalize_atom_signed(a) for a in atoms], axis=0)
    grid = make_grid(atoms_vis, ncols=ncols, pad=1, pad_value=0.0)
    grid = cv2.resize(
        grid,
        (grid.shape[1] * upscale, grid.shape[0] * upscale),
        interpolation=cv2.INTER_NEAREST
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='bwr', vmin=-1, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dictionary_atoms_spatial_representative(
        atoms: np.ndarray,
        save_path: str,
        n_show: int = 16,
        ncols: int = 4,
        num_bins: int = 12,
        select_mode: str = "directional"
):

    atoms = np.asarray(atoms)
    n_total = atoms.shape[0]

    infos = []
    for idx in range(n_total):
        atom = normalize_atom_signed(atoms[idx])
        angle, strength, hist = compute_atom_main_orientation(atom, num_bins=num_bins)
        infos.append({
            "idx": idx,
            "atom": atom,
            "angle": angle,
            "strength": strength,
            "hist": hist
        })

    if select_mode == "directional":
         
        infos = sorted(infos, key=lambda x: x["strength"], reverse=True)[:n_show]
        infos = sorted(infos, key=lambda x: (x["angle"], -x["strength"]))

    elif select_mode == "mixed":
         
        bins_deg = np.linspace(0, 180, num_bins + 1)
        selected = []
        used = set()

        per_bin = max(1, math.ceil(n_show / num_bins))
        for b in range(num_bins):
            bin_infos = [x for x in infos if bins_deg[b] <= x["angle"] < bins_deg[b + 1]]
            bin_infos = sorted(bin_infos, key=lambda x: x["strength"], reverse=True)
            for item in bin_infos[:per_bin]:
                if item["idx"] not in used:
                    selected.append(item)
                    used.add(item["idx"])

        selected = sorted(selected, key=lambda x: (-x["strength"], x["angle"]))[:n_show]
        infos = sorted(selected, key=lambda x: (x["angle"], -x["strength"]))

    else:
        raise ValueError("select_mode must be 'directional' or 'mixed'")

    n_show = min(n_show, len(infos))
    nrows = math.ceil(n_show / ncols)

    fig, axes = plt.subsets(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_show:
            item = infos[i]
            atom = item["atom"]
            angle = item["angle"]
            strength = item["strength"]
            idx = item["idx"]

            ax.imshow(atom, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"#{idx}\nDir={angle:.0f}°, P={strength:.2f}", fontsize=9)

             
            for r in range(atom.shape[0]):
                for c in range(atom.shape[1]):
                    ax.text(
                        c, r, f"{atom[r, c]:.1f}",
                        ha="center", va="center",
                        fontsize=8, color="black"
                    )

            ax.set_xticks(np.arange(-0.5, atom.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, atom.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dictionary_atoms_spectra(
        atoms: np.ndarray,
        save_path: str,
        n_show: int = 64,
        ncols: int = 8,
        pad_size: int = 32
):

    n_show = min(n_show, atoms.shape[0])
    atoms = atoms[:n_show]

    spectra = np.stack([fft_spectrum_2d_padded(a, pad_size=pad_size) for a in atoms], axis=0)

    nrows = math.ceil(n_show / ncols)
    fig, axes = plt.subsets(nrows, ncols, figsize=(8, 8))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_show:
            ax.imshow(spectra[i], cmap="gray", interpolation="nearest")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_dictionary_atoms(
        D: torch.Tensor,
        save_dir: str,
        image_channel: int = 0,
        n_show: int = 64,
        n_rep: int = 16,
        rep_select_mode: str = "directional"
):

    ensure_dir(save_dir)

    D_np = tensor_to_numpy(D)[0]   
    atoms = D_np[image_channel]  # [N_atoms, k, k]

     
    path_space_rep = os.path.join(save_dir, f"dict_atoms_spatial_representative_ch{image_channel}.png")
    plot_dictionary_atoms_spatial_representative(
        atoms,
        path_space_rep,
        n_show=n_rep,
        ncols=4,
        num_bins=12,
        select_mode=rep_select_mode
    )

     
    path_space_overview = os.path.join(save_dir, f"dict_atoms_spatial_overview_ch{image_channel}.png")
    plot_dictionary_atoms_spatial_overview(
        atoms,
        path_space_overview,
        n_show=n_show,
        ncols=8,
        upscale=28
    )

     
    path_spectrum = os.path.join(save_dir, f"dict_atoms_spectrum_ch{image_channel}.png")
    plot_dictionary_atoms_spectra(
        atoms,
        path_spectrum,
        n_show=n_show,
        ncols=8,
        pad_size=32
    )

     
    path_orientation = os.path.join(save_dir, f"dict_atoms_orientation_ch{image_channel}.png")
    mean_ori_hist = plot_orientation_distribution(
        atoms[:n_show],
        path_orientation,
        num_bins=10
    )

    print("Rebuttal dictionary visualization saved:")
    print(f"  Spatial representative : {path_space_rep}")
    print(f"  Spatial overview       : {path_space_overview}")
    print(f"  Spectrum atoms         : {path_spectrum}")
    print(f"  Orientation stats      : {path_orientation}")

    return {
        "orientation_hist_mean": mean_ori_hist.tolist()
    }