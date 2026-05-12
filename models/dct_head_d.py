import numpy as np
import torch


def dct_dict_initialization(
        y: torch.Tensor,
        C_in: int = 64,
        d_size: int = 3,
        sample_independent: bool = True,
        required_C_out: int = 12
) -> torch.Tensor:
    if d_size < 3:
        raise ValueError(f"Atom size d_size must be >= 3, but got {d_size}")

    N, C_out, H, W = y.shape
    if C_out != required_C_out:
        raise ValueError(
            f"The input y has C_out={C_out}, but required_C_out={required_C_out}"
        )

      
    if isinstance(sample_independent, str):
        sample_independent = sample_independent.lower() == 'true'

      
    dct_dict = np.zeros(
        (N, required_C_out, C_in, d_size, d_size),
        dtype=np.float32
    )

    def generate_standard_dct_1d(length: int) -> np.ndarray:
        dct_1d = np.zeros((length, length), dtype=np.float32)
        for k in range(length):    
            for i in range(length):    
                angle = (2 * i + 1) * k * np.pi / (2 * length)
                dct_1d[i, k] = np.cos(angle)    

              
            norm_coeff = np.sqrt(1.0 / length) if k == 0 else np.sqrt(2.0 / length)
            dct_1d[:, k] *= norm_coeff

        return dct_1d
      
    dct_1d = generate_standard_dct_1d(d_size)
    dct_2d_atoms = []
    for k1 in range(d_size):    
        for k2 in range(d_size):    
            atom_2d = np.outer(dct_1d[:, k1], dct_1d[:, k2])    
            dct_2d_atoms.append(atom_2d)

    dct_2d_atoms = np.array(dct_2d_atoms, dtype=np.float32)  # [d_size², d_size, d_size]
    generated_atom_num = dct_2d_atoms.shape[0]
      
    if generated_atom_num < required_C_out:
        missing_num = required_C_out - generated_atom_num
        supplement_atoms = dct_2d_atoms[:missing_num].copy()    

        rng = np.random.default_rng(seed=42)
        perturb = 1e-4 * rng.normal(size=supplement_atoms.shape)    
        perturb = perturb.astype(np.float32)    

        supplement_atoms += perturb
        selected_atoms = np.concatenate([dct_2d_atoms, supplement_atoms], axis=0)
    else:
        selected_atoms = dct_2d_atoms[:required_C_out]

      
    for n in range(N):
        for c_in in range(C_in):
            if sample_independent:
                  
                rng_sample = np.random.default_rng(seed=42 + n)
                sample_perturb = 1e-4 * rng_sample.normal(size=selected_atoms.shape)
                sample_perturb = sample_perturb.astype(np.float32)
                current_atoms = selected_atoms + sample_perturb
            else:
                current_atoms = selected_atoms

            dct_dict[n, :, c_in, :, :] = current_atoms

    for n in range(N):
        for c_out in range(required_C_out):
            for c_in in range(C_in):
                atom = dct_dict[n, c_out, c_in]
                atom_norm = np.linalg.norm(atom)
                if atom_norm > 1e-8: 
                    dct_dict[n, c_out, c_in] = atom / atom_norm

    dct_dict_tensor = torch.from_numpy(dct_dict).to(
        device=y.device,
        dtype=y.dtype,
        non_blocking=True
    )

    return dct_dict_tensor