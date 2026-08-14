import os

import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer

"""
This is a specific loader for Galaxy and AGN cutout dataset that are stored in seperate hdf5 files for each synthetic image. 
this code loads them and applies required transformations to the images and parameters. It returns the training and testing dataloaders for the diffusion model.
"""


def parse_hdf5_directory(data_dir):
    """
    Scans a directory for HDF5 files and aggregates all galaxy keys and physical parameters.
    Returns a list of (file_path, key) tuples and a consolidated NumPy array of parameters.
    """
    all_items = []
    all_params = []

    # locate all HDF5 files in the directory
    file_paths = glob.glob(os.path.join(data_dir, "*.h5")) + glob.glob(
        os.path.join(data_dir, "*.hdf5")
    )

    if not file_paths:
        raise FileNotFoundError(f"No .h5 or .hdf5 files found in directory: {data_dir}")

    print(f"Found {len(file_paths)} HDF5 files. Extracting metadata...")

    for f_path in file_paths:
        with h5py.File(f_path, "r") as f:
            for k in f.keys():
                meta = dict(f[k].attrs)

                # Filter out invalid or NaN data, shouldnt be any but just in case.
                if (
                    meta.get("stellar_mass", "NaN") == "NaN"
                    or meta.get("redshift", "NaN") == "NaN"
                    or meta.get("agn_fraction", "NaN") == "NaN"
                    or meta.get("sfr", "NaN") == "NaN"
                ):
                    continue

                # Parse strings to floats safely
                mass_str = str(meta["stellar_mass"]).replace(" Msun", "").strip()
                mass = float(mass_str)

                # Log10 scaling to manage large magnitude differences
                log_mass = np.log10(mass) if mass > 0 else 0.0
                redshift = float(meta["redshift"])
                agn_frac = float(meta["agn_fraction"])
                sfr = float(meta["sfr"])

                param_vector = [log_mass, redshift, agn_frac, sfr]

                # Store the exact file path alongside the key
                all_items.append((f_path, k))
                all_params.append(param_vector)

    print(f"Successfully indexed {len(all_items)} total galaxies across all files.")
    return all_items, np.array(all_params, dtype=np.float32)


class MultiFileGalaxyDataset(Dataset):
    def __init__(self, items, params, scaler_stats=None, p_min=-0.5, p_max=4.5):
        """
        Args:
            items: List of tuples (file_path, galaxy_key)
            params: NumPy array of shape (N, param_dim)
            scaler_stats: Fitted QuantileTransformer object for parameters.
            p_min: Pre-calculated 0.1th percentile of asinh-stretched pixels across the dataset.
            p_max: Pre-calculated 99.9th percentile of asinh-stretched pixels across the dataset.
        """
        self.items = items
        self.params = params
        self.file_handles = {}

        # percentile bounds
        self.p_min = p_min
        self.p_max = p_max

        if scaler_stats is None:
            self.scaler = QuantileTransformer(
                output_distribution="normal", random_state=42
            )
            self.params = self.scaler.fit_transform(self.params)
            self.scaler_stats = self.scaler
        else:
            self.scaler = scaler_stats
            self.params = self.scaler.transform(self.params)
            self.scaler_stats = self.scaler

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, key = self.items[idx]

        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, "r")

        raw_image = self.file_handles[file_path][key][:]
        image_np = raw_image.astype(np.float32)

        # apply log stretch, 1p to avoid log(0)
        image_np = np.log1p(image_np)

        # clip outliers to the robust percentiles FIRST
        image_np = np.clip(image_np, self.p_min, self.p_max)

        # scale to [-1.0, 1.0] using the clipped range
        image_np = 2.0 * (image_np - self.p_min) / (self.p_max - self.p_min) - 1.0

        image = torch.from_numpy(image_np).float()

        if image.ndim == 3 and image.shape[-1] in [1, 3]:
            image = image.permute(2, 0, 1)
        elif image.ndim == 2:
            image = image.unsqueeze(0)

        params = torch.tensor(self.params[idx], dtype=torch.float32)

        return image, params

    def close_handles(self):
        for handle in self.file_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.file_handles = {}

    def __del__(self):
        self.close_handles()


def estimate_robust_percentiles(items, num_samples=2500, lower_p=0.5, upper_p=99.5):
    """
    Dynamically estimates robust percentiles from a random sample of the dataset
    to ensure optimal dynamic range scaling without overloading memory.
    """
    print(f"Dynamically calculating robust percentiles over {num_samples} samples...")
    sampled_pixels = []

    sample_size = min(num_samples, len(items))
    sample_indices = np.random.choice(len(items), size=sample_size, replace=False)

    temp_handles = {}

    for idx in sample_indices:
        file_path, key = items[idx]
        if file_path not in temp_handles:
            temp_handles[file_path] = h5py.File(file_path, "r")

        raw_image = temp_handles[file_path][key][:].astype(np.float32)

        stretched_image = np.log1p(raw_image)

        sampled_pixels.append(stretched_image.flatten())

    for handle in temp_handles.values():
        handle.close()

    all_pixels = np.concatenate(sampled_pixels)
    p_min = np.percentile(all_pixels, lower_p)
    p_max = np.percentile(all_pixels, upper_p)

    print(
        f"Calculated dynamic scaling bounds -> p_min: {p_min:.5f}, p_max: {p_max:.5f}"
    )
    return p_min, p_max


def create_dataloaders(data_dir, batch_size=32, train_split=0.8, debug_limit=None):
    """
    Creates isolated training and testing DataLoaders.
    Includes a debug_limit for fast pipeline testing.
    Safely handles cases where train_split == 1.0 (no test set).
    """
    items, params = parse_hdf5_directory(data_dir)

    if debug_limit is not None and debug_limit < len(items):
        print(
            f"DEBUG MODE ACTIVE: Subsampling dataset to {debug_limit} total galaxies."
        )
        indices = torch.randperm(len(items))[:debug_limit].tolist()
        items = [items[i] for i in indices]
        params = params[indices]

    total_size = len(items)
    train_size = int(train_split * total_size)

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_items = [items[i] for i in train_indices]
    train_params = params[train_indices]

    test_items = [items[i] for i in test_indices]
    test_params = params[test_indices]

    p_min, p_max = estimate_robust_percentiles(
        train_items, num_samples=10000, lower_p=0.01, upper_p=99.999
    )

    train_dataset = MultiFileGalaxyDataset(
        train_items, train_params, p_min=p_min, p_max=p_max
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = None
    if len(test_items) > 0:
        test_dataset = MultiFileGalaxyDataset(
            test_items,
            test_params,
            scaler_stats=train_dataset.scaler_stats,
            p_min=p_min,
            p_max=p_max,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        print("Notice: train_split set to 1.0. No validation loader will be returned.")

    return train_loader, test_loader, train_dataset.scaler
