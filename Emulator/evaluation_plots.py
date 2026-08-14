import os
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List

import torch
import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DConditionModel


class GalaxyDiffusionInference:
    """
    Handles inference for the conditional galaxy diffusion model.

    Parameters
    ----------
    model_dir : Union[str, Path]
        Directory containing the 'best_model' folder and 'scaling_metadata.joblib'.
    device : str, optional
        Compute device ('cuda', 'mps', or 'cpu'). Defaults to 'cuda' if available.
    seed : int, optional
        Random seed for reproducible noise generation. Defaults to 42.
    """

    def __init__(
        self, model_dir: Union[str, Path], device: Optional[str] = None, seed: int = 42
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(self.seed)
        if "cuda" in self.device:
            torch.cuda.manual_seed_all(self.seed)

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads the UNet, diffusion scheduler, and data transformation metadata."""
        model_path = self.model_dir / "best_model"
        metadata_path = self.model_dir / "scaling_metadata.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        print(f"Loading UNet from {model_path} onto {self.device}...")
        self.model = UNet2DConditionModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
        )

        print(f"Loading scaling metadata from {metadata_path}...")
        self.metadata: Dict[str, Any] = joblib.load(metadata_path)

        self.param_scaler = self.metadata["param_scaler"]
        self.p_min = self.metadata["p_min"]
        self.p_max = self.metadata["p_max"]

    def preprocess_parameters(self, df_params: pl.DataFrame) -> torch.Tensor:
        required_cols = ["stellar_mass", "redshift", "agn_fraction", "sfr"]
        for col in required_cols:
            if col not in df_params.columns:
                raise ValueError(f"Missing required parameter column: {col}")

        processed_df = df_params.with_columns(
            pl.when(pl.col("stellar_mass") > 0)
            .then(pl.col("stellar_mass").log10())
            .otherwise(0.0)
            .alias("log_mass")
        )

        ordered_cols = ["log_mass", "redshift", "agn_fraction", "sfr"]
        raw_params_array = processed_df.select(ordered_cols).to_numpy()

        scaled_params = self.param_scaler.transform(raw_params_array)

        return torch.tensor(scaled_params, dtype=torch.float32, device=self.device)

    def inverse_image_transform(self, images: torch.Tensor) -> torch.Tensor:
        images = (images + 1.0) / 2.0
        images = images * (self.p_max - self.p_min) + self.p_min
        images = torch.sinh(images)
        return images

    @torch.no_grad()
    def generate(
        self,
        unscaled_params: pl.DataFrame,
        image_shape: Tuple[int, int, int] = (1, 64, 64),
        num_inference_steps: int = 50,
        return_raw_flux: bool = True,
    ) -> torch.Tensor:
        batch_size = unscaled_params.height
        class_labels = self.preprocess_parameters(unscaled_params)

        shape = (batch_size, *image_shape)
        image = torch.randn(shape, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        print(f"Generating {batch_size} galaxies across {num_inference_steps} steps...")

        for t in self.scheduler.timesteps:
            model_output = self.model(
                sample=image,
                timestep=t,
                class_labels=class_labels,
                encoder_hidden_states=None,
            )
            noise_pred = model_output.sample
            image = self.scheduler.step(noise_pred, t, image).prev_sample

        if return_raw_flux:
            image = self.inverse_image_transform(image)
        else:
            image = (image + 1.0) / 2.0

        return image.cpu()


def plot_galaxies_grid(
    images_list: List[torch.Tensor],
    params_list: List[pl.DataFrame],
    row_labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
    is_flux: bool = True,
) -> None:
    """
    Plots a unified grid where each row is a different experiment (varying a specific parameter).
    """
    rows = len(images_list)
    cols = images_list[0].shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r in range(rows):
        images_np = images_list[r].numpy()
        params_dicts = params_list[r].to_dicts()
        label = row_labels[r]

        for c in range(cols):
            ax = axes[r, c]
            img = images_np[c]
            p_dict = params_dicts[c]

            if img.shape[0] == 1:
                img_to_plot = img[0]
            else:
                img_to_plot = img.transpose(1, 2, 0)

            ax.imshow(img_to_plot, cmap="magma", origin="lower")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"Varying:\n{label}",
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    clip_on=False,
                )

            mass_str = (
                f"Mass: {p_dict['stellar_mass']:.2e}"
                if p_dict["stellar_mass"] > 1e4
                else f"Mass: {p_dict['stellar_mass']:.2f}"
            )
            param_text = (
                f"{mass_str}\n"
                f"z: {p_dict['redshift']:.2f}\n"
                f"AGN: {p_dict['agn_fraction']:.2f}\n"
                f"SFR: {p_dict['sfr']:.2f}"
            )

            ax.text(
                0.05,
                0.95,
                param_text,
                transform=ax.transAxes,
                color="white",
                fontsize=10,
                va="top",
                ha="left",
                bbox=dict(
                    facecolor="black",
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grid plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Fixed stellar mass, agn fraction, and sfr; varying redshift
    mock_experiments_redshifts = pl.DataFrame(
        {
            "stellar_mass": [1e10, 1e10, 1e10, 1e10],
            "redshift": [0.1, 1.0, 2.0, 3.0],
            "agn_fraction": [0.01, 0.01, 0.01, 0.01],
            "sfr": [5.0, 5.0, 5.0, 5.0],
        }
    )

    # Fixed redshift, agn fraction, and sfr; varying stellar mass
    mock_experiments_stellar_mass = pl.DataFrame(
        {
            "stellar_mass": [5e9, 1e10, 1e11, 1e12],
            "redshift": [0.1, 0.1, 0.1, 0.1],
            "agn_fraction": [0.01, 0.01, 0.01, 0.01],
            "sfr": [5.0, 5.0, 5.0, 5.0],
        }
    )

    # Fixed stellar mass, redshift, and sfr; varying agn fraction
    mock_experiments_agn_fraction = pl.DataFrame(
        {
            "stellar_mass": [1e10, 1e10, 1e10, 1e10],
            "redshift": [1, 1, 1, 1],
            "agn_fraction": [0.01, 0.1, 0.5, 0.99],
            "sfr": [5.0, 5.0, 5.0, 5.0],
        }
    )

    # Fixed stellar mass, redshift, and agn fraction; varying sfr
    mock_experiments_sfr = pl.DataFrame(
        {
            "stellar_mass": [1e10, 1e10, 1e10, 1e10],
            "redshift": [0.1, 0.1, 0.1, 0.1],
            "agn_fraction": [0.01, 0.01, 0.01, 0.01],
            "sfr": [1.0, 10.0, 40.0, 100.0],
        }
    )

    SAVE_DIR = "outputs_diffusers"

    mock_experiments_list = ["Redshift", "Stellar Mass", "AGN Fraction", "SFR"]
    mock_experiments_data = [
        mock_experiments_redshifts,
        mock_experiments_stellar_mass,
        mock_experiments_agn_fraction,
        mock_experiments_sfr,
    ]

    try:
        pipeline = GalaxyDiffusionInference(model_dir=SAVE_DIR, device="cpu")

        all_images = []

        for name, mock_experiments in zip(mock_experiments_list, mock_experiments_data):
            raw_flux_images = pipeline.generate(
                unscaled_params=mock_experiments,
                image_shape=(1, 64, 64),
                num_inference_steps=100,
                return_raw_flux=True,
            )
            all_images.append(raw_flux_images)

        plot_galaxies_grid(
            images_list=all_images,
            params_list=mock_experiments_data,
            row_labels=mock_experiments_list,
            save_path="inference_raw_flux_grid.png",
            is_flux=True,
        )

    except FileNotFoundError as e:
        print(f"Initialization Error: {e}")
