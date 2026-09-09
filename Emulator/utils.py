import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

import numpy as np


def generate_samples(
    model,
    noise_scheduler,
    cond_params,
    device,
    image_shape=(1, 64, 64),
    num_steps=1000,
):
    """
    Runs the reverse diffusion process to generate images.
    """
    model.eval()
    batch_size = cond_params.shape[0]

    # populates noise_scheduler.timesteps
    noise_scheduler.set_timesteps(num_inference_steps=num_steps)

    # pure Gaussian noise
    x = torch.randn((batch_size, *image_shape), device=device)

    # DDPM reverse process loop
    with torch.no_grad():
        # iterate over the scheduler's timesteps array
        for t in noise_scheduler.timesteps:

            # Predict noise residual
            predicted_noise = model(
                sample=x,
                timestep=t,
                encoder_hidden_states=None,
                class_labels=cond_params,
            ).sample

            x = noise_scheduler.step(predicted_noise, t, x).prev_sample

    return x


def plot_loss(train_losses, val_losses, save_dir="outputs"):
    """
    Saves a plot of the training and validation loss curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()


def plot_distributions(real_images, fake_images, save_dir="outputs"):
    """
    Compares the pixel intensity distributions of real vs. generated galaxies.
    """
    os.makedirs(save_dir, exist_ok=True)

    real_pixels = real_images.cpu().numpy().flatten()
    fake_pixels = fake_images.cpu().numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(real_pixels, bins=100, alpha=0.5, density=True, label="Real Data")
    plt.hist(fake_pixels, bins=100, alpha=0.5, density=True, label="Generated Data")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title("Learned Pixel Distribution vs Real")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "distribution_plot.png"))
    plt.close()


def plot_pretraining_distributions(dataset, save_dir="outputs"):
    """
    Plots the physical parameter distributions and pixel intensity
    distribution to verify data integrity before training.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("Generating pre-training distribution plots...")

    scaled_params = dataset.params

    if hasattr(dataset.scaler_stats, "inverse_transform"):
        unscaled_params = dataset.scaler_stats.inverse_transform(scaled_params)
    else:
        mean = dataset.scaler_stats["mean"]
        std = dataset.scaler_stats["std"]
        unscaled_params = (scaled_params * std) + mean

    param_names = ["Log10(Stellar Mass)", "Redshift", "AGN Fraction", "sfr"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(4):
        axes[i].hist(unscaled_params[:, i], bins=50, color="skyblue", edgecolor="black")
        axes[i].set_title(param_names[i])
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_distributions.png"))
    plt.close()

    # Randomly sample up to 1000 images, that should be enough to get a good idea of the pixel distribution
    sample_size = min(len(dataset), 1000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    pixel_values = []
    for idx in indices:
        # dataset[idx] triggers the HDF5 read and normalization
        image, _ = dataset[idx]
        pixel_values.append(image.numpy().flatten())

    pixel_values = np.concatenate(pixel_values)

    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=100, color="coral", alpha=0.7, log=True)
    plt.title(f"Normalized Pixel Distribution (N={sample_size} cutouts)")
    plt.xlabel("Pixel Intensity [-1.0 to 1.0]")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "pixel_distribution.png"))
    plt.close()

    print(f"Diagnostic plots saved to '{save_dir}'.")
