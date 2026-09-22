import os
import joblib
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, UNet2DConditionModel

from config import *

from dataset import create_dataloaders
from utils import (
    plot_loss,
    plot_distributions,
    generate_samples,
    plot_pretraining_distributions,
    EMAModel,
)
import time


def train_conditional_diffusion(
    model,
    train_loader,
    val_loader,
    noise_scheduler,
    optimizer,
    scheduler,
    device,
    num_epochs=1000,
    save_dir="outputs",
    start_epoch=0,
    best_val_loss=float("inf"),
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    scaler = torch.amp.GradScaler("cuda" if "cuda" in str(device) else "cpu")

    train_loss_history = []
    val_loss_history = []

    print(
        f"Starting/Resuming training for {num_epochs} epochs. Press Ctrl+C to interrupt."
    )

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0

            for images, params in train_loader:
                images = images.to(device, dtype=torch.float32)
                params = params.to(device, dtype=torch.float32)
                #  incase of NaN/Inf in the batch. can probaly remove...
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print("WARNING: NaN/Inf found in images batch! Skipping...")
                    continue
                if torch.isnan(params).any() or torch.isinf(params).any():
                    print("WARNING: NaN/Inf found in params batch! Skipping...")
                    continue

                optimizer.zero_grad(set_to_none=True)

                noise = torch.randn_like(images)
                bsz = images.shape[0]

                # sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                ).long()

                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                amp_dtype = (
                    torch.bfloat16
                    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                    else torch.float16
                )

                with torch.amp.autocast(
                    "cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype
                ):
                    model_output = model(
                        sample=noisy_images,
                        timestep=timesteps,
                        encoder_hidden_states=None,
                        class_labels=params,
                    )
                    noise_pred = model_output.sample
                    # try to prevent dtype mismatch issues by casting both to float
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            "WARNING: NaN/Inf detected in loss! Skipping backprop for this batch..."
                        )
                        optimizer.zero_grad()  # clear any bad gradients just in case
                        continue

                # scale loss and backprop
                scaler.scale(loss).backward()

                # unscale the gradients of the optimizer's assigned parameters in-place
                scaler.unscale_(optimizer)

                # clip the gradients to prevent explosion (1.0 is a standard safe value), probably not needed anymore....
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # step and update
                scaler.step(optimizer)
                scaler.update()

                # step scheduler per batch, not per epoch
                scheduler.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_images, val_params in val_loader:
                    val_images = val_images.to(device, dtype=torch.float32)
                    val_params = val_params.to(device, dtype=torch.float32)

                    if (
                        torch.isnan(val_images).any()
                        or torch.isinf(val_images).any()
                        or torch.isnan(val_params).any()
                        or torch.isinf(val_params).any()
                    ):
                        continue

                    noise = torch.randn_like(val_images)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (val_images.shape[0],),
                        device=device,
                    ).long()

                    noisy_images = noise_scheduler.add_noise(
                        val_images, noise, timesteps
                    )

                    with torch.amp.autocast(
                        "cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype
                    ):
                        val_output = model(
                            sample=noisy_images,
                            timestep=timesteps,
                            encoder_hidden_states=None,
                            class_labels=val_params,
                        )
                        noise_pred = val_output.sample

                    loss_val_fp32 = F.mse_loss(noise_pred.float(), noise.float())
                    val_loss += loss_val_fp32.item()

            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)

            print(
                f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(os.path.join(save_dir, "best_model"))
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(save_dir, "best_model", "optimizer_scheduler.pt"),
                )
                print(f" -> Saved new best model with Val Loss: {best_val_loss:.5f}")

            if (epoch + 1) % 10 == 0:
                plot_loss(train_loss_history, val_loss_history, save_dir)

                viz_images, viz_params = next(iter(val_loader))
                viz_images = viz_images.to(device, dtype=torch.float32)
                viz_params = viz_params.to(device, dtype=torch.float32)
                # asme samples each time for vis so we can see the progression of the model's generative capabilities
                num_samples = 16
                sample_params = viz_params[:num_samples]
                real_images_grid = viz_images[:num_samples]

                fake_images = generate_samples(
                    model,
                    noise_scheduler,
                    sample_params,
                    device,
                    image_shape=real_images_grid.shape[1:],
                )

                # denormalize from [-1, 1] to [0, 1] for easier visualization.
                fake_images = (fake_images + 1.0) / 2.0
                real_images_grid = (real_images_grid + 1.0) / 2.0

                # plot histograms
                plt.figure(figsize=(10, 5))
                plt.hist(
                    real_images_grid.cpu().numpy().flatten(),
                    bins=50,
                    alpha=0.7,
                    label="Real",
                    color="blue",
                )
                plt.title("Real vs Fake Images Pixel Distribution")
                plt.hist(
                    fake_images.cpu().numpy().flatten(),
                    bins=50,
                    alpha=0.7,
                    label="Fake",
                    color="orange",
                )
                plt.legend()
                plt.yscale("log")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"epoch_{epoch+1}_pixel_distribution.png")
                )
                plt.close()

                from matplotlib.colors import LogNorm

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        axes[i, j].axis("off")
                        if idx < fake_images.shape[0]:
                            axes[i, j].imshow(
                                fake_images[idx, 0].cpu().numpy(),
                                cmap="viridis",
                                norm=LogNorm(),
                            )
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_fake_images.png"))
                plt.close()

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        axes[i, j].axis("off")
                        if idx < real_images_grid.shape[0]:
                            axes[i, j].imshow(
                                real_images_grid[idx, 0].cpu().numpy(),
                                cmap="viridis",
                                norm=LogNorm(),
                            )
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_real_images.png"))
                plt.close()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Initiating graceful shutdown...")

    finally:
        print("Saving final visualizations and plots...")
        plot_loss(train_loss_history, val_loss_history, save_dir)

        # load best weights for evaluation
        best_model_path = os.path.join(save_dir, "best_model")
        if os.path.exists(best_model_path):
            model = UNet2DConditionModel.from_pretrained(best_model_path).to(device)
            print("Loaded best model for final evaluation.")

        model.eval()
        with torch.no_grad():
            real_images, eval_params = next(iter(val_loader))
            real_images = real_images.to(device, dtype=torch.float32)
            eval_params = eval_params.to(device, dtype=torch.float32)

            num_samples = min(100, real_images.shape[0])

            eval_params = eval_params[:num_samples]
            real_images = real_images[:num_samples]

            fake_images = generate_samples(
                model,
                noise_scheduler,
                eval_params,
                device,
                image_shape=real_images.shape[1:],
            )

            plot_distributions(real_images, fake_images, save_dir)

    print(f"Shutdown complete. Artifacts saved to '{save_dir}'.")


if __name__ == "__main__":
    t0 = time.time()

    train_loader, val_loader, _ = create_dataloaders(
        DATA_PATH, BATCH_SIZE, TRAIN_SPLIT, debug_limit=SAMPLE
    )

    scaling_metadata = {
        "param_scaler": train_loader.dataset.scaler_stats,
        "p_min": train_loader.dataset.p_min,
        "p_max": train_loader.dataset.p_max,
    }
    metadata_path = os.path.join(SAVE_DIR, "scaling_metadata.joblib")
    joblib.dump(scaling_metadata, metadata_path)
    print(f"Scaling metadata saved to: {metadata_path}")

    plot_pretraining_distributions(train_loader.dataset, SAVE_DIR)
    train_loader.dataset.close_handles()
    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    param_dim = 4

    # model can take a while to train, so this allows for exceding the maximum runtime of a single job on HPC clusters by allowing for checkpointing and resuming training.
    checkpoint_dir = os.path.join(SAVE_DIR, "best_model")
    opt_sch_path = os.path.join(checkpoint_dir, "optimizer_scheduler.pt")

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_dir):
        print(f"Found existing checkpoint at '{checkpoint_dir}'. Loading model...")
        model = UNet2DConditionModel.from_pretrained(checkpoint_dir).to(DEVICE)
    else:
        print("No checkpoint found. Initializing new model...")
        model = UNet2DConditionModel(
            sample_size=64,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
            class_embed_type="projection",
            projection_class_embeddings_input_dim=param_dim,
        ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    total_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_steps
    )

    if os.path.exists(opt_sch_path):
        print(f"Loading optimizer and scheduler states...")
        checkpoint = torch.load(opt_sch_path, map_location=DEVICE, weights_only=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(
            f"Resuming from epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.5f})"
        )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    train_conditional_diffusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_dir=SAVE_DIR,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )
    t1 = time.time()
    train_time = (t1 - t0) / 60 / 60
    print(f"Total training time: {t1 - t0:.2f} seconds")
    print(f"Total training time: {train_time:.2f} hours")
