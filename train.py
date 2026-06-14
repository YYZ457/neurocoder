"""
NeuroCoder Training Script
===========================
Optimized for RTX 4060 (8 GB VRAM).

Key optimizations for consumer GPU:
- BF16 mixed precision (native on 4060)
- Gradient checkpointing (trades compute for memory)
- Gradient accumulation (simulates larger batch)
- Small batch size with careful memory management

Usage:
    python train.py                          # Train with default config
    python train.py --config max             # Push 4060 to its limit
    python train.py --config small           # Quick experiments
    python train.py --data ./my_code         # Train on your own code
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import NeuroCoderConfig, TrainingConfig, CONFIG_4060, CONFIG_SMALL, CONFIG_MAX, CONFIG_CLOUD
from model import NeuroCoder
from data import CodeTokenizer, create_dataloaders, download_sample_data


# ===========================================================================
# Training utilities
# ===========================================================================

class TrainingStats:
    """Track and display training metrics."""

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.total_steps = 0
        self.total_tokens = 0
        self.best_eval_loss = float("inf")
        self.start_time = time.time()

        self.loss_history = []
        self.eval_history = []
        self.lr_history = []

    def update(self, loss, ce_loss, aux_loss, lr, tokens_per_step, ce_raw=None):
        self.step += 1
        self.total_steps += 1
        self.total_tokens += tokens_per_step
        self.loss_history.append({
            "step": self.total_steps,
            "loss": loss,
            "ce_loss": ce_loss,
            "ce_raw": ce_raw if ce_raw is not None else ce_loss,
            "aux_loss": aux_loss,
            "lr": lr,
        })
        self.lr_history.append(lr)

    def format_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log_step(self, loss, ce_loss, aux_loss, lr, tokens_per_step):
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.total_tokens / max(elapsed, 1)

        print(
            f"  Step {self.total_steps:>6d} | "
            f"Loss: {loss:.4f} (CE: {ce_loss:.4f}, Aux: {aux_loss:.4f}) | "
            f"LR: {lr:.2e} | "
            f"Tok/s: {tokens_per_sec:,.0f} | "
            f"Time: {self.format_time(elapsed)}"
        )

    def log_eval(self, eval_loss, eval_perplexity):
        self.eval_history.append({
            "step": self.total_steps,
            "loss": eval_loss,
            "perplexity": eval_perplexity,
        })
        is_best = eval_loss < self.best_eval_loss
        if is_best:
            self.best_eval_loss = eval_loss
        print(f"  [Eval] Eval | Loss: {eval_loss:.4f} | PPL: {eval_perplexity:.1f} {'*BEST* BEST' if is_best else ''}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "loss_history": self.loss_history,
                "eval_history": self.eval_history,
                "lr_history": self.lr_history,
                "total_steps": self.total_steps,
                "total_tokens": self.total_tokens,
                "best_eval_loss": self.best_eval_loss,
            }, f, indent=2)


def save_checkpoint(
    model, optimizer, scheduler, scaler, stats, config, path: str
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "stats": {
            "step": stats.total_steps,
            "total_tokens": stats.total_tokens,
            "best_eval_loss": stats.best_eval_loss,
        },
        "config": config,
    }
    try:
        torch.save(checkpoint, path)
        print(f"  [Save] Saved checkpoint: {path}")
    except Exception as e:
        print(f"  [Warn] Save failed (training continues): {e}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"  [Load] Loaded checkpoint from step {checkpoint['stats']['step']}")
    return checkpoint["stats"]


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    model_config: NeuroCoderConfig,
    train_config: TrainingConfig,
    train_loader,
    eval_loader=None,
):
    """
    Main training loop.

    Memory budget for RTX 4060:
        Model weights (FP16):      ~0.4 GB
        Gradients (FP16):          ~0.4 GB
        Optimizer (Adam FP32):     ~1.6 GB  (m + v)
        Activations:               ~1-2 GB  (with checkpointing)
        Batch data:                ~0.5 GB
        CUDA overhead:             ~0.5 GB
        ─────────────────────────────────────
        Total:                     ~5-6 GB  ✓ fits 8 GB
    """
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Compute: {torch.cuda.get_device_capability(0)}")

    # Create model
    print("\nCreating NeuroCoder model...")
    model = NeuroCoder(model_config)
    model.print_architecture()
    model = model.to(device)

    # Compile model (PyTorch 2.0+ for speedup)
    if train_config.compile_model and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

    # Enable gradient checkpointing
    if model_config.use_gradient_checkpointing:
        print("  Gradient checkpointing: enabled")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=model_config.weight_decay,
    )

    # Learning rate scheduler: warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=model_config.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=model_config.max_steps - model_config.warmup_steps,
        eta_min=model_config.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[model_config.warmup_steps],
    )

    # AMP scaler
    scaler = GradScaler(device.type, enabled=model_config.use_bf16)

    # Resume from checkpoint
    start_step = 0
    if train_config.resume_from:
        stats_dict = load_checkpoint(
            train_config.resume_from, model, optimizer, scheduler, scaler
        )
        start_step = stats_dict["step"]

    # Training stats
    stats = TrainingStats()
    stats.total_steps = start_step

    # Output directories
    os.makedirs(train_config.output_dir, exist_ok=True)
    os.makedirs(train_config.log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"  Effective batch size: {model_config.batch_size * model_config.gradient_accumulation}")
    print(f"  Max steps: {model_config.max_steps}")
    print(f"  Warmup: {model_config.warmup_steps}")
    print(f"  LR: {model_config.learning_rate}")
    print(f"  BF16: {model_config.use_bf16}")
    print(f"  Checkpoint every: {train_config.save_every}")
    print(f"{'='*60}\n")

    model.train()
    global_step = start_step
    train_iter = iter(train_loader)
    step_timer = time.time()
    crash_path = os.path.join(train_config.output_dir, "crash_recovery.pt")

    try:
        while global_step < model_config.max_steps:
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_ce = 0.0
            accum_aux = 0.0
            accum_ce_raw = 0.0

            # Gradient accumulation
            for micro_step in range(model_config.gradient_accumulation):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                    stats.epoch += 1

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=model_config.use_bf16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] / model_config.gradient_accumulation
                    ce_loss = outputs["ce_loss"]
                    aux_loss = outputs["aux_loss"]
                    ce_raw = outputs.get("ce_raw", ce_loss)

                scaler.scale(loss).backward()

                accum_loss += loss.item()
                accum_ce += ce_loss.item() / model_config.gradient_accumulation
                accum_aux += aux_loss.item() / model_config.gradient_accumulation
                accum_ce_raw += ce_raw.item() / model_config.gradient_accumulation

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            tokens_per_step = (
                model_config.batch_size
                * model_config.gradient_accumulation
                * model_config.max_seq_len
            )

            current_lr = scheduler.get_last_lr()[0]
            stats.update(accum_loss, accum_ce, accum_aux, current_lr, tokens_per_step, ce_raw=accum_ce_raw)

            step_time = time.time() - step_timer
            step_timer = time.time()
            vram = torch.cuda.memory_allocated() / 1024**3 if device.type == 'cuda' else 0
            eta_s = step_time * (model_config.max_steps - global_step)
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)
            print(f"  Step {global_step:>5d}/{model_config.max_steps} | "
                  f"Loss: {accum_loss:.4f} | CE: {accum_ce:.4f} | "
                  f"LR: {current_lr:.2e} | {step_time:.1f}s | VRAM: {vram:.1f}GB | "
                  f"ETA: {eta_h}h{eta_m}m"
                  + (f" | RawCE: {accum_ce_raw:.4f}" if accum_ce_raw != accum_ce else ""))

            if eval_loader is not None and global_step % train_config.eval_every == 0:
                eval_loss, eval_ppl = evaluate(model, eval_loader, device, model_config)
                stats.log_eval(eval_loss, eval_ppl)
                model.train()

            if global_step % train_config.save_every == 0:
                ckpt_path = os.path.join(train_config.output_dir, f"checkpoint-{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, scaler, stats, model_config, ckpt_path)
                save_checkpoint(model, optimizer, scheduler, scaler, stats, model_config,
                               os.path.join(train_config.output_dir, "latest.pt"))
                print(f"  [Save] Checkpoint saved at step {global_step}")
                # Keep only the latest checkpoint to save disk space
                old_ckpts = sorted(Path(train_config.output_dir).glob("checkpoint-*.pt"))
                for old in old_ckpts[:-1]:
                    old.unlink(missing_ok=True)
                    print(f"  [Clean] Removed old checkpoint: {old.name}")

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n  [!] Interrupted: {type(e).__name__}: {e}")
        save_checkpoint(model, optimizer, scheduler, scaler, stats, model_config, crash_path)
        stats.save(os.path.join(train_config.log_dir, "training_stats_partial.json"))
        print(f"  [Save] Crash saved at step {global_step} -> {crash_path}")
        print(f"  Resume: --resume {crash_path}")
        return model, stats

    # Final save
    final_path = os.path.join(train_config.output_dir, "final.pt")
    save_checkpoint(model, optimizer, scheduler, scaler, stats, model_config, final_path)
    save_checkpoint(model, optimizer, scheduler, scaler, stats, model_config,
                   os.path.join(train_config.output_dir, "latest.pt"))

    stats.save(os.path.join(train_config.log_dir, "training_stats.json"))

    if os.path.exists(crash_path):
        os.remove(crash_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"  Total steps: {global_step}")
    print(f"  Total tokens: {stats.total_tokens:,}")
    print(f"  Best eval loss: {stats.best_eval_loss:.4f}")
    print(f"  Time: {stats.format_time(time.time() - stats.start_time)}")
    print(f"  Model saved: {final_path}")
    print(f"{'='*60}")

    return model, stats


@torch.no_grad()
def evaluate(model, eval_loader, device, config, max_batches=20):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_bf16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()
            total_ce += outputs["ce_loss"].item()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_ce = total_ce / max(n_batches, 1)
    perplexity = math.exp(min(avg_ce, 20))

    return avg_loss, perplexity


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NeuroCoder")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "small", "max", "cloud"],
                        help="Model configuration preset")
    parser.add_argument("--data", type=str, default="",
                        help="Path to training data")
    parser.add_argument("--eval-data", type=str, default="",
                        help="Path to evaluation data")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=0,
                        help="Override max training steps")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint")
    parser.add_argument("--download-data", action="store_true",
                        help="Download sample Python code data")
    args = parser.parse_args()

    # Select config
    config_map = {
        "default": CONFIG_4060,
        "small": CONFIG_SMALL,
        "max": CONFIG_MAX,
        "cloud": CONFIG_CLOUD,
    }

    # When resuming, load the checkpoint's original config to avoid architecture mismatch
    if args.resume:
        print(f"Loading checkpoint config from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            model_config = ckpt["config"]
            # Fill in any new config fields that the old checkpoint doesn't have
            import dataclasses
            cli_config = config_map[args.config]
            for field in dataclasses.fields(cli_config):
                if not hasattr(model_config, field.name):
                    setattr(model_config, field.name, getattr(cli_config, field.name))
            print(f"  Using checkpoint config: d_model={model_config.d_model}, "
                  f"n_blocks={model_config.n_blocks}, n_experts={model_config.n_experts}")
            print(f"  focal_loss_gamma={model_config.focal_loss_gamma}")
        else:
            model_config = config_map[args.config]
            print(f"  No config in checkpoint, using --config={args.config}")
    else:
        model_config = config_map[args.config]

    if args.steps > 0:
        model_config.max_steps = args.steps

    # Training config
    train_cfg = TrainingConfig(
        train_data_path=args.data,
        eval_data_path=args.eval_data,
        output_dir=args.output,
        log_dir=os.path.join(args.output, "..", "logs"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=False,  # torch.compile too slow on first run
        resume_from=args.resume if args.resume else None,
    )

    # Prepare data
    if args.download_data or not args.data:
        print("Downloading sample data...")
        data_path = download_sample_data()
        train_cfg.train_data_path = data_path
    else:
        data_path = args.data

    # Create tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = CodeTokenizer()  # uses relative paths

    # Create dataloaders
    print("\nPreparing data...")
    train_loader, eval_loader = create_dataloaders(
        train_path=data_path,
        tokenizer=tokenizer,
        config=model_config,
        eval_path=args.eval_data if args.eval_data else None,
    )

    # Train
    model, stats = train(
        model_config=model_config,
        train_config=train_cfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    return model


if __name__ == "__main__":
    main()
