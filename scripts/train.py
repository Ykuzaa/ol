"""Train one variant and one phase."""

import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oceanlens.utils import load_config, seed_everything
from oceanlens.models.system import OceanLensSystem
from oceanlens.data.datamodule import OceanDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True,
                        choices=["cno", "fm"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cno_ckpt", type=str, default=None,
                        help="CNO checkpoint path (required for FM phase with CNO)")
    parser.add_argument("--fm_ckpt", type=str, default=None,
                        help="FM checkpoint path used to initialize FM fine-tuning")
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--fast_dev_run", type=int, default=0,
                        help="Run a short Lightning smoke test when > 0")
    args = parser.parse_args()

    # Load config
    config_dir = args.config_dir or str(Path(__file__).parent.parent / "configs")
    cfg = load_config(args.variant, config_dir)
    if args.fast_dev_run:
        cfg.data.num_workers = 0

    # Validate
    if args.phase == "fm" and cfg.use_cno and args.cno_ckpt is None:
        parser.error("FM phase with CNO requires --cno_ckpt")
    if args.fm_ckpt is not None and args.phase != "fm":
        parser.error("--fm_ckpt can only be used with --phase fm")
    if args.phase == "cno" and not cfg.use_cno:
        parser.error(f"Variant {args.variant} has use_cno=False, no CNO to train")

    # Seed
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Set GPU (skip if already set by launcher script)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Output directory
    run_dir = Path(cfg.paths.output_dir) / args.variant / args.phase
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = OceanLensSystem(cfg, phase=args.phase)
    if args.cno_ckpt:
        model.load_cno_checkpoint(args.cno_ckpt)
    if args.fm_ckpt:
        model.load_fm_checkpoint(args.fm_ckpt)

    # Data
    dm = OceanDataModule(cfg, phase=args.phase)

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="logs",
    )

    # Callbacks
    tcfg = cfg.training[args.phase]
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename=f"{args.variant}-{args.phase}" + "-{epoch:03d}",
        monitor=cfg.logging.monitor,
        mode="min",
        save_top_k=cfg.logging.save_top_k,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=tcfg.max_epochs,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=tcfg.accumulate_grad_batches,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.logging.val_check_interval,
        callbacks=[checkpoint, lr_monitor],
        logger=logger,
        deterministic=True,
        precision=tcfg.precision,
        gradient_clip_val=float(getattr(tcfg, "gradient_clip_val", 1.0)),
        gradient_clip_algorithm=getattr(tcfg, "gradient_clip_algorithm", "norm"),
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
