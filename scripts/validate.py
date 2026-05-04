"""Run validation on one checkpoint."""

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oceanlens.utils import load_config, seed_everything
from oceanlens.models.system import OceanLensSystem
from oceanlens.data.datamodule import OceanDataModule
import pytorch_lightning as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True, choices=["cno", "fm"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cno_ckpt", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.variant)
    seed_everything(cfg.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = OceanLensSystem.load_from_checkpoint(args.ckpt, cfg=cfg, phase=args.phase)
    if args.cno_ckpt:
        model.load_cno_checkpoint(args.cno_ckpt)

    dm = OceanDataModule(cfg, phase=args.phase)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed")
    trainer.validate(model, dm)


if __name__ == "__main__":
    main()
