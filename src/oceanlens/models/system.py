"""Unified Lightning module for all variants."""

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .cno import CNO2d
from .fm_unet import FlowMatchingUNet
from .variants import build_fm_condition, build_mu, compute_fm_target
from ..eval.metrics import compute_per_variable_metrics
from ..losses.cno import cno_loss
from ..losses.fm import flow_matching_loss


class OceanLensSystem(pl.LightningModule):
    """Train CNO or FM with one unified class."""

    def __init__(self, cfg, phase="cno"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.variables = list(cfg.variables)
        self.save_hyperparameters(ignore=["cfg"])

        self.cno = None
        if cfg.use_cno:
            self.cno = CNO2d(
                in_channels=cfg.cno.in_channels,
                out_channels=cfg.cno.out_channels,
                hidden_channels=list(cfg.cno.hidden_channels),
                n_res_blocks=cfg.cno.n_res_blocks,
                kernel_size=cfg.cno.kernel_size,
            )

        self.fm = None
        if phase == "fm":
            self.fm = FlowMatchingUNet(
                in_channels=cfg.fm_in_channels,
                out_channels=cfg.fm.out_channels,
                hidden_channels=list(cfg.fm.hidden_channels),
                time_dim=cfg.fm.time_dim,
                n_res=cfg.fm.n_res,
                attn_heads=cfg.fm.attn_heads,
            )

            if self.cno is not None:
                self.cno.eval()
                for param in self.cno.parameters():
                    param.requires_grad = False

    def _load_submodule_state(self, module, ckpt_path, prefixes, module_name):
        """Load one submodule from a Lightning or plain PyTorch checkpoint."""
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            cleaned = {}
            for key, value in state["state_dict"].items():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        cleaned[key[len(prefix) :]] = value
                        break
            state = cleaned or state["state_dict"]

        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            missing_preview = ", ".join(missing[:8])
            unexpected_preview = ", ".join(unexpected[:8])
            raise RuntimeError(
                f"Checkpoint {ckpt_path} is not compatible with {module_name}. "
                f"Missing keys ({len(missing)}): {missing_preview}. "
                f"Unexpected keys ({len(unexpected)}): {unexpected_preview}."
            )

    def load_cno_checkpoint(self, ckpt_path):
        """Load a pretrained CNO."""
        self._load_submodule_state(self.cno, ckpt_path, prefixes=("cno.", "model.cno."), module_name="CNO")
        self.cno.eval()

    def load_fm_checkpoint(self, ckpt_path):
        """Load pretrained FM weights without restoring the CNO."""
        if self.fm is None:
            raise ValueError("Cannot load FM checkpoint when phase is not 'fm'")
        self._load_submodule_state(self.fm, ckpt_path, prefixes=("fm.", "model.fm."), module_name="FM")

    def _compute_mu(self, lr):
        """Run the deterministic branch."""
        cno_output = self.cno(lr)
        return build_mu(cno_output, lr, self.cfg)

    @torch.no_grad()
    def sample(self, lr, mask, n_steps=None):
        """Generate one sample."""
        if n_steps is None:
            n_steps = self.cfg.fm.num_inference_steps

        mu = self._compute_mu(lr) if self.cfg.use_cno else None
        condition = build_fm_condition(mu, lr, mask, self.cfg)
        x = torch.randn_like(lr)
        dt = 1.0 / n_steps
        for index in range(n_steps):
            t = torch.full((x.shape[0],), index * dt, device=x.device)
            v = self.fm(x, t, condition)
            x = x + v * dt

        if getattr(self.cfg, "fm_target", None) == "residual_lr":
            return (lr + x) * mask
        if self.cfg.variant == "ablation":
            return x * mask
        return (mu + x) * mask

    @torch.no_grad()
    def sample_ensemble(self, lr, mask, n_members=8, n_steps=None):
        """Generate multiple samples."""
        if n_steps is None:
            n_steps = self.cfg.fm.num_inference_steps
        members = []
        for _ in range(n_members):
            members.append(self.sample(lr, mask, n_steps=n_steps))
        return torch.stack(members, dim=0)

    def training_step(self, batch, batch_idx):
        del batch_idx
        lr = batch["lr"]
        hr = batch["hr"]
        mask = batch["mask"]

        if self.phase == "cno":
            pred = self.cno(lr)
            loss = cno_loss(pred, hr, lr, mask, self.cfg)
            self.log("train/cno_loss", loss, prog_bar=True)
            self.log("train/loss", loss, prog_bar=True)
            return loss

        mu = self._compute_mu(lr) if self.cfg.use_cno else None
        target = compute_fm_target(hr, mu, lr, mask, self.cfg)
        condition = build_fm_condition(mu, lr, mask, self.cfg)
        loss = flow_matching_loss(self.fm, target, condition, mask=mask, cfg=self.cfg)
        self.log("train/fm_loss", loss, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]
        mask = batch["mask"]

        if self.phase == "cno":
            pred = self.cno(lr)
            loss = cno_loss(pred, hr, lr, mask, self.cfg)
            self.log("val/cno_loss", loss, prog_bar=True, sync_dist=True)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)

            full_pred = pred if self.cfg.cno_mode == "direct" else (lr + pred) * mask
            metrics = compute_per_variable_metrics(full_pred, hr * mask, mask, self.variables)
            for var_name, var_metrics in metrics.items():
                for metric_name, value in var_metrics.items():
                    self.log(f"val/{var_name}/{metric_name}", value, sync_dist=True)
            return loss

        mu = self._compute_mu(lr) if self.cfg.use_cno else None
        target = compute_fm_target(hr, mu, lr, mask, self.cfg)
        condition = build_fm_condition(mu, lr, mask, self.cfg)
        loss = flow_matching_loss(self.fm, target, condition, mask=mask, cfg=self.cfg)
        self.log("val/fm_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            pred = self.sample(lr, mask, n_steps=self.cfg.fm.num_inference_steps)
            metrics = compute_per_variable_metrics(pred, hr * mask, mask, self.variables)
            for var_name, var_metrics in metrics.items():
                for metric_name, value in var_metrics.items():
                    self.log(f"val/{var_name}/{metric_name}", value, sync_dist=True)
        return loss

    def configure_optimizers(self):
        tcfg = self.cfg.training[self.phase]
        params = self.cno.parameters() if self.phase == "cno" else self.fm.parameters()
        optimizer = AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=tcfg.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=tcfg.max_epochs - tcfg.warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[tcfg.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
