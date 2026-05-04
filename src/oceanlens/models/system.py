"""Unified Lightning module for all variants."""

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .cno import CNO2d
from .fm_dit import FlowMatchingDiT
from .fm_song_unet import SongUNetPosEmbd
from .fm_unet import FlowMatchingUNet
from .fm_swin import FlowMatchingSwinUNet
from .variants import build_fm_condition, build_mu, compute_fm_target
from ..eval.metrics import compute_per_variable_metrics
from ..losses.cno import cno_loss
from ..losses.diffusion import diffusion_bridge_loss, edm_denoising_loss
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
            fm_arch = getattr(cfg.fm, "arch", "unet")
            if fm_arch == "swin_unet":
                model_cfg = getattr(cfg, "model", {})
                condition_cfg = getattr(cfg, "condition", {})
                self.fm = FlowMatchingSwinUNet(
                    x_channels=int(getattr(cfg.fm, "x_channels", cfg.fm.out_channels)),
                    condition_channels=int(getattr(condition_cfg, "condition_channels", cfg.fm_in_channels - cfg.fm.out_channels)),
                    dynamic_channels=int(getattr(condition_cfg, "dynamic_channels", 0)),
                    out_channels=cfg.fm.out_channels,
                    patch_size=int(getattr(model_cfg, "patch_size", 4)),
                    embed_dim=int(getattr(model_cfg, "embed_dim", 96)),
                    depths=list(getattr(model_cfg, "depths", [2, 2, 4, 2])),
                    heads=list(getattr(model_cfg, "heads", [3, 6, 12, 6])),
                    window_size=int(getattr(model_cfg, "window_size", 8)),
                    time_dim=cfg.fm.time_dim,
                    cross_attn_bottleneck=bool(getattr(model_cfg, "cross_attn_bottleneck", True)),
                )
            elif fm_arch == "dit_pixel":
                model_cfg = getattr(cfg, "model", {})
                condition_channels = int(getattr(model_cfg, "condition_channels", cfg.fm_in_channels - cfg.fm.out_channels))
                self.fm = FlowMatchingDiT(
                    x_channels=int(getattr(cfg.fm, "x_channels", cfg.fm.out_channels)),
                    condition_channels=condition_channels,
                    out_channels=cfg.fm.out_channels,
                    patch_size=int(getattr(model_cfg, "patch_size", 4)),
                    embed_dim=int(getattr(model_cfg, "embed_dim", 384)),
                    depth=int(getattr(model_cfg, "depth", 12)),
                    heads=int(getattr(model_cfg, "heads", 8)),
                    mlp_ratio=float(getattr(model_cfg, "mlp_ratio", 4.0)),
                    time_dim=cfg.fm.time_dim,
                    qk_norm=bool(getattr(model_cfg, "qk_norm", True)),
                    rope_2d=bool(getattr(model_cfg, "rope_2d", True)),
                )
            elif fm_arch == "song_unet":
                model_cfg = getattr(cfg, "model", {})
                condition_channels = int(getattr(model_cfg, "condition_channels", cfg.fm_in_channels - cfg.fm.out_channels))
                self.fm = SongUNetPosEmbd(
                    x_channels=int(getattr(cfg.fm, "x_channels", cfg.fm.out_channels)),
                    condition_channels=condition_channels,
                    out_channels=cfg.fm.out_channels,
                    img_resolution=int(getattr(model_cfg, "img_resolution", getattr(cfg.data, "patch_size", 256))),
                    model_channels=int(getattr(model_cfg, "model_channels", 128)),
                    channel_mult=list(getattr(model_cfg, "channel_mult", [1, 2, 2, 2])),
                    channel_mult_emb=int(getattr(model_cfg, "channel_mult_emb", 4)),
                    num_blocks=int(getattr(model_cfg, "num_blocks", 4)),
                    attn_resolutions=list(getattr(model_cfg, "attn_resolutions", [16, 32])),
                    time_dim=cfg.fm.time_dim,
                    gridtype=getattr(model_cfg, "gridtype", "learnable"),
                    n_grid_channels=int(getattr(model_cfg, "N_grid_channels", getattr(model_cfg, "n_grid_channels", 4))),
                    dropout=float(getattr(model_cfg, "dropout", 0.0)),
                    attn_heads=int(getattr(model_cfg, "attn_heads", getattr(cfg.fm, "attn_heads", 8))),
                )
            else:
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

    def _recoarsen_cfg(self):
        loss_cfg = getattr(self.cfg, "loss", None)
        return getattr(loss_cfg, "recoarsen", None) if loss_cfg is not None else None

    def _recoarsen_weight(self):
        recoarsen_cfg = self._recoarsen_cfg()
        if recoarsen_cfg is None or not getattr(recoarsen_cfg, "enabled", False):
            return 0.0
        base_weight = float(getattr(recoarsen_cfg, "weight", 0.0))
        warmup_epochs = int(getattr(recoarsen_cfg, "warmup_epochs", 0))
        if warmup_epochs <= 0:
            return base_weight
        progress = min((self.current_epoch + 1) / warmup_epochs, 1.0)
        return base_weight * progress

    def _loss_weight(self, name):
        loss_cfg = getattr(self.cfg, "loss", None)
        cfg = getattr(loss_cfg, name, None) if loss_cfg is not None else None
        if cfg is None or not getattr(cfg, "enabled", False):
            return 0.0
        base_weight = float(getattr(cfg, "weight", 0.0))
        warmup_epochs = int(getattr(cfg, "warmup_epochs", 0))
        if warmup_epochs <= 0:
            return base_weight
        progress = min((self.current_epoch + 1) / warmup_epochs, 1.0)
        return base_weight * progress

    def _downsample_with_mask(self, field, mask, target_size):
        height, width = field.shape[-2:]
        target_h, target_w = target_size
        factor_h = max(1, int(round(height / target_h)))
        factor_w = max(1, int(round(width / target_w)))
        down_field = F.avg_pool2d(
            field * mask,
            kernel_size=(factor_h, factor_w),
            stride=(factor_h, factor_w),
            ceil_mode=True,
        )
        down_mask = F.avg_pool2d(
            mask,
            kernel_size=(factor_h, factor_w),
            stride=(factor_h, factor_w),
            ceil_mode=True,
        ).clamp(min=1.0e-6)
        return down_field / down_mask, down_mask

    def _recoarsen_loss(self, fm_out, mu, lr, lr_native, mask):
        recoarsen_cfg = self._recoarsen_cfg()
        if recoarsen_cfg is None or not getattr(recoarsen_cfg, "enabled", False):
            return None

        t_expand = fm_out["t"][:, None, None, None]
        x1_pred = fm_out["x_t"] + (1.0 - t_expand) * fm_out["v_pred"]
        if getattr(self.cfg, "fm_target", None) == "residual_lr":
            pred_hr = (lr + x1_pred) * mask
        elif self.cfg.variant == "ablation":
            pred_hr = x1_pred * mask
        else:
            pred_hr = (mu.detach() + x1_pred) * mask

        target_size = lr_native.shape[-2:]
        pred_pool, pooled_mask = self._downsample_with_mask(pred_hr, mask, target_size)
        if pred_pool.shape[-2:] != lr_native.shape[-2:]:
            pred_pool = F.interpolate(pred_pool, size=lr_native.shape[-2:], mode="bilinear", align_corners=False)
            pooled_mask = F.interpolate(pooled_mask, size=lr_native.shape[-2:], mode="nearest")
        lr_pool = lr_native

        loss = (pred_pool - lr_pool).square()
        weights = getattr(getattr(self.cfg, "loss", None), "var_weights", None)
        if weights is not None:
            weights = torch.as_tensor(list(weights), device=loss.device, dtype=loss.dtype).view(1, -1, 1, 1)
            loss = loss * weights

        valid = (pooled_mask >= float(getattr(recoarsen_cfg, "min_ocean_fraction", 0.5))).to(loss.dtype)
        if valid.shape[1] == 1 and loss.shape[1] != 1:
            valid = valid.expand(-1, loss.shape[1], -1, -1)
        denom = valid.sum().clamp(min=1.0)
        return (loss * valid).sum() / denom

    def _compose_prediction(self, x1_pred, mu, lr, mask):
        if getattr(self.cfg, "fm_target", None) == "residual_lr":
            return (lr + x1_pred) * mask
        if self.cfg.variant == "ablation":
            return x1_pred * mask
        return (mu.detach() + x1_pred) * mask

    def _predicted_x1(self, fm_out):
        if "pred_x1" in fm_out:
            return fm_out["pred_x1"]
        t_expand = fm_out["t"][:, None, None, None]
        return fm_out["x_t"] + (1.0 - t_expand) * fm_out["v_pred"]

    def _recoarsen_loss_from_pred(self, pred_hr, lr_native, mask):
        recoarsen_cfg = self._recoarsen_cfg()
        if recoarsen_cfg is None or not getattr(recoarsen_cfg, "enabled", False):
            return None
        pred_pool, pooled_mask = self._downsample_with_mask(pred_hr, mask, lr_native.shape[-2:])
        if pred_pool.shape[-2:] != lr_native.shape[-2:]:
            pred_pool = F.interpolate(pred_pool, size=lr_native.shape[-2:], mode="bilinear", align_corners=False)
            pooled_mask = F.interpolate(pooled_mask, size=lr_native.shape[-2:], mode="nearest")
        loss = (pred_pool - lr_native).square()
        weights = getattr(getattr(self.cfg, "loss", None), "var_weights", None)
        if weights is not None:
            weights = torch.as_tensor(list(weights), device=loss.device, dtype=loss.dtype).view(1, -1, 1, 1)
            loss = loss * weights
        valid = (pooled_mask >= float(getattr(recoarsen_cfg, "min_ocean_fraction", 0.5))).to(loss.dtype)
        if valid.shape[1] == 1 and loss.shape[1] != 1:
            valid = valid.expand(-1, loss.shape[1], -1, -1)
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)

    def _gradient_loss(self, pred_hr, hr, mask):
        grad_cfg = getattr(getattr(self.cfg, "loss", None), "gradient", None)
        if grad_cfg is None or not getattr(grad_cfg, "enabled", False):
            return None
        variables = list(getattr(grad_cfg, "variables", []))
        if not variables:
            indices = list(range(pred_hr.shape[1]))
        else:
            indices = [self.variables.index(name) for name in variables if name in self.variables]
        if not indices:
            return None

        losses = []
        for index in indices:
            pred = pred_hr[:, index : index + 1]
            target = hr[:, index : index + 1]
            pred_grad = self._gradient_magnitude(pred)
            target_grad = self._gradient_magnitude(target)
            loss = (pred_grad - target_grad).square()
            denom = mask.sum().clamp(min=1.0)
            losses.append((loss * mask).sum() / denom)
        return torch.stack(losses).mean()

    @staticmethod
    def _gradient_magnitude(field):
        grad_y = torch.zeros_like(field)
        grad_x = torch.zeros_like(field)
        grad_y[:, :, 1:-1, :] = 0.5 * (field[:, :, 2:, :] - field[:, :, :-2, :])
        grad_y[:, :, 0:1, :] = field[:, :, 1:2, :] - field[:, :, 0:1, :]
        grad_y[:, :, -1:, :] = field[:, :, -1:, :] - field[:, :, -2:-1, :]
        grad_x[:, :, :, 1:-1] = 0.5 * (field[:, :, :, 2:] - field[:, :, :, :-2])
        grad_x[:, :, :, 0:1] = field[:, :, :, 1:2] - field[:, :, :, 0:1]
        grad_x[:, :, :, -1:] = field[:, :, :, -1:] - field[:, :, :, -2:-1]
        return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)

    def _spectral_loss(self, pred_hr, hr, mask):
        spectral_cfg = getattr(getattr(self.cfg, "loss", None), "spectral", None)
        if spectral_cfg is None or not getattr(spectral_cfg, "enabled", False):
            return None
        pred = pred_hr * mask
        target = hr * mask
        pred_power = torch.fft.rfft2(pred.float(), norm="ortho").abs().square()
        target_power = torch.fft.rfft2(target.float(), norm="ortho").abs().square()
        if getattr(spectral_cfg, "log_power", True):
            pred_power = torch.log1p(pred_power)
            target_power = torch.log1p(target_power)
        loss = (pred_power - target_power).square()
        weights = getattr(getattr(self.cfg, "loss", None), "var_weights", None)
        if weights is not None:
            weights = torch.as_tensor(list(weights), device=loss.device, dtype=loss.dtype).view(1, -1, 1, 1)
            loss = loss * weights
        return loss.mean().to(dtype=pred_hr.dtype)

    def _masked_rmse(self, pred, target, mask):
        valid = mask.to(device=pred.device, dtype=pred.dtype)
        if valid.shape[1] == 1 and pred.shape[1] != 1:
            valid = valid.expand(-1, pred.shape[1], -1, -1)
        denom = valid.sum().clamp(min=1.0)
        return (((pred - target).square() * valid).sum() / denom).sqrt()

    def _validation_diagnostics(self, pred, hr, mask, batch):
        target = hr * mask
        residual = (pred - target) * mask
        ocean = mask.to(dtype=pred.dtype)
        denom = ocean.sum(dim=(1, 2, 3)).clamp(min=1.0)
        per_sample_mean = residual.sum(dim=(1, 2, 3)) / denom
        centered = residual - per_sample_mean[:, None, None, None]
        per_sample_std = ((centered.square() * ocean).sum(dim=(1, 2, 3)) / denom).sqrt().mean()
        mean_residual = residual.sum() / ocean.sum().clamp(min=1.0)
        self.log("val/per_sample_std", per_sample_std, sync_dist=True)
        self.log("val/mean_residual", mean_residual, sync_dist=True)

        lr_native = batch.get("lr_native")
        if lr_native is not None:
            pred_pool, pooled_mask = self._downsample_with_mask(pred, mask, lr_native.shape[-2:])
            if pred_pool.shape[-2:] != lr_native.shape[-2:]:
                pred_pool = F.interpolate(pred_pool, size=lr_native.shape[-2:], mode="bilinear", align_corners=False)
                pooled_mask = F.interpolate(pooled_mask, size=lr_native.shape[-2:], mode="nearest")
            valid = (pooled_mask >= 0.5).to(dtype=pred.dtype)
            if valid.shape[1] == 1 and pred_pool.shape[1] != 1:
                valid = valid.expand(-1, pred_pool.shape[1], -1, -1)
            coarsen_error = ((pred_pool - lr_native).square() * valid).sum() / valid.sum().clamp(min=1.0)
            self.log("val/coarsen_error", coarsen_error.sqrt(), sync_dist=True)

        dist_coast = batch.get("dist_coast")
        if dist_coast is not None:
            coastal = ((dist_coast > 0.0) & (dist_coast <= 0.08)).to(dtype=pred.dtype) * mask
            self.log("val/coastal_rmse", self._masked_rmse(pred, target, coastal), sync_dist=True)

        lat = batch.get("lat")
        lon_sin = batch.get("lon_sin")
        lon_cos = batch.get("lon_cos")
        if lat is not None:
            arctic = (lat >= (60.0 / 90.0)).to(dtype=pred.dtype) * mask
            self.log("val/arctic_rmse", self._masked_rmse(pred, target, arctic), sync_dist=True)
        if lat is not None and lon_sin is not None and lon_cos is not None:
            lon = torch.atan2(lon_sin, lon_cos) * (180.0 / torch.pi)
            lat_deg = lat * 90.0
            med = (
                (lat_deg >= 30.0)
                & (lat_deg <= 46.5)
                & (lon >= -6.0)
                & (lon <= 37.0)
            ).to(dtype=pred.dtype) * mask
            self.log("val/mediterranean_rmse", self._masked_rmse(pred, target, med), sync_dist=True)

    def _extra_condition(self, batch):
        return {
            "siconc": batch.get("siconc"),
            "dist_coast": batch.get("dist_coast"),
            "region_id": batch.get("region_id"),
            "lat": batch.get("lat"),
            "lon_sin": batch.get("lon_sin"),
            "lon_cos": batch.get("lon_cos"),
        }

    def _fm_objective(self, hr, lr, lr_native, mask, mu, condition, stage):
        objective = getattr(self.cfg.fm, "objective", "flow_matching")
        if objective == "edm_denoising":
            target = compute_fm_target(hr, mu, lr, mask, self.cfg)
            fm_out = edm_denoising_loss(self.fm, target, condition, mask=mask, cfg=self.cfg, return_details=True)
            pred_hr = self._compose_prediction(fm_out["pred_x1"], mu, lr, mask)
        elif objective == "diffusion_bridge":
            if mu is None:
                raise ValueError("diffusion_bridge objective requires use_cno=true")
            fm_out = diffusion_bridge_loss(self.fm, hr, mu, condition, mask=mask, cfg=self.cfg, return_details=True)
            pred_hr = fm_out["pred_x1"] * mask
        else:
            target = compute_fm_target(hr, mu, lr, mask, self.cfg)
            fm_out = flow_matching_loss(self.fm, target, condition, mask=mask, cfg=self.cfg, return_details=True)
            pred_hr = self._compose_prediction(self._predicted_x1(fm_out), mu, lr, mask)

        loss = fm_out["loss"]
        recoarsen_loss = self._recoarsen_loss_from_pred(pred_hr, lr_native, mask) if lr_native is not None else None
        if recoarsen_loss is not None:
            weight = self._recoarsen_weight()
            self.log(f"{stage}/recoarsen_loss", recoarsen_loss, prog_bar=False, sync_dist=(stage == "val"))
            self.log(f"{stage}/recoarsen_weight", weight, prog_bar=False, sync_dist=(stage == "val"))
            loss = loss + weight * recoarsen_loss
        spectral_loss = self._spectral_loss(pred_hr, hr, mask)
        if spectral_loss is not None:
            weight = self._loss_weight("spectral")
            self.log(f"{stage}/spectral_loss", spectral_loss, prog_bar=False, sync_dist=(stage == "val"))
            self.log(f"{stage}/spectral_weight", weight, prog_bar=False, sync_dist=(stage == "val"))
            loss = loss + weight * spectral_loss
        gradient_loss = self._gradient_loss(pred_hr, hr, mask)
        if gradient_loss is not None:
            weight = self._loss_weight("gradient")
            self.log(f"{stage}/gradient_loss", gradient_loss, prog_bar=False, sync_dist=(stage == "val"))
            self.log(f"{stage}/gradient_weight", weight, prog_bar=False, sync_dist=(stage == "val"))
            loss = loss + weight * gradient_loss
        return loss, fm_out, pred_hr

    @torch.no_grad()
    def sample(self, lr, mask, n_steps=None, extra=None):
        """Generate one sample."""
        if n_steps is None:
            n_steps = self.cfg.fm.num_inference_steps

        mu = self._compute_mu(lr) if self.cfg.use_cno else None
        condition = build_fm_condition(mu, lr, mask, self.cfg, extra=extra)
        objective = getattr(self.cfg.fm, "objective", "flow_matching")
        if objective == "edm_denoising":
            edm_cfg = getattr(self.cfg, "edm", getattr(self.cfg, "diffusion", None))
            sigma_min = float(getattr(edm_cfg, "sigma_min", 0.002))
            sigma_max = float(getattr(edm_cfg, "sigma_max", 80.0))
            rho = float(getattr(edm_cfg, "rho", 7.0))
            ramp = torch.linspace(0, 1, n_steps, device=lr.device)
            sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
            x = torch.randn_like(lr) * sigmas[0]
            for index in range(n_steps):
                sigma = sigmas[index]
                sigma_next = sigmas[index + 1]
                labels = sigma.log().expand(x.shape[0])
                denoised = self.fm(x, labels, condition)
                d_cur = (x - denoised) / sigma.clamp(min=1.0e-6)
                x = x + (sigma_next - sigma) * d_cur
            return self._compose_prediction(x, mu, lr, mask)

        if objective == "diffusion_bridge":
            bridge_cfg = getattr(self.cfg, "bridge", None)
            sigma_min = float(getattr(bridge_cfg, "sigma_min", 0.01))
            sigma_max = float(getattr(bridge_cfg, "sigma_max", 1.0))
            sigma = (sigma_min + sigma_max) * 0.5
            x = (mu + sigma * torch.randn_like(mu)) * mask
            dt = 1.0 / n_steps
            for index in range(n_steps):
                t = torch.full((x.shape[0],), index * dt, device=x.device)
                v = self.fm(x, t, condition)
                x = (x + v * dt) * mask
            return x * mask

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
    def sample_ensemble(self, lr, mask, n_members=8, n_steps=None, extra=None):
        """Generate multiple samples."""
        if n_steps is None:
            n_steps = self.cfg.fm.num_inference_steps
        members = []
        for _ in range(n_members):
            members.append(self.sample(lr, mask, n_steps=n_steps, extra=extra))
        return torch.stack(members, dim=0)

    def training_step(self, batch, batch_idx):
        del batch_idx
        lr = batch["lr"]
        lr_native = batch.get("lr_native")
        hr = batch["hr"]
        mask = batch["mask"]
        extra = self._extra_condition(batch)

        if self.phase == "cno":
            pred = self.cno(lr)
            loss = cno_loss(pred, hr, lr, mask, self.cfg)
            self.log("train/cno_loss", loss, prog_bar=True)
            self.log("train/loss", loss, prog_bar=True)
            return loss

        mu = self._compute_mu(lr) if self.cfg.use_cno else None
        condition = build_fm_condition(mu, lr, mask, self.cfg, extra=extra)
        loss, fm_out, _ = self._fm_objective(hr, lr, lr_native, mask, mu, condition, "train")
        self.log("train/fm_loss", fm_out["loss"], prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch["lr"]
        lr_native = batch.get("lr_native")
        hr = batch["hr"]
        mask = batch["mask"]
        extra = self._extra_condition(batch)

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
        condition = build_fm_condition(mu, lr, mask, self.cfg, extra=extra)
        loss, fm_out, pred_hr = self._fm_objective(hr, lr, lr_native, mask, mu, condition, "val")
        self.log("val/fm_loss", fm_out["loss"], prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            pred = self.sample(lr, mask, n_steps=self.cfg.fm.num_inference_steps, extra=extra)
            metrics = compute_per_variable_metrics(pred, hr * mask, mask, self.variables)
            for var_name, var_metrics in metrics.items():
                for metric_name, value in var_metrics.items():
                    self.log(f"val/{var_name}/{metric_name}", value, sync_dist=True)
            self._validation_diagnostics(pred, hr, mask, batch)
        return loss

    def configure_optimizers(self):
        tcfg = self.cfg.training[self.phase]
        params = self.cno.parameters() if self.phase == "cno" else self.fm.parameters()
        optimizer = AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=tcfg.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=tcfg.max_epochs - tcfg.warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[tcfg.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
