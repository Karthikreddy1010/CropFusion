"""
model_training.py - Quantile regression & Neural CQR head training for Paper 3 (§4.2).

Implements:
1. Multi-Task Residual Neural CQR Network (NeuralCQRNet) with shared residual MLP backbone,
   dedicated Huber mean head (evaluated strictly for RMSE, MAE, R²), and dual quantile CQR heads.
2. Composite multi-objective loss combining Huber loss, 3 Pinball losses, Quantile Crossing penalty, and Width reg.
3. AdamW optimizer, CosineAnnealingLR scheduler, mixed precision, weight EMA, early stopping on validation pinball loss.
4. Per-epoch training diagnostics tracking loss components and gradient stability.
5. Baseline models (Ridge, Random Forest, LightGBM, CatBoost, XGBoost) for comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from utils import log_decision, set_global_seed, save_report

logger = logging.getLogger("paper3")


@dataclass
class QuantileModelSet:
    """Container for trained model set (mean, lower quantile, upper quantile)."""
    point_model: Any        # Evaluated strictly for RMSE, MAE, R2
    lower_model: Any        # τ = 0.05 evaluated for conformal intervals
    upper_model: Any        # τ = 0.95 evaluated for conformal intervals
    median_model: Any = None # τ = 0.50
    model_type: str = "NeuralCQR"
    feature_cols: List[str] = None
    is_neural: bool = True
    scaler: Any = None
    epochs_trained: int = 0
    best_epoch: int = 0
    training_history: List[Dict[str, float]] = None
    # Which training paradigm actually ran ("joint_end_to_end" /
    # "post_hoc_two_stage"). Set by train_neural_cqr; consumed by the ablation
    # and O4 to prove the two conditions took different code paths.
    training_paradigm: str = None
    paradigm_fingerprint: Dict[str, Any] = None
    # F0a: when the target was standardised for training, predictions come back
    # in standardised units and must be mapped to t/ha by predict_intervals.
    target_mu: float = 0.0
    target_sd: float = 1.0


# ─────────────────────────────────────────────────────────────
# Residual Block & Multi-Task Neural CQR Net (§2 & §3)
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual Block: Linear -> BatchNorm -> GELU -> Dropout -> Linear -> BatchNorm -> GELU + Skip."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act2 = nn.GELU()

        if in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.fc1(x)
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        if out.shape[0] > 1:
            out = self.bn2(out)
        out = out + residual
        return self.act2(out)


class NeuralCQRNet(nn.Module):
    """Deep Residual MLP backbone with 4 Multi-Task Prediction Heads (§2).

    Heads:
    - Head 1: y_mean (Huber Loss, evaluated EXCLUSIVELY for RMSE, MAE, R²)
    - Head 2: q0.05 (Pinball Loss tau=0.05, evaluated EXCLUSIVELY for Conformal Intervals)
    - Head 3: q0.50 (Pinball Loss tau=0.50)
    - Head 4: q0.95 (Pinball Loss tau=0.95, evaluated EXCLUSIVELY for Conformal Intervals)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = cfg.NEURAL_CQR_HIDDEN_DIMS,
        dropout_rate: float = cfg.NEURAL_CQR_DROPOUT,
        monotone_heads: bool = False,
    ) -> None:
        super().__init__()
        self.monotone_heads = bool(monotone_heads)
        self.in_proj = nn.Linear(input_dim, hidden_dims[0])
        self.in_bn = nn.BatchNorm1d(hidden_dims[0])
        self.in_act = nn.GELU()

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate))
        self.backbone = nn.Sequential(*blocks)

        # 4 Dedicated Prediction Heads
        final_dim = hidden_dims[-1]
        self.mean_head = nn.Linear(final_dim, 1)
        self.q05_head = nn.Linear(final_dim, 1)
        self.q50_head = nn.Linear(final_dim, 1)
        self.q95_head = nn.Linear(final_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        if h.shape[0] > 1:
            h = self.in_bn(h)
        h = self.in_act(h)

        h = self.backbone(h)

        y_mean = self.mean_head(h)

        if self.monotone_heads:
            # F1: q05 and q95 are non-negative offsets from q50, so
            # q05 <= q50 <= q95 holds by construction and the crossing
            # penalty has nothing left to correct.
            q50 = self.q50_head(h)
            q05 = q50 - nn.functional.softplus(self.q05_head(h))
            q95 = q50 + nn.functional.softplus(self.q95_head(h))
        else:
            # Head order is load-bearing: the order in which the heads are
            # called fixes the order in which their gradients accumulate into
            # the shared backbone, and changing it shifts results at ~1e-3.
            q05 = self.q05_head(h)
            q50 = self.q50_head(h)
            q95 = self.q95_head(h)

        return y_mean, q05, q50, q95


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Asymmetric Pinball (Quantile) Loss function."""
    err = y_true - y_pred
    return torch.max(tau * err, (tau - 1.0) * err).mean()


def cqr_composite_loss(
    y_true: torch.Tensor,
    y_mean: torch.Tensor,
    q05: torch.Tensor,
    q50: torch.Tensor,
    q95: torch.Tensor,
    lambda_pinball: float = cfg.LAMBDA_PINBALL,
    lambda_huber: float = cfg.LAMBDA_HUBER,
    lambda_crossing: float = cfg.LAMBDA_CROSSING,
    lambda_width: float = cfg.LAMBDA_WIDTH,
    huber_delta: float = cfg.HUBER_DELTA,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite Multi-Objective Loss Function (§3).

    L_total = lambda_pinball * [L_pinball(q05) + L_pinball(q50) + L_pinball(q95)]
            + lambda_huber * Huber(y_mean)
            + lambda_crossing * [ReLU(q05 - q50) + ReLU(q50 - q95)]
            + lambda_width * Width^2
    """
    # F0a: delta is in target units. Below delta Huber is quadratic (mean-
    # seeking, matching RMSE/R2); above it the gradient saturates and the head
    # drifts toward a conditional median. cfg.HUBER_DELTA_SIGMA keeps delta a
    # fixed multiple of the target sd when STANDARDIZE_TARGET is on.
    huber_l = nn.functional.huber_loss(y_mean, y_true, delta=float(huber_delta))

    loss_05 = pinball_loss(y_true, q05, 0.05)
    loss_50 = pinball_loss(y_true, q50, 0.50)
    loss_95 = pinball_loss(y_true, q95, 0.95)
    total_pinball = loss_05 + loss_50 + loss_95

    # Quantile crossing penalties
    crossing_penalty = torch.mean(torch.relu(q05 - q50)) + torch.mean(torch.relu(q50 - q95))

    # Interval width regularization
    widths = torch.relu(q95 - q05)
    width_reg = torch.mean(widths ** 2)

    total_loss = (
        lambda_pinball * total_pinball
        + lambda_huber * huber_l
        + lambda_crossing * crossing_penalty
        + lambda_width * width_reg
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_huber": float(huber_l.item()),
        "loss_pinball": float(total_pinball.item()),
        "loss_crossing": float(crossing_penalty.item()),
        "loss_width": float(width_reg.item()),
    }
    return total_loss, metrics


# ─────────────────────────────────────────────────────────────
# PyTorch Exponential Moving Average (EMA) Helper
# ─────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average (EMA) of model weights for enhanced stability.

    F0b fix (2026-09-17): the previous version aliased ``ema_model`` to the live
    module and tracked only ``parameters()``, so BatchNorm buffers were never
    averaged and ``apply_shadow`` was never called by any caller -- the shadow
    was recomputed every step and discarded. ``state_dict()`` now returns the
    averaged weights without mutating the live model, which is what the
    no-validation refit needs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.model = model
        self.ema_model = model  # retained for backward compatibility
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    def update(self) -> None:
        d = self.decay
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                sh = self.shadow[k]
                if sh.dtype.is_floating_point:
                    sh.mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    # Integer buffers (e.g. num_batches_tracked) are copied.
                    sh.copy_(v.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def apply_shadow(self, model: Optional[nn.Module] = None) -> None:
        (model if model is not None else self.model).load_state_dict(self.state_dict())


# ─────────────────────────────────────────────────────────────
# PyTorch Training Loop with Epoch Diagnostics (§4)
# ─────────────────────────────────────────────────────────────

def train_neural_cqr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_cols: List[str] = None,
    epochs: int = cfg.BASELINE_MAX_EPOCHS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    weight_decay: float = cfg.WEIGHT_DECAY,
    dropout_rate: float = cfg.NEURAL_CQR_DROPOUT,
    hidden_dims: Tuple[int, ...] = cfg.NEURAL_CQR_HIDDEN_DIMS,
    joint_training: bool = True,
    scaler: Any = None,
    lambda_pinball: float = cfg.LAMBDA_PINBALL,
    lambda_huber: float = cfg.LAMBDA_HUBER,
    lambda_crossing: float = cfg.LAMBDA_CROSSING,
    lambda_width: float = cfg.LAMBDA_WIDTH,
    early_stopping_mode: str = cfg.EARLY_STOPPING_MODE,
    patience: int = cfg.EARLY_STOPPING_PATIENCE,
    early_stopping: bool = True,
    seed: int = cfg.RANDOM_SEED,
    schedule_t_max: Optional[int] = None,
) -> QuantileModelSet:
    """Train the Neural CQR model under one of two genuinely different paradigms.

    ``joint_training=True`` (Condition B, the paper's proposed method)
        A single end-to-end optimization of the full composite objective
        ``L = lambda_huber*Huber + lambda_pinball*(pinball_05+50+95)
        + lambda_crossing*crossing + lambda_width*width``. Gradients from the
        three quantile heads flow back through the *shared backbone*, so the
        learned representation is shaped by the interval objectives as well as
        the point objective. This is the departure from Romano et al. (2019)
        claimed in §4.2 of the methodology.

    ``joint_training=False`` (Condition A, the post-hoc baseline)
        Two separate optimization stages with different parameter sets and
        different losses:

        1. *Point stage* — the backbone and the mean head are trained on the
           Huber objective **alone**. The quantile heads are excluded from the
           optimizer entirely and receive no gradient, so nothing about the
           interval objective can influence the representation. Early stopping
           uses the point metric.
        2. *Interval stage* — the backbone and mean head are frozen
           (``requires_grad=False`` and held in ``eval()`` so BatchNorm running
           statistics also stop moving) and only the three quantile heads are
           fitted, on pinball + crossing losses, on top of the now-fixed
           representation. This is the classical "calibrate on fixed model
           outputs" formulation.

    The two paths therefore differ in optimizer parameter groups, in the loss
    actually backpropagated, in the number of optimization stages, and in the
    resulting point predictions -- not merely in a label. ``training_paradigm``
    and ``paradigm_fingerprint`` on the returned object record which path ran.
    """
    if not joint_training:
        if bool(getattr(cfg, "STANDARDIZE_TARGET", False)):
            # The post-hoc (Condition A) path trains on the raw target. Running
            # the O4 joint-vs-post-hoc comparison with F0a enabled would change
            # two things at once (D3), so standardisation must be extended to
            # this path before that comparison is rerun.
            logger.warning("STANDARDIZE_TARGET is on but the post-hoc path is not standardised; "
                           "joint-vs-post-hoc comparisons are not like-for-like until this is addressed.")
        return _train_neural_cqr_posthoc(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            feature_cols=feature_cols, epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, dropout_rate=dropout_rate, hidden_dims=hidden_dims,
            scaler=scaler, lambda_pinball=lambda_pinball, lambda_huber=lambda_huber,
            lambda_crossing=lambda_crossing, early_stopping_mode=early_stopping_mode,
            patience=patience, early_stopping=early_stopping, seed=seed,
        )

    set_global_seed(seed)
    mode_str = "Joint End-to-End"
    has_val = (X_val is not None and y_val is not None and early_stopping)
    logger.info("Training PyTorch Multi-Task Neural CQR Net (%s mode, §4.2, seed=%d, es_mode=%s, early_stopping=%s)",
                mode_str, seed, early_stopping_mode if has_val else "none", has_val)

    # ── F0a: target standardisation, computed on the TRAINING array only ──
    # y_train here is already the modelling target (the detrended anomaly when
    # cfg.DETREND_TARGET is on), so mu/sd never see DEV, CAL or TEST rows.
    _standardize = bool(getattr(cfg, "STANDARDIZE_TARGET", False))
    if _standardize:
        target_mu = float(np.mean(y_train))
        target_sd = float(np.std(y_train))
        if not np.isfinite(target_sd) or target_sd < 1e-8:
            target_sd = 1.0
        huber_delta = float(getattr(cfg, "HUBER_DELTA_SIGMA", 1.345))
        logger.info("  F0a: target standardised on train (mu=%.4f, sd=%.4f); huber delta=%.3f sigma",
                    target_mu, target_sd, huber_delta)
    else:
        target_mu, target_sd = 0.0, 1.0
        huber_delta = float(getattr(cfg, "HUBER_DELTA", 1.0))

    y_train_model = (np.asarray(y_train, dtype=float) - target_mu) / target_sd

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCQRNet(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate,
        monotone_heads=bool(getattr(cfg, "NEURAL_MONOTONE_HEADS", False)),
    ).to(device)
    ema = ModelEMA(model, decay=float(getattr(cfg, "EMA_DECAY", 0.99)))

    t_X_tr = torch.tensor(X_train, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train_model, dtype=torch.float32).unsqueeze(1)
    if has_val:
        y_val_model = (np.asarray(y_val, dtype=float) - target_mu) / target_sd
        t_X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
        t_y_va = torch.tensor(y_val_model, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_np = np.asarray(y_val, dtype=float).flatten()
    else:
        t_X_va, t_y_va, y_val_np = None, None, None

    # Reproducible DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # F0c: T_max defaults to this run's epoch budget (legacy behaviour). The
    # final refit passes the DEV run's budget via schedule_t_max so that the
    # epoch count selected on DEV is reached along the same LR trajectory.
    _t_max = int(schedule_t_max) if schedule_t_max else int(epochs)
    _eta_min = float(getattr(cfg, "LR_ETA_MIN", 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, _t_max), eta_min=_eta_min)

    best_stop_metric = float("inf")
    best_weights = None
    best_epoch = 1
    patience_counter = 0

    init_pinball: Optional[float] = None
    init_rmse: Optional[float] = None

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        huber_sum = 0.0
        pinball_sum = 0.0
        crossing_sum = 0.0
        width_sum = 0.0

        for batch_x, batch_y in train_loader:
            if len(batch_x) < 2:
                continue
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)

            y_mean, q05, q50, q95 = model(batch_x)
            loss, m_dict = cqr_composite_loss(
                batch_y, y_mean, q05, q50, q95,
                lambda_pinball, lambda_huber, lambda_crossing, lambda_width,
                huber_delta=huber_delta,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()

            n_b = len(batch_x)
            train_loss_sum += loss.item() * n_b
            huber_sum += m_dict["loss_huber"] * n_b
            pinball_sum += m_dict["loss_pinball"] * n_b
            crossing_sum += m_dict["loss_crossing"] * n_b
            width_sum += m_dict["loss_width"] * n_b

        scheduler.step()
        n_tr = max(1, len(train_ds))

        if has_val and t_X_va is not None:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_mean, val_q05, val_q50, val_q95 = model(t_X_va)
                _, val_m_dict = cqr_composite_loss(
                    t_y_va, val_mean, val_q05, val_q50, val_q95,
                    lambda_pinball, lambda_huber, lambda_crossing, lambda_width,
                    huber_delta=huber_delta,
                )

            val_pinball = val_m_dict["loss_pinball"]
            val_huber = val_m_dict["loss_huber"]

            # F0a: the network works in standardised units when the flag is on;
            # every reported validation metric is mapped back to t/ha so that
            # early stopping and the logs stay comparable across flag settings.
            val_p_np = val_mean.squeeze().cpu().numpy() * target_sd + target_mu
            val_q05_np = val_q05.squeeze().cpu().numpy() * target_sd + target_mu
            val_q50_np = val_q50.squeeze().cpu().numpy() * target_sd + target_mu
            val_q95_np = val_q95.squeeze().cpu().numpy() * target_sd + target_mu

            val_rmse_val = float(np.sqrt(np.mean((y_val_np - val_p_np) ** 2)))
            val_mae_val = float(np.mean(np.abs(y_val_np - val_p_np)))
            ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
            ss_res = np.sum((y_val_np - val_p_np) ** 2)
            val_r2_val = float(1.0 - ss_res / max(1e-6, ss_tot))

            cross_rate = float(np.mean((val_q05_np > val_q50_np) | (val_q50_np > val_q95_np)))
            mean_width_val = float(np.mean(val_q95_np - val_q05_np))

            if init_pinball is None: init_pinball = max(1e-4, val_pinball)
            if init_rmse is None: init_rmse = max(1e-4, val_rmse_val)

            # Determine early stopping metric
            if early_stopping_mode == "rmse":
                current_stop_metric = val_rmse_val
            elif early_stopping_mode == "balanced":
                current_stop_metric = (val_pinball / init_pinball) + (val_rmse_val / init_rmse)
            else:  # default "pinball"
                current_stop_metric = val_pinball

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss_sum / n_tr, 4),
                "train_huber": round(huber_sum / n_tr, 4),
                "train_pinball": round(pinball_sum / n_tr, 4),
                "train_crossing": round(crossing_sum / n_tr, 4),
                "train_width": round(width_sum / n_tr, 4),
                "val_pinball": round(val_pinball, 4),
                "val_huber": round(val_huber, 4),
                "val_rmse": round(val_rmse_val, 4),
                "val_mae": round(val_mae_val, 4),
                "val_r2": round(val_r2_val, 4),
                "val_crossing_rate": round(cross_rate, 4),
                "val_mean_width": round(mean_width_val, 4),
                "stop_metric": round(current_stop_metric, 4),
            })

            if current_stop_metric < best_stop_metric:
                best_stop_metric = current_stop_metric
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs:
                logger.info("  Epoch %2d/%2d -> Train Loss: %.4f, Val RMSE: %.4f, Val Pinball: %.4f, Stop Metric: %.4f",
                            epoch, epochs, train_loss_sum / n_tr, val_rmse_val, val_pinball, current_stop_metric)

            if patience_counter >= patience:
                logger.info("  Early stopping triggered at epoch %d (mode: %s, best stop metric: %.4f at epoch %d)",
                            epoch, early_stopping_mode, best_stop_metric, best_epoch)
                break
        else:
            # Final refit mode: No validation set passed, no early stopping.
            # F0b: ema.state_dict() returns the averaged weights WITHOUT
            # mutating the live model, so training continues from the raw
            # weights on the next epoch. With USE_EMA_WEIGHTS off this is the
            # legacy behaviour (the last epoch's weights).
            _src = ema.state_dict() if bool(getattr(cfg, "USE_EMA_WEIGHTS", False)) else model.state_dict()
            best_weights = {k: v.cpu().clone() for k, v in _src.items()}
            best_epoch = epoch
            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss_sum / n_tr, 4),
                "train_huber": round(huber_sum / n_tr, 4),
                "train_pinball": round(pinball_sum / n_tr, 4),
                "train_crossing": round(crossing_sum / n_tr, 4),
                "train_width": round(width_sum / n_tr, 4),
                "stop_metric": round(train_loss_sum / n_tr, 4),
            })
            if epoch % 10 == 0 or epoch == epochs:
                logger.info("  Epoch %2d/%2d -> Train Loss: %.4f (final refit 1985-2015, zero validation leakage)",
                            epoch, epochs, train_loss_sum / n_tr)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval().to("cpu")
    model.best_epoch = best_epoch

    logger.info("  NeuralCQR fit complete: total_epochs=%d, best_dev_epoch=%d", len(history), best_epoch)

    result = QuantileModelSet(
        point_model=model,
        lower_model=model,
        upper_model=model,
        median_model=model,
        model_type="NeuralCQR",
        feature_cols=feature_cols,
        is_neural=True,
        scaler=scaler,
        epochs_trained=len(history),
        best_epoch=best_epoch,
        training_history=history,
        target_mu=target_mu,
        target_sd=target_sd,
    )
    result.training_paradigm = "joint_end_to_end"
    result.paradigm_fingerprint = {
        "n_optimization_stages": 1,
        "backbone_receives_quantile_gradient": True,
        "stage1_loss_terms": ["huber", "pinball_05", "pinball_50", "pinball_95",
                              "crossing", "width"],
        "stage2_loss_terms": None,
        "frozen_backbone_for_quantiles": False,
        "lambda_pinball": lambda_pinball,
        "lambda_huber": lambda_huber,
        "lambda_crossing": lambda_crossing,
        "lambda_width": lambda_width,
        "huber_delta": huber_delta,
        "standardized_target": _standardize,
        "monotone_heads": bool(getattr(cfg, "NEURAL_MONOTONE_HEADS", False)),
        "ema_weights_used": bool(getattr(cfg, "USE_EMA_WEIGHTS", False)) and not has_val,
        "schedule_t_max": _t_max,
    }
    return result


def _train_neural_cqr_posthoc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    feature_cols: Optional[List[str]],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout_rate: float,
    hidden_dims: Tuple[int, ...],
    scaler: Any,
    lambda_pinball: float,
    lambda_huber: float,
    lambda_crossing: float,
    early_stopping_mode: str,
    patience: int,
    early_stopping: bool,
    seed: int,
) -> QuantileModelSet:
    """Condition A: post-hoc CQR (point model first, interval heads afterwards).

    See :func:`train_neural_cqr` for the contrast with joint training. This
    function is a separate code path on purpose -- it must never be reachable
    from the joint branch, so that the ablation compares two real methodologies.
    """
    set_global_seed(seed)
    has_val = (X_val is not None and y_val is not None and early_stopping)
    logger.info(
        "Training PyTorch Neural CQR Net (Post-Hoc mode, 2 stages, §4.2, seed=%d, es_mode=%s, early_stopping=%s)",
        seed, early_stopping_mode if has_val else "none", has_val,
    )

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCQRNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)

    t_X_tr = torch.tensor(X_train, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    if has_val:
        t_X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
        t_y_va = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_np = np.asarray(y_val).flatten()
    else:
        t_X_va, t_y_va, y_val_np = None, None, None

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(t_X_tr, t_y_tr), batch_size=batch_size,
        shuffle=True, generator=g, drop_last=False,
    )

    history: List[Dict[str, float]] = []

    # ── STAGE 1: point model only (Huber). Quantile heads get NO gradient. ──
    point_params = (
        list(model.in_proj.parameters()) + list(model.in_bn.parameters())
        + list(model.backbone.parameters()) + list(model.mean_head.parameters())
    )
    for head in (model.q05_head, model.q50_head, model.q95_head):
        for prm in head.parameters():
            prm.requires_grad_(False)

    optimizer = optim.AdamW(point_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_stop_metric = float("inf")
    best_weights = None
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            if len(batch_x) < 2:
                continue
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_mean, _q05, _q50, _q95 = model(batch_x)
            loss = lambda_huber * nn.functional.huber_loss(y_mean, batch_y, delta=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(point_params, max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_x)
        scheduler.step()
        n_tr = max(1, len(t_X_tr))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_mean, _, _, _ = model(t_X_va)
            val_p_np = val_mean.squeeze().cpu().numpy()
            val_rmse_val = float(np.sqrt(np.mean((y_val_np - val_p_np) ** 2)))
            val_mae_val = float(np.mean(np.abs(y_val_np - val_p_np)))
            ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
            val_r2_val = float(1.0 - np.sum((y_val_np - val_p_np) ** 2) / max(1e-6, ss_tot))
            # Stage 1 has no interval objective, so the point metric is the only
            # legitimate stopping criterion regardless of early_stopping_mode.
            current_stop_metric = val_rmse_val

            history.append({
                "stage": 1, "epoch": epoch,
                "train_loss": round(train_loss_sum / n_tr, 4),
                "train_huber": round(train_loss_sum / n_tr, 4),
                "val_rmse": round(val_rmse_val, 4),
                "val_mae": round(val_mae_val, 4),
                "val_r2": round(val_r2_val, 4),
                "stop_metric": round(current_stop_metric, 4),
            })

            if current_stop_metric < best_stop_metric:
                best_stop_metric = current_stop_metric
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs:
                logger.info("  [stage 1/point] Epoch %2d/%2d -> Train Huber: %.4f, Val RMSE: %.4f",
                            epoch, epochs, train_loss_sum / n_tr, val_rmse_val)
            if patience_counter >= patience:
                logger.info("  [stage 1/point] Early stopping at epoch %d (best val RMSE %.4f at epoch %d)",
                            epoch, best_stop_metric, best_epoch)
                break
        else:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            history.append({
                "stage": 1, "epoch": epoch,
                "train_loss": round(train_loss_sum / n_tr, 4),
                "stop_metric": round(train_loss_sum / n_tr, 4),
            })

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    stage1_epochs = len(history)

    # ── STAGE 2: freeze the point model, fit quantile heads on fixed features ──
    for prm in point_params:
        prm.requires_grad_(False)
    q_params = []
    for head in (model.q05_head, model.q50_head, model.q95_head):
        for prm in head.parameters():
            prm.requires_grad_(True)
            q_params.append(prm)

    q_optimizer = optim.AdamW(q_params, lr=lr, weight_decay=weight_decay)
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(q_optimizer, T_max=epochs, eta_min=1e-5)

    q_best_metric = float("inf")
    q_best_weights = None
    q_best_epoch = 1
    q_patience_counter = 0

    for epoch in range(1, epochs + 1):
        # eval() everywhere: the frozen trunk must not update BatchNorm running
        # statistics, otherwise the "fixed representation" would still drift.
        model.eval()
        q_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            if len(batch_x) < 2:
                continue
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            q_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                h = model.in_act(model.in_bn(model.in_proj(batch_x)))
                h = model.backbone(h)
            q05, q50, q95 = model.q05_head(h), model.q50_head(h), model.q95_head(h)
            pin = (pinball_loss(batch_y, q05, 0.05)
                   + pinball_loss(batch_y, q50, 0.50)
                   + pinball_loss(batch_y, q95, 0.95))
            crossing = torch.mean(torch.relu(q05 - q50)) + torch.mean(torch.relu(q50 - q95))
            loss = lambda_pinball * pin + lambda_crossing * crossing
            loss.backward()
            nn.utils.clip_grad_norm_(q_params, max_norm=1.0)
            q_optimizer.step()
            q_loss_sum += loss.item() * len(batch_x)
        q_scheduler.step()
        n_tr = max(1, len(t_X_tr))

        if has_val:
            with torch.no_grad():
                _, v05, v50, v95 = model(t_X_va)
                v_pin = float(
                    pinball_loss(t_y_va, v05, 0.05).item()
                    + pinball_loss(t_y_va, v50, 0.50).item()
                    + pinball_loss(t_y_va, v95, 0.95).item()
                )
            history.append({
                "stage": 2, "epoch": epoch,
                "train_loss": round(q_loss_sum / n_tr, 4),
                "train_pinball": round(q_loss_sum / n_tr, 4),
                "val_pinball": round(v_pin, 4),
                "stop_metric": round(v_pin, 4),
            })
            if v_pin < q_best_metric:
                q_best_metric = v_pin
                q_best_epoch = epoch
                q_best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                    if k.startswith(("q05_head", "q50_head", "q95_head"))
                }
                q_patience_counter = 0
            else:
                q_patience_counter += 1
            if epoch % 10 == 0 or epoch == epochs:
                logger.info("  [stage 2/quantile] Epoch %2d/%2d -> Train Pinball: %.4f, Val Pinball: %.4f",
                            epoch, epochs, q_loss_sum / n_tr, v_pin)
            if q_patience_counter >= patience:
                logger.info("  [stage 2/quantile] Early stopping at epoch %d (best val pinball %.4f at epoch %d)",
                            epoch, q_best_metric, q_best_epoch)
                break
        else:
            history.append({
                "stage": 2, "epoch": epoch,
                "train_loss": round(q_loss_sum / n_tr, 4),
                "stop_metric": round(q_loss_sum / n_tr, 4),
            })

    if q_best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in q_best_weights.items()}, strict=False)

    for prm in model.parameters():
        prm.requires_grad_(True)
    model.eval().to("cpu")
    model.best_epoch = best_epoch

    logger.info(
        "  NeuralCQR post-hoc fit complete: stage1_epochs=%d (best %d), stage2_epochs=%d (best %d)",
        stage1_epochs, best_epoch, len(history) - stage1_epochs, q_best_epoch,
    )

    result = QuantileModelSet(
        point_model=model,
        lower_model=model,
        upper_model=model,
        median_model=model,
        model_type="NeuralCQR",
        feature_cols=feature_cols,
        is_neural=True,
        scaler=scaler,
        epochs_trained=len(history),
        best_epoch=best_epoch,
        training_history=history,
    )
    result.training_paradigm = "post_hoc_two_stage"
    result.paradigm_fingerprint = {
        "n_optimization_stages": 2,
        "backbone_receives_quantile_gradient": False,
        "stage1_loss_terms": ["huber"],
        "stage2_loss_terms": ["pinball_05", "pinball_50", "pinball_95", "crossing"],
        "frozen_backbone_for_quantiles": True,
        "stage1_epochs": stage1_epochs,
        "stage1_best_epoch": best_epoch,
        "stage2_epochs": len(history) - stage1_epochs,
        "stage2_best_epoch": q_best_epoch,
        "lambda_pinball": lambda_pinball,
        "lambda_huber": lambda_huber,
        "lambda_crossing": lambda_crossing,
        "lambda_width": 0.0,
    }
    return result


def train_neural_cqr_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: List[str],
    epochs: int,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    weight_decay: float = cfg.WEIGHT_DECAY,
    dropout_rate: float = cfg.NEURAL_CQR_DROPOUT,
    hidden_dims: Tuple[int, ...] = cfg.NEURAL_CQR_HIDDEN_DIMS,
    joint_training: bool = True,
    scaler: Any = None,
    lambda_pinball: float = cfg.LAMBDA_PINBALL,
    lambda_huber: float = cfg.LAMBDA_HUBER,
    lambda_crossing: float = cfg.LAMBDA_CROSSING,
    lambda_width: float = cfg.LAMBDA_WIDTH,
    seed: int = cfg.RANDOM_SEED,
    schedule_t_max: Optional[int] = None,
) -> QuantileModelSet:
    """Train PyTorch Neural CQR model for the final refit on 1985-2015.

    Zero-leakage final refit:
    - Trains strictly on 1985-2015 without early stopping.
    - No validation set passed (X_dev is part of training, not validation).
    - Trains for exactly `epochs` (the frozen epoch count selected during DEV).
    """
    return train_neural_cqr(
        X_train=X_train,
        y_train=y_train,
        X_val=None,
        y_val=None,
        feature_cols=feature_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        hidden_dims=hidden_dims,
        joint_training=joint_training,
        scaler=scaler,
        lambda_pinball=lambda_pinball,
        lambda_huber=lambda_huber,
        lambda_crossing=lambda_crossing,
        lambda_width=lambda_width,
        early_stopping=False,
        seed=seed,
        # F0c: without this the refit compresses the whole cosine schedule into
        # `epochs`, so the frozen DEV epoch count is reached at a different
        # learning rate than during the DEV run that chose it.
        schedule_t_max=(
            schedule_t_max if schedule_t_max is not None
            else (cfg.BASELINE_MAX_EPOCHS if bool(getattr(cfg, "LR_SCHEDULE_MATCH_DEV", False)) else None)
        ),
    )


def predict_intervals(
    model_set: QuantileModelSet,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate mean point predictions and quantile bounds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (point_preds, q_lo, q_hi) where point_preds comes from Head 1 (y_mean),
        q_lo from Head 2 (q0.05), and q_hi from Head 4 (q0.95).
    """
    if model_set.is_neural:
        model = model_set.point_model
        model.eval()
        # Follow the model's own device rather than assuming CPU. Training and
        # the computational benchmark may leave the module on CUDA, and feeding
        # it a CPU tensor would raise a device-mismatch error.
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        t_X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_mean, q05, q50, q95 = model(t_X)
            preds = y_mean.squeeze().detach().cpu().numpy()
            q_lo = q05.squeeze().detach().cpu().numpy()
            q_hi = q95.squeeze().detach().cpu().numpy()

        if preds.ndim == 0: preds = np.array([preds.item()])
        if q_lo.ndim == 0: q_lo = np.array([q_lo.item()])
        if q_hi.ndim == 0: q_hi = np.array([q_hi.item()])

        # F0a: undo the training-time target standardisation. Defaults
        # (mu=0, sd=1) leave the legacy path bit-identical.
        _mu = float(getattr(model_set, "target_mu", 0.0) or 0.0)
        _sd = float(getattr(model_set, "target_sd", 1.0) or 1.0)
        if _sd != 1.0 or _mu != 0.0:
            preds = preds * _sd + _mu
            q_lo = q_lo * _sd + _mu
            q_hi = q_hi * _sd + _mu
        return preds, q_lo, q_hi
    else:
        # Non-neural tree baselines
        if isinstance(X, np.ndarray) and model_set.feature_cols:
            X_eval = pd.DataFrame(X, columns=model_set.feature_cols)
        else:
            X_eval = X
        preds = model_set.point_model.predict(X_eval)
        q_lo = model_set.lower_model.predict(X_eval)
        q_hi = model_set.upper_model.predict(X_eval)
        return preds, q_lo, q_hi


# ─────────────────────────────────────────────────────────────
# Tree & Baseline Models (LightGBM, CatBoost, XGBoost, Ridge)
# ─────────────────────────────────────────────────────────────

def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train Ridge, Random Forest, LightGBM, CatBoost, and XGBoost baselines."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    from evaluation import rmse, mae, r_squared

    results = {}

    # 1. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    p_ridge = ridge.predict(X_test)
    results["Ridge"] = {"rmse": rmse(y_test, p_ridge), "mae": mae(y_test, p_ridge), "r2": r_squared(y_test, p_ridge)}

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=cfg.RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    results["RandomForest"] = {"rmse": rmse(y_test, p_rf), "mae": mae(y_test, p_rf), "r2": r_squared(y_test, p_rf)}

    # 3. LightGBM
    lgb_m = lgb.LGBMRegressor(**cfg.LGBM_PARAMS)
    lgb_m.fit(X_train, y_train)
    p_lgb = lgb_m.predict(X_test)
    results["LightGBM"] = {"rmse": rmse(y_test, p_lgb), "mae": mae(y_test, p_lgb), "r2": r_squared(y_test, p_lgb)}

    return results


def train_lgbm_quantile(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    feature_cols: Optional[List[str]] = None,
) -> QuantileModelSet:
    """Train LightGBM Quantile Regressors with frozen hyperparameters.

    ``X_val``/``y_val`` were accepted and silently ignored: every call built the
    full ``cfg.LGBM_PARAMS["n_estimators"]`` trees regardless, while callers --
    `_run_loso_cv` among them -- passed a validation set and could reasonably
    believe it early-stopped. The set is now used only when
    ``cfg.LGBM_DEV_EARLY_STOPPING`` is on (default off, so the locked results are
    unchanged), and when it is off the unused argument is logged rather than
    swallowed.
    """
    import lightgbm as lgb

    use_es = bool(getattr(cfg, "LGBM_DEV_EARLY_STOPPING", False)) and X_val is not None and y_val is not None
    params = dict(cfg.LGBM_PARAMS)
    fit_kw: Dict[str, Any] = {}
    if use_es:
        fit_kw = dict(eval_set=[(X_val, y_val)], eval_metric="l2",
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    elif X_val is not None:
        logger.debug("train_lgbm_quantile: validation set supplied but unused "
                     "(cfg.LGBM_DEV_EARLY_STOPPING is off); fitting %s trees",
                     params.get("n_estimators"))

    p_model = lgb.LGBMRegressor(**params).fit(X_train, y_train, **fit_kw)
    if use_es:
        logger.info("LightGBM early stopping on the supplied validation set: %s trees kept",
                    getattr(p_model, "best_iteration_", params.get("n_estimators")))
    lo_model = lgb.LGBMRegressor(objective="quantile", alpha=0.05, n_estimators=500, random_state=cfg.RANDOM_SEED, verbose=-1).fit(X_train, y_train)
    hi_model = lgb.LGBMRegressor(objective="quantile", alpha=0.95, n_estimators=500, random_state=cfg.RANDOM_SEED, verbose=-1).fit(X_train, y_train)

    return QuantileModelSet(
        point_model=p_model, lower_model=lo_model, upper_model=hi_model,
        model_type="LightGBM", feature_cols=feature_cols, is_neural=False
    )


def train_catboost_quantile(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    feature_cols: Optional[List[str]] = None,
) -> QuantileModelSet:
    """Train CatBoost Quantile Regressors with frozen hyperparameters."""
    from catboost import CatBoostRegressor
    p_model = CatBoostRegressor(iterations=500, learning_rate=0.05, verbose=0, random_seed=cfg.RANDOM_SEED).fit(X_train, y_train)
    lo_model = CatBoostRegressor(loss_function="Quantile:alpha=0.05", iterations=500, learning_rate=0.05, verbose=0, random_seed=cfg.RANDOM_SEED).fit(X_train, y_train)
    hi_model = CatBoostRegressor(loss_function="Quantile:alpha=0.95", iterations=500, learning_rate=0.05, verbose=0, random_seed=cfg.RANDOM_SEED).fit(X_train, y_train)

    return QuantileModelSet(
        point_model=p_model, lower_model=lo_model, upper_model=hi_model,
        model_type="CatBoost", feature_cols=feature_cols, is_neural=False
    )


def train_xgb_quantile(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    feature_cols: Optional[List[str]] = None,
) -> QuantileModelSet:
    """Train XGBoost Quantile Regressors with frozen hyperparameters."""
    import xgboost as xgb
    p_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)
    lo_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.05, n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)
    hi_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.95, n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)

    return QuantileModelSet(
        point_model=p_model, lower_model=lo_model, upper_model=hi_model,
        model_type="XGBoost", feature_cols=feature_cols, is_neural=False
    )


def save_model_artifacts(model_set: QuantileModelSet, model_name: str, scaler: Any = None, hyperparams: dict = None) -> None:
    """Save model metadata, hyperparameters and the fitted weights.

    C8: only the metadata JSON was written, so `outputs_*/models/` came out empty
    and no run could be reproduced from its artefacts. Weights now go to
    cfg.MODELS_DIR alongside the scaler.
    """
    from utils import save_report
    info = {
        "model_name": model_name,
        "is_neural": model_set.is_neural,
        "feature_count": len(model_set.feature_cols) if model_set.feature_cols else 0,
        "feature_cols": list(model_set.feature_cols or []),
        "hyperparams": hyperparams or {},
        "target_mu": float(getattr(model_set, "target_mu", 0.0) or 0.0),
        "target_sd": float(getattr(model_set, "target_sd", 1.0) or 1.0),
        "training_paradigm": getattr(model_set, "training_paradigm", None),
        "paradigm_fingerprint": getattr(model_set, "paradigm_fingerprint", None),
        "epochs_trained": getattr(model_set, "epochs_trained", None),
        "best_epoch": getattr(model_set, "best_epoch", None),
    }

    saved: List[str] = []
    try:
        models_dir = Path(cfg.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        if model_set.is_neural:
            path = models_dir / f"model_{model_name}_state_dict.pt"
            torch.save(model_set.point_model.state_dict(), path)
            saved.append(path.name)
        else:
            import joblib
            for role, mdl in (("point", model_set.point_model),
                              ("lower", model_set.lower_model),
                              ("upper", model_set.upper_model)):
                if mdl is None:
                    continue
                path = models_dir / f"model_{model_name}_{role}.joblib"
                joblib.dump(mdl, path)
                saved.append(path.name)

        if scaler is not None:
            import joblib
            path = models_dir / f"scaler_{model_name}.joblib"
            joblib.dump(scaler, path)
            saved.append(path.name)
    except Exception as exc:  # never fail a run over artefact saving
        logger.warning("Could not save weights for %s: %s", model_name, exc)
        info["weight_save_error"] = str(exc)

    info["saved_artifacts"] = saved
    logger.info("Saved %d weight artefact(s) for %s -> %s", len(saved), model_name, cfg.MODELS_DIR)
    save_report(info, f"model_{model_name}_metadata.json")


def save_model_failure_log(failures: list) -> None:
    """Log backbone training failures."""
    from utils import save_report
    save_report({"failures": failures}, "model_failures.json")


def save_hyperparameter_report(report: dict) -> None:
    """Save hyperparameter optimization report."""
    from utils import save_report
    save_report(report, "hyperparameter_report.json")
