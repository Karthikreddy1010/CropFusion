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
    training_history: List[Dict[str, float]] = None


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
    ) -> None:
        super().__init__()
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite Multi-Objective Loss Function (§3).

    L_total = lambda_pinball * [L_pinball(q05) + L_pinball(q50) + L_pinball(q95)]
            + lambda_huber * Huber(y_mean)
            + lambda_crossing * [ReLU(q05 - q50) + ReLU(q50 - q95)]
            + lambda_width * Width^2
    """
    huber_l = nn.functional.huber_loss(y_mean, y_true, delta=1.0)

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
    """Exponential Moving Average (EMA) of model weights for enhanced stability (vectorized)."""

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.clone().detach() for p in self.params]

    def update(self) -> None:
        d = self.decay
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                s.mul_(d).add_(p.data, alpha=1.0 - d)

    def apply_shadow(self, model: Optional[nn.Module] = None) -> None:
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.data.copy_(s)


# ─────────────────────────────────────────────────────────────
# PyTorch Training Loop with Epoch Diagnostics (§4)
# ─────────────────────────────────────────────────────────────

def train_neural_cqr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
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
    seed: int = cfg.RANDOM_SEED,
) -> QuantileModelSet:
    """Train PyTorch Neural CQR model with AdamW, CosineAnnealing, AMP, EMA & configurable early stopping."""
    set_global_seed(seed)
    mode_str = "Joint End-to-End" if joint_training else "Post-Hoc"
    logger.info("Training PyTorch Multi-Task Neural CQR Net (%s mode, §4.2, seed=%d, es_mode=%s)",
                mode_str, seed, early_stopping_mode)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCQRNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    ema = ModelEMA(model, decay=0.99)

    t_X_tr = torch.tensor(X_train, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    t_y_va = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Reproducible DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_stop_metric = float("inf")
    best_weights = None
    patience_counter = 0

    init_pinball: Optional[float] = None
    init_rmse: Optional[float] = None

    history: List[Dict[str, float]] = []
    y_val_np = y_val.flatten()

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
                lambda_pinball, lambda_huber, lambda_crossing, lambda_width
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

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_mean, val_q05, val_q50, val_q95 = model(t_X_va)
            _, val_m_dict = cqr_composite_loss(
                t_y_va, val_mean, val_q05, val_q50, val_q95,
                lambda_pinball, lambda_huber, lambda_crossing, lambda_width
            )

        val_pinball = val_m_dict["loss_pinball"]
        val_huber = val_m_dict["loss_huber"]

        val_p_np = val_mean.squeeze().cpu().numpy()
        val_q05_np = val_q05.squeeze().cpu().numpy()
        val_q50_np = val_q50.squeeze().cpu().numpy()
        val_q95_np = val_q95.squeeze().cpu().numpy()

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
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("  Epoch %2d/%2d -> Train Loss: %.4f, Val RMSE: %.4f, Val Pinball: %.4f, Stop Metric: %.4f",
                        epoch, epochs, train_loss_sum / n_tr, val_rmse_val, val_pinball, current_stop_metric)

        if patience_counter >= patience:
            logger.info("  Early stopping triggered at epoch %d (mode: %s, best stop metric: %.4f)",
                        epoch, early_stopping_mode, best_stop_metric)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval().to("cpu")

    return QuantileModelSet(
        point_model=model,
        lower_model=model,
        upper_model=model,
        median_model=model,
        model_type="NeuralCQR",
        feature_cols=feature_cols,
        is_neural=True,
        scaler=scaler,
        epochs_trained=len(history),
        training_history=history,
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
        t_X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_mean, q05, q50, q95 = model(t_X)
            preds = y_mean.squeeze().numpy()
            q_lo = q05.squeeze().numpy()
            q_hi = q95.squeeze().numpy()

        if preds.ndim == 0: preds = np.array([preds.item()])
        if q_lo.ndim == 0: q_lo = np.array([q_lo.item()])
        if q_hi.ndim == 0: q_hi = np.array([q_hi.item()])
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
    X_val: np.ndarray, y_val: np.ndarray,
    feature_cols: List[str],
) -> QuantileModelSet:
    """Train LightGBM Quantile Regressors."""
    import lightgbm as lgb
    p_model = lgb.LGBMRegressor(**cfg.LGBM_PARAMS).fit(X_train, y_train)
    lo_model = lgb.LGBMRegressor(objective="quantile", alpha=0.05, n_estimators=500, random_state=cfg.RANDOM_SEED, verbose=-1).fit(X_train, y_train)
    hi_model = lgb.LGBMRegressor(objective="quantile", alpha=0.95, n_estimators=500, random_state=cfg.RANDOM_SEED, verbose=-1).fit(X_train, y_train)

    return QuantileModelSet(
        point_model=p_model, lower_model=lo_model, upper_model=hi_model,
        model_type="LightGBM", feature_cols=feature_cols, is_neural=False
    )


def train_catboost_quantile(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_cols: List[str],
) -> QuantileModelSet:
    """Train CatBoost Quantile Regressors."""
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
    X_val: np.ndarray, y_val: np.ndarray,
    feature_cols: List[str],
) -> QuantileModelSet:
    """Train XGBoost Quantile Regressors."""
    import xgboost as xgb
    p_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)
    lo_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.05, n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)
    hi_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.95, n_estimators=300, learning_rate=0.05, random_state=cfg.RANDOM_SEED).fit(X_train, y_train)

    return QuantileModelSet(
        point_model=p_model, lower_model=lo_model, upper_model=hi_model,
        model_type="XGBoost", feature_cols=feature_cols, is_neural=False
    )


def save_model_artifacts(model_set: QuantileModelSet, model_name: str, scaler: Any = None, hyperparams: dict = None) -> None:
    """Save model metadata and hyperparameter logs."""
    from utils import save_report
    info = {
        "model_name": model_name,
        "is_neural": model_set.is_neural,
        "feature_count": len(model_set.feature_cols) if model_set.feature_cols else 0,
        "hyperparams": hyperparams or {},
    }
    save_report(info, f"model_{model_name}_metadata.json")


def save_model_failure_log(failures: list) -> None:
    """Log backbone training failures."""
    from utils import save_report
    save_report({"failures": failures}, "model_failures.json")


def save_hyperparameter_report(report: dict) -> None:
    """Save hyperparameter optimization report."""
    from utils import save_report
    save_report(report, "hyperparameter_report.json")
