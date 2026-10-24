"""Train PARD from raw scenario-action CSV; this module is never invoked implicitly.

Example (run on the server only):
    PYTHONPATH=src python -m radar.pard.train --csv data/simulated/...csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, PROJECT_ROOT

from .config import PARDConfig, SCENARIO, TARGET
from .features import RawFeatureSpec
from .losses import pinball_loss, scenario_pair_ranking, top_action_weight
from .model import PARDModel


class FrameDataset(Dataset):
    def __init__(self, x: Dict[str, np.ndarray], reward: np.ndarray, scenario: np.ndarray):
        self.x = x
        self.reward = reward.astype(np.float32)
        self.scenario = scenario.astype(np.int64)

    def __len__(self) -> int:
        return len(self.reward)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {**{k: torch.from_numpy(v[i]) for k, v in self.x.items()}, "reward": torch.tensor(self.reward[i]), "scenario": torch.tensor(self.scenario[i])}


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin-1")


def split_by_scenario(frame: pd.DataFrame, cfg: PARDConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The unit of splitting is a complete jammer scenario, never a CSV row."""
    if SCENARIO not in frame:
        raise ValueError(f"{SCENARIO} is required; row-wise train/validation splitting is forbidden for PARD.")
    scenario_ids = np.sort(frame[SCENARIO].unique())
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(scenario_ids)
    n_train = int(len(scenario_ids) * cfg.train_fraction)
    n_val = int(len(scenario_ids) * cfg.val_fraction)
    train_ids = set(scenario_ids[:n_train])
    val_ids = set(scenario_ids[n_train:n_train + n_val])
    return (frame[frame[SCENARIO].isin(train_ids)].copy(), frame[frame[SCENARIO].isin(val_ids)].copy(), frame[~frame[SCENARIO].isin(train_ids | val_ids)].copy())


def _loader(frame: pd.DataFrame, spec: RawFeatureSpec, batch_size: int, shuffle: bool) -> DataLoader:
    x = spec.transform(frame)
    ds = FrameDataset(x, frame[TARGET].to_numpy(np.float32), frame[SCENARIO].to_numpy(np.int64))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _adversarial_consistency(model: PARDModel, x: Dict[str, torch.Tensor], cfg: PARDConfig) -> torch.Tensor:
    """First-order inner adversary on state numeric tokens only.

    The radius lives in normalised feature units and is deliberately restricted
    to the eight jammer fields; radar actions and jammer categories never move.
    """
    numeric = x["numeric"]
    delta = torch.zeros_like(numeric, requires_grad=True)
    for _ in range(cfg.adversarial_steps):
        pred = model(numeric + delta, x["present"], x["categorical"], x["physics"])
        risk = model.risk_score(pred, cfg.risk_beta, cfg.cvar_weight).mean()
        grad = torch.autograd.grad(risk, delta, create_graph=False)[0]
        mask = torch.zeros_like(delta)
        mask[:, :8] = 1.0
        delta = (delta - cfg.adversarial_radius * grad.sign() * mask).detach().requires_grad_(True)
    perturbed = model(numeric + delta.detach(), x["present"], x["categorical"], x["physics"])
    nominal = model(numeric, x["present"], x["categorical"], x["physics"])
    # Penalise the adversarial lower-tail drop, not harmless optimistic changes.
    return F.relu(model.risk_score(nominal, cfg.risk_beta, cfg.cvar_weight) - model.risk_score(perturbed, cfg.risk_beta, cfg.cvar_weight)).mean()


def validate(model: PARDModel, loader: DataLoader, cfg: PARDConfig, device: torch.device) -> float:
    model.eval()
    errors = []
    median_idx = int(np.argmin(np.abs(np.asarray(cfg.quantiles) - 0.5)))
    with torch.no_grad():
        for batch in loader:
            b = _batch_to_device(batch, device)
            pred = model(b["numeric"], b["present"], b["categorical"], b["physics"])
            errors.append((pred[:, median_idx] - b["reward"]).abs().cpu())
    return float(torch.cat(errors).mean())


def train(csv_path: Path, output_dir: Path, cfg: PARDConfig) -> None:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    frame = _read_csv(csv_path)
    required = {SCENARIO, TARGET}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    train_frame, val_frame, test_frame = split_by_scenario(frame, cfg)
    spec = RawFeatureSpec.fit(train_frame)
    train_loader = _loader(train_frame, spec, cfg.batch_size, True)
    val_loader = _loader(val_frame, spec, cfg.batch_size, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_sizes = [len(spec.vocabularies[c]) for c in spec.vocabularies]
    model = PARDModel(vocab_sizes, cfg.quantiles, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout, cfg.fourier_bands).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    q = torch.tensor(cfg.quantiles, device=device).view(1, -1)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, stale = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            b = _batch_to_device(batch, device)
            pred = model(b["numeric"], b["present"], b["categorical"], b["physics"])
            risk = model.risk_score(pred, cfg.risk_beta, cfg.cvar_weight)
            weights = top_action_weight(b["reward"], b["scenario"])
            per_row = torch.maximum(q * (b["reward"].unsqueeze(1) - pred), (q - 1.0) * (b["reward"].unsqueeze(1) - pred)).mean(dim=1)
            q_loss = (per_row * weights).mean()
            rank_loss = scenario_pair_ranking(risk, b["reward"], b["scenario"])
            robust_loss = _adversarial_consistency(model, b, cfg)
            loss = q_loss + cfg.ranking_weight * rank_loss + cfg.robust_weight * robust_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_mae = validate(model, val_loader, cfg, device)
        print(f"epoch={epoch:03d} val_median_mae={val_mae:.4f}")
        if val_mae < best:
            best, stale = val_mae, 0
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "feature_spec": spec.to_dict(), "vocab_sizes": vocab_sizes, "best_val_median_mae": best}, output_dir / "pard_best.pt")
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    manifest = {"train_rows": len(train_frame), "val_rows": len(val_frame), "test_rows": len(test_frame), "train_scenarios": int(train_frame[SCENARIO].nunique()), "val_scenarios": int(val_frame[SCENARIO].nunique()), "test_scenarios": int(test_frame[SCENARIO].nunique()), "best_val_median_mae": best, "csv": str(csv_path), "config": cfg.to_dict()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone PARD algorithm.")
    parser.add_argument("--csv", type=Path, default=DATA_SIMULATED_HIGH_CONTRAST_CSV)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "pard")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = PARDConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    train(args.csv, args.output, cfg)


if __name__ == "__main__":
    main()
