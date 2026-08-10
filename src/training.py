"""Training, splitting, sampling, inference, and metric utilities."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_split(
    dataset,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[int(dataset[index].group_id.item())].append(index)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    n_groups = len(group_ids)
    n_train = max(1, int(round(train_fraction * n_groups)))
    n_validation = max(1, int(round(validation_fraction * n_groups)))
    if n_train + n_validation >= n_groups:
        n_train = max(1, n_groups - 2)
        n_validation = 1

    train_groups = set(group_ids[:n_train])
    validation_groups = set(group_ids[n_train : n_train + n_validation])
    test_groups = set(group_ids[n_train + n_validation :])
    return DatasetSplit(
        train=[idx for group in train_groups for idx in groups[group]],
        validation=[idx for group in validation_groups for idx in groups[group]],
        test=[idx for group in test_groups for idx in groups[group]],
    )

def random_split(
    dataset,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    num_graphs = len(dataset)
    if num_graphs < 3:
        raise ValueError("At least three graphs are required for train/validation/test splitting")

    indices = list(range(num_graphs))
    random.Random(seed).shuffle(indices)
    n_train = max(1, min(int(round(train_fraction * num_graphs)), num_graphs - 2))
    n_val = max(1, min(int(round(validation_fraction * num_graphs)), num_graphs - n_train - 1))

    return DatasetSplit(
        train=indices[:n_train], 
        validation=indices[n_train : n_train + n_val], 
        test=indices[n_train + n_val :],
    )

class EpochSubsetSampler(Sampler):
    """Draw a new subset without replacement every epoch."""

    def __init__(self, dataset_size, num_samples, seed: int = 42):
        self.dataset_size = int(dataset_size)
        self.num_samples = min(int(num_samples), self.dataset_size)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        indices = torch.randperm(self.dataset_size, generator=self.generator)
        return iter(indices[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def graph_balanced_axial_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    member_batch: Tensor,
    num_graphs: int,
    beta: float = 1.0,
) -> Tensor:
    per_member = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )
    graph_sum = torch.zeros(num_graphs, device=prediction.device, dtype=prediction.dtype)
    graph_sum.index_add_(0, member_batch, per_member)
    graph_count = torch.bincount(member_batch, minlength=num_graphs).clamp_min(1)
    return (graph_sum / graph_count).mean()


def run_axial_epoch(
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(batch, compute_static_action=False)
            loss = graph_balanced_axial_smooth_l1(
                output["axial_norm"],
                batch.y,
                output["member_batch"],
                batch.num_graphs,
            )
            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss) * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


def checkpoint_path_for_config(
    checkpoint_dir: str | Path,
    task_name: str,
    dataset_metadata: dict[str, Any],
    model_config: dict[str, Any],
    *,
    seed: int,
) -> Path:
    """Return a stable checkpoint path for a dataset/model/seed configuration."""
    signature = {
        "task_name": task_name,
        "dataset_config": dataset_metadata.get("config", {}),
        "model_config": model_config,
        "seed": int(seed),
    }
    digest = hashlib.sha1(
        json.dumps(signature, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()[:12]
    return Path(checkpoint_dir) / f"{task_name}_{digest}.pt"


def load_model_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_model_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    dataset_metadata: dict[str, Any],
    best_validation_loss: float,
    seed: int,
    history: dict[str, list[float]] | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "dataset_metadata": dataset_metadata,
            "best_validation_loss": best_validation_loss,
            "seed": int(seed),
            "history": history,
        },
        checkpoint_path,
    )


@torch.no_grad()
def collect_axial_predictions(model: nn.Module, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    axial_true = []
    axial_pred = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, compute_static_action=False)
        axial_true.append(batch.axial_force.detach().cpu())
        axial_pred.append(output["axial_force"].detach().cpu())
    return {
        "axial_true": torch.cat(axial_true).numpy(),
        "axial_pred": torch.cat(axial_pred).numpy(),
    }


@torch.no_grad()
def collect_static_predictions(model: nn.Module, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    static_true = []
    static_pred = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, compute_static_action=True)
        static_true.append(batch.static_action.view(-1).detach().cpu())
        static_pred.append(output["static_action"].detach().cpu())
    return {
        "static_true": torch.cat(static_true).numpy(),
        "static_pred": torch.cat(static_pred).numpy(),
    }


def graph_balanced_displacement_loss(
    prediction: Tensor,
    target: Tensor,
    node_batch: Tensor,
    free_mask: Tensor,
    num_graphs: int,
    beta: float = 0.1,
) -> Tensor:
    per_node = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    ).mean(dim=-1)
    active_error = per_node[free_mask]
    active_batch = node_batch[free_mask]
    graph_sum = torch.zeros(
        num_graphs, device=prediction.device, dtype=prediction.dtype
    )
    graph_count = torch.zeros_like(graph_sum)
    graph_sum.index_add_(0, active_batch, active_error)
    graph_count.index_add_(
        0, active_batch, torch.ones_like(active_error, dtype=prediction.dtype)
    )
    return (graph_sum / graph_count.clamp_min(1.0)).mean()


def train_displacement_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = graph_balanced_displacement_loss(
            prediction,
            batch.y,
            batch.batch,
            batch.free_mask,
            batch.num_graphs,
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate_displacement_loss(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch)
        loss = graph_balanced_displacement_loss(
            prediction,
            batch.y,
            batch.batch,
            batch.free_mask,
            batch.num_graphs,
        )
        total_loss += float(loss) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def collect_displacement_predictions(
    model: nn.Module, loader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    node_true: list[np.ndarray] = []
    node_pred: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        prediction_norm = model(batch)
        scale = batch.displacement_scale[batch.batch].unsqueeze(-1)
        prediction = prediction_norm * scale
        target = batch.y * scale

        free = batch.free_mask
        node_true.append(target[free].cpu().numpy())
        node_pred.append(prediction[free].cpu().numpy())

        for graph_index in range(batch.num_graphs):
            mask = (batch.batch == graph_index) & free
            truth_g = target[mask]
            prediction_g = prediction[mask]

    return {
        "node_true": np.concatenate(node_true, axis=0),
        "node_pred": np.concatenate(node_pred, axis=0),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
