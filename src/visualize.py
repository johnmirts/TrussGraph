from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def plot_training_history(history):
    """Plot training and validation loss and return the Matplotlib figure."""
    epochs = np.arange(1, len(history["train"]) + 1)
    validation = np.asarray(history["val"], dtype=float)
    validated = np.isfinite(validation)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, history["train"], label="Train")
    axis.plot(
        epochs[validated],
        validation[validated],
        marker="o",
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Graph-balanced Smooth L1 loss")
    axis.set_title("Training history")
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure


def plot_axial_force_parity(
    predictions,
    r2,
    seed=42,
    max_points=40_000,
):
    """Plot predicted versus true member axial forces."""
    rng = np.random.default_rng(seed)
    num_edges = len(predictions["axial_true"])

    if num_edges > max_points:
        shown = rng.choice(num_edges, size=max_points, replace=False)
    else:
        shown = np.arange(num_edges)

    true_force = predictions["axial_true"][shown]
    pred_force = predictions["axial_pred"][shown]
    limit = max(
        float(np.max(np.abs(true_force))),
        float(np.max(np.abs(pred_force))),
        1e-8,
    )

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_force, pred_force, s=8, alpha=0.35)
    axis.plot([-limit, limit], [-limit, limit], linestyle="--", linewidth=1.5)
    axis.set_xlabel("True axial force")
    axis.set_ylabel("Predicted axial force")
    axis.set_title(f"Member axial-force parity, test set (R²={r2:.3f})")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    figure.tight_layout()
    plt.show()
    return figure


def plot_axial_force_error_distribution(predictions, bins=80):
    """Plot the distribution of member axial-force residuals."""
    axial_error = predictions["axial_pred"] - predictions["axial_true"]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(axial_error, bins=bins)
    axis.set_xlabel("Predicted - true axial force")
    axis.set_ylabel("Member count")
    axis.set_title("Axial-force error distribution")
    figure.tight_layout()
    plt.show()
    return figure


def plot_static_action_parity(predictions, r2):
    """Plot static action computed from predicted forces against the reference."""
    true_static = predictions["static_true"]
    pred_static = predictions["static_pred"]
    low = float(min(true_static.min(), pred_static.min()))
    high = float(max(true_static.max(), pred_static.max()))

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_static, pred_static, s=18, alpha=0.55)
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    axis.set_xlabel("True static action")
    axis.set_ylabel("Static action from predicted axial forces")
    axis.set_title(f"Static-action parity, test set (R²={r2:.3f})")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    figure.tight_layout()
    plt.show()
    return figure


def plot_static_action_relative_error(predictions, bins=60):
    """Plot graph-level relative static-action error in percent."""
    relative_error = (
        (predictions["static_pred"] - predictions["static_true"])
        / np.maximum(np.abs(predictions["static_true"]), 1e-8)
    ) * 100.0

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(relative_error, bins=bins)
    axis.set_xlabel("Static-action relative error (%)")
    axis.set_ylabel("Structure count")
    axis.set_title("Static-action error distribution")
    figure.tight_layout()
    plt.show()
    return figure


def predict_single_graph(model, graph, device):
    """Predict physical member axial forces for one graph."""
    model.eval()
    graph_device = graph.to(device)
    with torch.no_grad():
        output = model(graph_device)
    return output["axial_force"].detach().cpu().numpy()


def _add_member_field(axis, graph, values, title, norm, cmap):
    positions = graph.pos_raw.cpu().numpy()
    member_index = graph.member_index.cpu().numpy()
    segments = [
        positions[member_index[:, idx]]
        for idx in range(member_index.shape[1])
    ]

    collection = LineCollection(
        segments,
        linewidths=3,
        cmap=cmap,
        norm=norm,
    )
    collection.set_array(np.asarray(values))

    axis.add_collection(collection)
    axis.scatter(positions[:, 0], positions[:, 1], s=18, color="black")
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_title(title)
    return collection


def plot_member_force_comparison(
    graph,
    true_values,
    predicted_values,
    cmap="bwr_r",
):
    """Plot true and predicted axial-force fields using one shared scale.

    Negative values are red and positive values are blue with the default
    reversed blue-white-red colormap.
    """
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)

    shared_bound = max(
        float(np.max(np.abs(true_values))),
        float(np.max(np.abs(predicted_values))),
        1e-8,
    )
    norm = TwoSlopeNorm(vmin=-shared_bound, vcenter=0.0, vmax=shared_bound)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    _add_member_field(
        axes[0], graph, true_values, "True member axial forces", norm, cmap
    )
    collection = _add_member_field(
        axes[1], graph, predicted_values, "Predicted member axial forces", norm, cmap
    )
    figure.colorbar(collection, ax=axes, label="Axial force", shrink=0.9)
    plt.show()
    return figure
