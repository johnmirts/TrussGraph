"""Visualization helpers for truss learning notebooks."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def plot_training_history(history, train_key=None, val_key=None, log_y=False):
    train_key = train_key or ("train" if "train" in history else "train_loss")
    val_key = val_key or ("val" if "val" in history else "validation_loss")
    epochs = np.arange(1, len(history[train_key]) + 1)
    validation = np.asarray(history[val_key], dtype=float)
    validated = np.isfinite(validation)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, history[train_key], label="Train")
    axis.plot(epochs[validated], validation[validated], marker="o", label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Graph-balanced Smooth L1 loss")
    axis.set_title("Training history")
    if log_y:
        axis.set_yscale("log")
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure


def plot_axial_force_parity(predictions, r2, seed=42, max_points=40_000):
    rng = np.random.default_rng(seed)
    num_edges = len(predictions["axial_true"])
    shown = (
        rng.choice(num_edges, size=max_points, replace=False)
        if num_edges > max_points
        else np.arange(num_edges)
    )
    true_force = predictions["axial_true"][shown]
    pred_force = predictions["axial_pred"][shown]
    limit = max(float(np.max(np.abs(true_force))), float(np.max(np.abs(pred_force))), 1e-8)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_force, pred_force, s=8, alpha=0.35)
    axis.plot([-limit, limit], [-limit, limit], linestyle="--", linewidth=1.5)
    axis.set_xlabel("True axial force")
    axis.set_ylabel("Predicted axial force")
    axis.set_title(f"Member axial-force parity, test set (R2={r2:.3f})")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    figure.tight_layout()
    plt.show()
    return figure


def plot_axial_force_error_distribution(predictions, bins=80):
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
    true_static = predictions["static_true"]
    pred_static = predictions["static_pred"]
    low = float(min(true_static.min(), pred_static.min()))
    high = float(max(true_static.max(), pred_static.max()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_static, pred_static, s=18, alpha=0.55)
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    axis.set_xlabel("True static action")
    axis.set_ylabel("Static action from predicted axial forces")
    axis.set_title(f"Static-action parity, test set (R2={r2:.3f})")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    figure.tight_layout()
    plt.show()
    return figure


def plot_static_action_relative_error(predictions, bins=60):
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
    model.eval()
    graph_device = graph.to(device)
    with torch.no_grad():
        output = model(graph_device)
    return output["axial_force"].detach().cpu().numpy()


def _add_member_field(axis, graph, values, title, norm, cmap):
    positions = graph.pos_raw.cpu().numpy()
    member_index = graph.member_index.cpu().numpy()
    segments = [positions[member_index[:, idx]] for idx in range(member_index.shape[1])]
    collection = LineCollection(segments, linewidths=3, cmap=cmap, norm=norm)
    collection.set_array(np.asarray(values))
    axis.add_collection(collection)
    axis.scatter(positions[:, 0], positions[:, 1], s=18, color="black")
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_title(title)
    return collection


def plot_member_force_comparison(graph, true_values, predicted_values, cmap="bwr_r"):
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    shared_bound = max(
        float(np.max(np.abs(true_values))),
        float(np.max(np.abs(predicted_values))),
        1e-8,
    )
    norm = TwoSlopeNorm(vmin=-shared_bound, vcenter=0.0, vmax=shared_bound)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    _add_member_field(axes[0], graph, true_values, "True member axial forces", norm, cmap)
    collection = _add_member_field(
        axes[1], graph, predicted_values, "Predicted member axial forces", norm, cmap
    )
    figure.colorbar(collection, ax=axes, label="Axial force", shrink=0.9)
    plt.show()
    return figure


def plot_displacement_parity(true_nodes, predicted_nodes, plane):
    true_magnitude = np.linalg.norm(true_nodes, axis=1)
    predicted_magnitude = np.linalg.norm(predicted_nodes, axis=1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    parity_items = [
        (true_nodes[:, 0], predicted_nodes[:, 0], f"U{plane[0]}"),
        (true_nodes[:, 1], predicted_nodes[:, 1], f"U{plane[1]}"),
        (true_magnitude, predicted_magnitude, "Magnitude"),
    ]
    for axis, (truth, prediction_values, label) in zip(axes, parity_items):
        axis.scatter(truth, prediction_values, s=8, alpha=0.35)
        lower = min(truth.min(), prediction_values.min())
        upper = max(truth.max(), prediction_values.max())
        axis.plot([lower, upper], [lower, upper], linestyle="--")
        axis.set_title(f"{label}: predicted vs true")
        axis.set_xlabel("True displacement")
        axis.set_ylabel("Predicted displacement")
    figure.tight_layout()
    plt.show()
    return figure


def plot_displacement_residuals(true_nodes, predicted_nodes, plane, bins=60):
    residuals = predicted_nodes - true_nodes
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for component, axis in enumerate(axes):
        axis.hist(residuals[:, component], bins=bins)
        axis.axvline(0.0, linestyle="--")
        axis.set_title(f"Residual distribution for U{plane[component]}")
        axis.set_xlabel("Predicted - true displacement")
        axis.set_ylabel("Free nodes")
    figure.tight_layout()
    plt.show()
    return figure


def plot_graph_relative_l2(results, bins=50):
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(results["graph_relative_l2"], bins=bins)
    axis.set_title("Distribution of graph relative L2 error")
    axis.set_xlabel("Relative L2 error")
    axis.set_ylabel("Test trusses")
    figure.tight_layout()
    plt.show()
    return figure


def plot_panel_performance(panel_performance):
    figure, axis = plt.subplots(figsize=(8, 4))
    panel_performance["median"].plot(kind="bar", ax=axis)
    axis.set_title("Median graph relative L2 error by panel count")
    axis.set_xlabel("Number of panels")
    axis.set_ylabel("Median relative L2 error")
    figure.tight_layout()
    plt.show()
    return figure


def plot_deformed_truss(graph, prediction, plane="xz", title="Test example"):
    position = graph.pos_raw.cpu().numpy()
    truth = graph.displacement.cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    physical_edges = graph.member_index.cpu().numpy().T
    span = np.ptp(position, axis=0).max()
    max_displacement = max(
        np.linalg.norm(truth, axis=1).max(),
        np.linalg.norm(prediction, axis=1).max(),
        1e-12,
    )
    amplification = 0.12 * span / max_displacement
    true_deformed = position + amplification * truth
    predicted_deformed = position + amplification * prediction

    figure, axis = plt.subplots(figsize=(11, 5))
    for source, target in physical_edges:
        axis.plot(
            position[[source, target], 0],
            position[[source, target], 1],
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
        )
        axis.plot(
            true_deformed[[source, target], 0],
            true_deformed[[source, target], 1],
            linewidth=1.6,
        )
        axis.plot(
            predicted_deformed[[source, target], 0],
            predicted_deformed[[source, target], 1],
            linestyle=":",
            linewidth=1.8,
        )

    support = ~graph.free_mask.cpu().numpy()
    axis.scatter(position[support, 0], position[support, 1], marker="s", s=55, label="Supports")
    axis.plot([], [], linestyle="--", label="Undeformed")
    axis.plot([], [], linewidth=1.6, label="True deformation")
    axis.plot([], [], linestyle=":", linewidth=1.8, label="Predicted deformation")
    axis.set_aspect("equal")
    axis.set_title(f"{title} - deformation amplification x{amplification:.2g}")
    axis.set_xlabel(plane[0].upper())
    axis.set_ylabel(plane[1].upper())
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure
