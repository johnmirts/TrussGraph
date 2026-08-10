"""Dataset filtering and scaling helpers for truss graph imports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


BASE_COORD_SLICE = slice(2, 4)
BASE_FORCE_SLICE = slice(4, 6)
BASE_DISPLACEMENT_SLICE = slice(6, 8)
MEMBER_LENGTH_COLUMN = 0
MEMBER_AXIAL_COLUMN = 1


@dataclass(frozen=True)
class MaterialScales:
    """Scale factors for geometry and linear-elastic response quantities."""

    length: float
    force: float
    displacement: float


def material_scales(
    *,
    span: float,
    load_per_length: float,
    young_modulus: float,
    area: float,
) -> MaterialScales:
    """Return result scale factors for a uniformly loaded linear truss family."""
    total_load = float(load_per_length) * float(span)
    axial_stiffness = float(young_modulus) * float(area)
    if axial_stiffness <= 0:
        raise ValueError("young_modulus * area must be positive")
    return MaterialScales(
        length=float(span),
        force=total_load,
        displacement=total_load * float(span) / axial_stiffness,
    )


def graph_coordinates(data: Data) -> torch.Tensor:
    """Return in-plane node coordinates from either base or task datasets."""
    if hasattr(data, "pos_raw"):
        return data.pos_raw
    return data.x[:, BASE_COORD_SLICE]


def graph_member_lengths(data: Data) -> torch.Tensor:
    """Return physical member lengths from either base or task datasets."""
    if hasattr(data, "member_length"):
        return data.member_length
    return data.member_attr[:, MEMBER_LENGTH_COLUMN]


def minimum_member_length(data: Data) -> float:
    return float(graph_member_lengths(data).min().item())


def minimum_member_angle_degrees(data: Data) -> float:
    """Return the smallest angle between incident members at any node."""
    coordinates = graph_coordinates(data)
    member_index = data.member_index
    vectors_by_node: list[list[torch.Tensor]] = [[] for _ in range(data.num_nodes)]

    for start, end in member_index.t().tolist():
        vector = coordinates[end] - coordinates[start]
        vectors_by_node[start].append(vector)
        vectors_by_node[end].append(-vector)

    minimum = 180.0
    for vectors in vectors_by_node:
        if len(vectors) < 2:
            continue
        for i, first in enumerate(vectors[:-1]):
            first_norm = first.norm().clamp_min(1e-12)
            for second in vectors[i + 1 :]:
                cosine = torch.dot(first, second) / (
                    first_norm * second.norm().clamp_min(1e-12)
                )
                angle = math.degrees(math.acos(float(cosine.clamp(-1.0, 1.0))))
                minimum = min(minimum, angle)
    return minimum


def minimum_angle_filter(minimum_degrees: float):
    """Return a PyG ``pre_filter`` callable for minimum nodal member angle."""

    def keep(data: Data) -> bool:
        return minimum_member_angle_degrees(data) >= float(minimum_degrees)

    return keep


def minimum_length_filter(minimum_length: float):
    """Return a PyG ``pre_filter`` callable for minimum member length."""

    def keep(data: Data) -> bool:
        return minimum_member_length(data) >= float(minimum_length)

    return keep


def scale_results(
    data: Data,
    *,
    force_scale: float,
    displacement_scale: float | None = None,
    length_scale: float = 1.0,
) -> Data:
    """Return a scaled copy of a truss graph.

    The helper supports the lean ``TrussBaseDataset`` representation and the
    task datasets. It never mutates the input graph, which makes it suitable for
    both ``pre_transform`` and runtime ``transform`` use.
    """
    out = data.clone()
    force_scale = float(force_scale)
    displacement_scale = force_scale if displacement_scale is None else float(displacement_scale)
    length_scale = float(length_scale)

    if hasattr(out, "x") and out.x.size(-1) >= BASE_DISPLACEMENT_SLICE.stop:
        out.x[:, BASE_COORD_SLICE] *= length_scale
        out.x[:, BASE_FORCE_SLICE] *= force_scale
        out.x[:, BASE_DISPLACEMENT_SLICE] *= displacement_scale

    if hasattr(out, "member_attr") and out.member_attr.size(-1) >= 2:
        out.member_attr[:, MEMBER_LENGTH_COLUMN] *= length_scale
        out.member_attr[:, MEMBER_AXIAL_COLUMN] *= force_scale

    if hasattr(out, "edge_attr") and out.edge_attr.size(-1) >= 2:
        out.edge_attr[:, MEMBER_LENGTH_COLUMN] *= length_scale
        out.edge_attr[:, MEMBER_AXIAL_COLUMN] *= force_scale

    for name in ("pos_raw", "member_length"):
        if hasattr(out, name):
            setattr(out, name, getattr(out, name) * length_scale)
    for name in ("axial_force", "load_scale", "static_action"):
        if hasattr(out, name):
            scale = abs(force_scale) * length_scale if name == "static_action" else force_scale
            setattr(out, name, getattr(out, name) * scale)
    for name in ("displacement", "displacement_scale"):
        if hasattr(out, name):
            setattr(out, name, getattr(out, name) * displacement_scale)
    return out


def scale_to_material(
    data: Data,
    *,
    span: float,
    load_per_length: float,
    young_modulus: float,
    area: float,
) -> Data:
    """Scale geometry, forces, and displacement to a material/load case."""
    scales = material_scales(
        span=span,
        load_per_length=load_per_length,
        young_modulus=young_modulus,
        area=area,
    )
    return scale_results(
        data,
        length_scale=scales.length,
        force_scale=scales.force,
        displacement_scale=scales.displacement,
    )
