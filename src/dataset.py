from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset


_PANEL_SEED_RE = re.compile(r".*?(?:p|panels?)[_-]?(?P<panels>\d+).*?seed[_-]?(?P<seed>\d+)", re.IGNORECASE)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_panels_seed(path: Path, payload: dict) -> Tuple[int, int]:
    panels = int(payload.get("panels", -1))
    seed = int(payload.get("seed", -1))
    match = _PANEL_SEED_RE.match(path.parent.name)
    if match:
        if panels < 0:
            panels = int(match.group("panels"))
        if seed < 0:
            seed = int(match.group("seed"))
    return panels, seed


def _static_action(axial_force: Tensor, member_length: Tensor) -> Tensor:
    """Static action measure: sum_e |N_e| L_e."""
    return (axial_force.abs() * member_length).sum()


def json_to_pyg_data(
    json_path: str | Path,
    raw_root: str | Path,
    typology_to_id: Dict[str, int],
    group_to_id: Dict[str, int],
) -> Data:
    """Convert one truss JSON file to a PyG ``Data`` object.

    Node features
    -------------
    [x/span, z/span, applied_fx/load_scale, applied_fz/load_scale,
     support_flag, normalized_degree]

    Directed message-passing edge features
    ---------------------------------------
    [dx/span, dz/span, length/span]

    Original-member features used by the edge decoder
    -------------------------------------------------
    [length/span, |cos(theta)|, |sin(theta)|]
    """
    json_path = Path(json_path)
    raw_root = Path(raw_root)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_nodes = sorted(payload["nodes"], key=lambda n: int(n["node_id"]))
    raw_edges = sorted(payload["edges"], key=lambda e: int(e["edge_id"]))
    node_id_to_idx = {int(node["node_id"]): idx for idx, node in enumerate(raw_nodes)}

    pos_raw = torch.tensor(
        [[_safe_float(node.get("X")), _safe_float(node.get("Z"))] for node in raw_nodes],
        dtype=torch.float32,
    )
    mins = pos_raw.min(dim=0).values
    maxs = pos_raw.max(dim=0).values
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max().clamp_min(1e-8)
    pos = (pos_raw - center) / span

    # Only external applied loads are inputs. Reactions (is_load == 0) are excluded.
    applied_load = torch.zeros((len(raw_nodes), 2), dtype=torch.float32)
    for force in payload.get("forces", []):
        if int(force.get("is_load", 0)) != 1:
            continue
        anchor_id = int(force["anchor_id"])
        if anchor_id not in node_id_to_idx:
            continue
        idx = node_id_to_idx[anchor_id]
        applied_load[idx, 0] += _safe_float(force.get("X"))
        applied_load[idx, 1] += _safe_float(force.get("Z"))

    # Sum of applied load-vector magnitudes. Axial forces are normalized by the
    # same quantity so the network learns a load-scale-independent response.
    load_scale = applied_load.norm(dim=-1).sum().clamp_min(1e-8)
    applied_load_norm = applied_load / load_scale

    support_flag = torch.tensor(
        [[1.0 - _safe_float(node.get("is_free", 1.0))] for node in raw_nodes],
        dtype=torch.float32,
    )
    degree = torch.tensor(
        [[_safe_float(node.get("valency", 0.0))] for node in raw_nodes],
        dtype=torch.float32,
    )
    degree = degree / degree.max().clamp_min(1.0)

    x = torch.cat([pos, applied_load_norm, support_flag, degree], dim=-1)

    starts: List[int] = []
    ends: List[int] = []
    lengths: List[float] = []
    axial: List[float] = []
    directed_attr_forward: List[List[float]] = []
    member_attr: List[List[float]] = []

    for edge in raw_edges:
        start = node_id_to_idx[int(edge["start_id"])]
        end = node_id_to_idx[int(edge["end_id"])]
        delta_raw = pos_raw[end] - pos_raw[start]
        computed_length = float(delta_raw.norm().item())
        length = _safe_float(edge.get("length"), computed_length)
        length = max(length, 1e-8)
        delta_norm = delta_raw / span
        length_norm = length / float(span.item())
        direction = delta_raw / length

        starts.append(start)
        ends.append(end)
        lengths.append(length)
        axial.append(_safe_float(edge.get("axial_f")))
        directed_attr_forward.append(
            [float(delta_norm[0]), float(delta_norm[1]), float(length_norm)]
        )
        member_attr.append([
            length_norm,
            float(direction[0])**2,
            float(direction[1])**2,
            float(direction[0]) * float(direction[1]) #cos_theta * sin_theta
        ])

    member_index = torch.tensor([starts, ends], dtype=torch.long)
    forward_attr = torch.tensor(directed_attr_forward, dtype=torch.float32)
    reverse_attr = forward_attr.clone()
    reverse_attr[:, :2] *= -1.0

    # Duplicate members in both directions only for message passing. Predictions
    # are made once per physical member through ``member_index``.
    edge_index = torch.cat([member_index, member_index.flip(0)], dim=1)
    edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

    member_length = torch.tensor(lengths, dtype=torch.float32)
    axial_force = torch.tensor(axial, dtype=torch.float32)
    y = axial_force / load_scale

    relative = json_path.relative_to(raw_root)
    typology = relative.parts[0] if len(relative.parts) > 1 else "unknown"
    panels, seed = _parse_panels_seed(json_path, payload)
    group_key = f"{typology}/panels_{panels}/seed_{seed}"

    graph = Data(
        x=x,
        pos=pos,
        pos_raw=pos_raw,
        edge_index=edge_index,
        edge_attr=edge_attr,
        member_index=member_index,
        member_attr=torch.tensor(member_attr, dtype=torch.float32),
        member_length=member_length,
        y=y,
        axial_force=axial_force,
        load_scale=load_scale.view(1),
        static_action=_static_action(axial_force, member_length).view(1),
        typology_id=torch.tensor([typology_to_id[typology]], dtype=torch.long),
        group_id=torch.tensor([group_to_id[group_key]], dtype=torch.long),
        panels=torch.tensor([panels], dtype=torch.long),
        seed=torch.tensor([seed], dtype=torch.long),
        graph_id=torch.tensor([int(payload.get("graph_id", -1))], dtype=torch.long),
        num_members=torch.tensor([len(raw_edges)], dtype=torch.long),
    )
    return graph


class TrussDataset(InMemoryDataset):
    """PyTorch Geometric in-memory dataset."""

    def __init__(
        self,
        root: str | Path = "data",
        cache_name: str = "trusses_v1",
        typologies: Sequence[str] | None = None,
        max_seeds_per_panel: int | None = None,
        max_designs_per_seed: int | None = None,
        sampling_seed: int = 42,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ) -> None:
        self.selected_typologies = (
            tuple(sorted(str(name) for name in typologies)) if typologies is not None else None
        )
        self.max_seeds_per_panel = max_seeds_per_panel
        self.max_designs_per_seed = max_designs_per_seed
        self.sampling_seed = int(sampling_seed)

        for name, value in (
            ("max_seeds_per_panel", max_seeds_per_panel),
            ("max_designs_per_seed", max_designs_per_seed),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")

        selection = {
            "typologies": self.selected_typologies,
            "max_seeds_per_panel": self.max_seeds_per_panel,
            "max_designs_per_seed": self.max_designs_per_seed,
            "sampling_seed": self.sampling_seed,
        }
        signature = hashlib.sha1(
            json.dumps(selection, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        self.cache_name = f"{cache_name}_{signature}"

        super().__init__(
            root=str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        with open(self.processed_paths[0], "rb") as handle:
            payload = pickle.load(handle)
        self.data = payload["data"]
        self.slices = payload["slices"]
        self.metadata = payload["metadata"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.cache_name}.pkl"]

    def download(self) -> None:
        # Local dataset: nothing to download for now.
        return None

    def _select_json_paths(self, raw_root: Path) -> List[Path]:
        """Select seed folders and designs before opening any JSON files.

        Sampling is balanced per ``(typology, panel_count)``: for every panel
        count, at most ``max_seeds_per_panel`` seed folders are selected, and
        from each selected folder at most ``max_designs_per_seed`` designs are
        selected. The selection is deterministic for ``sampling_seed``.
        """
        typology_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
        if self.selected_typologies is not None:
            allowed = set(self.selected_typologies)
            typology_dirs = [path for path in typology_dirs if path.name in allowed]

        selected: List[Path] = []
        for typology_dir in typology_dirs:
            by_panel: Dict[int, List[Path]] = defaultdict(list)
            for seed_dir in sorted(path for path in typology_dir.iterdir() if path.is_dir()):
                match = _PANEL_SEED_RE.match(seed_dir.name)
                if match is None:
                    continue
                by_panel[int(match.group("panels"))].append(seed_dir)

            for panels, seed_dirs in sorted(by_panel.items()):
                seed_dirs = sorted(seed_dirs)
                if self.max_seeds_per_panel is not None:
                    rng = random.Random(
                        f"{self.sampling_seed}:{typology_dir.name}:panels_{panels}"
                    )
                    count = min(self.max_seeds_per_panel, len(seed_dirs))
                    seed_dirs = sorted(rng.sample(seed_dirs, count))

                for seed_dir in seed_dirs:
                    graph_paths = sorted(seed_dir.glob("*.json"))
                    if self.max_designs_per_seed is not None:
                        rng = random.Random(
                            f"{self.sampling_seed}:{typology_dir.name}:{seed_dir.name}"
                        )
                        count = min(self.max_designs_per_seed, len(graph_paths))
                        graph_paths = sorted(rng.sample(graph_paths, count))
                    selected.extend(graph_paths)

        return sorted(selected)

    def process(self) -> None:
        raw_root = Path(self.raw_dir)
        json_paths = self._select_json_paths(raw_root)
        if not json_paths:
            raise FileNotFoundError(
                f"No selected JSON files found below {raw_root}. Check the typology and sampling filters."
            )

        typologies = []
        group_keys: List[str] = []
        for path in json_paths:
            relative = path.relative_to(raw_root)
            typology = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            typologies.append(typology)
            panels, seed = _parse_panels_seed(path, {})
            if panels < 0 or seed < 0:
                with path.open("r", encoding="utf-8") as handle:
                    header = json.load(handle)
                panels, seed = _parse_panels_seed(path, header)
            group_keys.append(f"{typology}/panels_{panels}/seed_{seed}")

        typologies = sorted(typologies)
        typology_to_id = {name: idx for idx, name in enumerate(typologies)}
        group_to_id = {name: idx for idx, name in enumerate(sorted(set(group_keys)))}

        data_list: List[Data] = []
        for path in json_paths:
            graph = json_to_pyg_data(path, raw_root, typology_to_id, group_to_id)
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        metadata = {
            "num_graphs": len(data_list),
            "num_raw_files": len(json_paths),
            "selection": {
                "typologies": self.selected_typologies,
                "max_seeds_per_panel": self.max_seeds_per_panel,
                "max_designs_per_seed": self.max_designs_per_seed,
                "sampling_seed": self.sampling_seed,
            },
            "typologies": typologies,
            "typology_to_id": typology_to_id,
            "group_to_id": group_to_id,
            "node_features": [
                "x_over_span",
                "z_over_span",
                "applied_fx_over_load_scale",
                "applied_fz_over_load_scale",
                "support_flag",
                "normalized_degree",
            ],
            "directed_edge_features": ["dx_over_span", "dz_over_span", "length_over_span"],
            "member_features": ["length_over_span", "abs_cos_theta", "abs_sin_theta"],
            "target": "axial_force_over_total_applied_load_magnitude",
            "static_action": "sum(abs(axial_force) * member_length)",
        }
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        with open(self.processed_paths[0], "wb") as handle:
            pickle.dump(
                {"data": data, "slices": slices, "metadata": metadata},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def random_split_indices(
    num_graphs: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    if num_graphs < 3:
        raise ValueError("At least three graphs are required for train/validation/test splitting")

    indices = list(range(num_graphs))
    random.Random(seed).shuffle(indices)
    n_train = max(1, min(int(round(train_fraction * num_graphs)), num_graphs - 2))
    n_val = max(1, min(int(round(val_fraction * num_graphs)), num_graphs - n_train - 1))
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def grouped_split_indices(
    dataset: TrussDataset,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split complete typology/panel/seed groups to reduce near-duplicate leakage.

    Falls back to a graph-random split when fewer than three distinct groups
    are present.
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        group_id = int(dataset.get(idx).group_id.item())
        groups[group_id].append(idx)

    if len(groups) < 3:
        return random_split_indices(len(dataset), train_fraction, val_fraction, seed)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    n_groups = len(group_ids)
    n_train = max(1, int(round(train_fraction * n_groups)))
    n_val = max(1, int(round(val_fraction * n_groups)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train : n_train + n_val])
    test_groups = set(group_ids[n_train + n_val :])

    train_idx = [i for gid in train_groups for i in groups[gid]]
    val_idx = [i for gid in val_groups for i in groups[gid]]
    test_idx = [i for gid in test_groups for i in groups[gid]]
    return train_idx, val_idx, test_idx
