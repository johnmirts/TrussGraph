"""Task-specific datasets for planar-truss graph learning."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset


ZENODO_RECORD_URL = "https://zenodo.org/records/18419272"
ZENODO_FILE_URL = "https://zenodo.org/records/18419272/files/{file_name}?download=1"

TYPOLOGY_FILES: Dict[str, str] = {
    "Fink": "Fink.zip",
    "Howe": "Howe.zip",
    "KTruss": "KTruss.zip",
    "Pratt": "Pratt.zip",
    "Warren": "Warren.zip",
}

_TYPOLOGY_ALIASES = {
    "fink": "Fink",
    "howe": "Howe",
    "ktruss": "KTruss",
    "k_truss": "KTruss",
    "k-truss": "KTruss",
    "pratt": "Pratt",
    "warren": "Warren",
}
_PANEL_SEED_RE = re.compile(
    r".*?(?:p|panels?)[_-]?(?P<panels>\d+).*?seed[_-]?(?P<seed>\d+)",
    re.IGNORECASE,
)
_PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_typology(name: str) -> str:
    normalized = str(name).strip()
    key = normalized.lower().replace(" ", "").replace("-", "_")
    if key in _TYPOLOGY_ALIASES:
        return _TYPOLOGY_ALIASES[key]
    if normalized in TYPOLOGY_FILES:
        return normalized
    raise ValueError(
        f"Unknown typology {name!r}. Expected one of {sorted(TYPOLOGY_FILES)}."
    )


def _positive_or_none(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive or None")
    return value


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _natural_path_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", str(path))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _sample_deterministically(values: Sequence, limit: Optional[int], seed: int) -> list:
    values = list(values)
    if limit is None or len(values) <= limit:
        return values
    rng = random.Random(seed)
    return sorted(rng.sample(values, limit), key=lambda value: str(value))


def _static_action(axial_force: Tensor, member_length: Tensor) -> Tensor:
    return (axial_force.abs() * member_length).sum()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")
    archive.extractall(destination)


def _download_typology(raw_root: Path, typology: str) -> None:
    file_name = TYPOLOGY_FILES[typology]
    url = ZENODO_FILE_URL.format(file_name=file_name)
    download_dir = raw_root / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / file_name

    if not zip_path.exists():
        print(f"Downloading {typology} from {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as exc:  # pragma: no cover - network dependent.
            raise RuntimeError(
                f"Could not download {typology} from {url}. "
                f"Download it manually from {ZENODO_RECORD_URL} and unzip it "
                f"under {raw_root}."
            ) from exc

    with zipfile.ZipFile(zip_path) as archive:
        file_names = [
            name for name in archive.namelist() if name and not name.endswith("/")
        ]
        top_levels = {Path(name).parts[0] for name in file_names if Path(name).parts}
        if typology in top_levels:
            _safe_extract(archive, raw_root)
        else:
            target_dir = raw_root / typology
            target_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, target_dir)

    if not (raw_root / typology).is_dir():
        raise RuntimeError(
            f"Downloaded {file_name}, but could not find extracted folder "
            f"{raw_root / typology}."
        )


def _select_json_paths(
    raw_root: Path,
    *,
    typologies: Optional[Sequence[str]],
    panels: Optional[Sequence[int]],
    seeds: Optional[Sequence[int]],
    max_seeds_per_panel: Optional[int],
    max_designs_per_seed: Optional[int],
    sampling_seed: int,
) -> List[Tuple[Path, str, int, int]]:
    typology_dirs = [
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name != "_downloads" and path.name in TYPOLOGY_FILES
    ]
    if typologies is not None:
        wanted = set(typologies)
        typology_dirs = [
            path for path in typology_dirs if _canonical_typology(path.name) in wanted
        ]

    seed_groups: Dict[Tuple[str, int], List[Tuple[Path, int]]] = defaultdict(list)
    for typology_dir in sorted(typology_dirs, key=lambda path: path.name):
        typology = _canonical_typology(typology_dir.name)
        for directory in typology_dir.rglob("*"):
            if not directory.is_dir():
                continue
            match = _PANEL_SEED_RE.fullmatch(directory.name)
            if match is None:
                continue
            panel_count = int(match.group("panels"))
            seed = int(match.group("seed"))
            if panels is not None and panel_count not in panels:
                continue
            if seeds is not None and seed not in seeds:
                continue
            seed_groups[(typology, panel_count)].append((directory, seed))

    selected: List[Tuple[Path, str, int, int]] = []
    for (typology, panel_count), seed_dirs in sorted(seed_groups.items()):
        seed_dirs = sorted(seed_dirs, key=lambda item: (item[1], item[0].name))
        seed_dirs = _sample_deterministically(
            seed_dirs,
            max_seeds_per_panel,
            seed=_stable_seed(sampling_seed, typology, panel_count, "seeds"),
        )
        for seed_dir, seed in seed_dirs:
            files = sorted(seed_dir.glob("*.json"), key=_natural_path_key)
            files = _sample_deterministically(
                files,
                max_designs_per_seed,
                seed=_stable_seed(sampling_seed, typology, panel_count, seed, "designs"),
            )
            selected.extend((path, typology, panel_count, seed) for path in files)

    return sorted(selected, key=lambda item: _natural_path_key(item[0]))


def _parse_common_graph(raw: dict, plane: str) -> dict[str, Tensor | int]:
    axes = _PLANE_AXES[plane]
    nodes = sorted(raw["nodes"], key=lambda node: int(node["node_id"]))
    edges = sorted(raw["edges"], key=lambda edge: int(edge.get("edge_id", 0)))
    if not nodes:
        raise ValueError("Graph contains no nodes")

    node_id_to_index = {int(node["node_id"]): idx for idx, node in enumerate(nodes)}
    coordinates_3d = torch.tensor(
        [[_safe_float(node.get(axis)) for axis in ("X", "Y", "Z")] for node in nodes],
        dtype=torch.float32,
    )
    displacement_3d = torch.tensor(
        [[_safe_float(node.get(axis)) for axis in ("Ux", "Uy", "Uz")] for node in nodes],
        dtype=torch.float32,
    )
    coordinates = coordinates_3d[:, list(axes)]
    displacement = displacement_3d[:, list(axes)]

    applied_load = torch.zeros((len(nodes), 2), dtype=torch.float32)
    for force in raw.get("forces", []):
        if int(force.get("is_load", 0)) != 1:
            continue
        anchor_id = int(force["anchor_id"])
        if anchor_id not in node_id_to_index:
            continue
        vector_3d = torch.tensor(
            [_safe_float(force.get(axis)) for axis in ("X", "Y", "Z")],
            dtype=torch.float32,
        )
        applied_load[node_id_to_index[anchor_id]] += vector_3d[list(axes)]

    free_mask = torch.tensor(
        [bool(int(node.get("is_free", 1))) for node in nodes],
        dtype=torch.bool,
    )
    valency_from_json = torch.tensor(
        [_safe_float(node.get("valency")) for node in nodes],
        dtype=torch.float32,
    )

    starts: List[int] = []
    ends: List[int] = []
    lengths: List[float] = []
    axial_forces: List[float] = []
    degree = torch.zeros(len(nodes), dtype=torch.float32)
    for edge in edges:
        start = node_id_to_index[int(edge["start_id"])]
        end = node_id_to_index[int(edge["end_id"])]
        if start == end:
            raise ValueError("Self-loop found in physical truss member")
        delta = coordinates[end] - coordinates[start]
        length = max(_safe_float(edge.get("length"), float(delta.norm())), 1e-8)
        starts.append(start)
        ends.append(end)
        lengths.append(length)
        axial_forces.append(_safe_float(edge.get("axial_f")))
        degree[start] += 1.0
        degree[end] += 1.0

    valency = torch.where(valency_from_json > 0, valency_from_json, degree)
    return {
        "coordinates": coordinates,
        "displacement": displacement,
        "applied_load": applied_load,
        "free_mask": free_mask,
        "valency": valency,
        "member_index": torch.tensor([starts, ends], dtype=torch.long),
        "member_length": torch.tensor(lengths, dtype=torch.float32),
        "axial_force": torch.tensor(axial_forces, dtype=torch.float32),
        "graph_id": int(raw.get("graph_id", -1)),
        "num_members": len(edges),
    }


def _normalize_coordinates(coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    coord_min = coordinates.min(dim=0).values
    coord_max = coordinates.max(dim=0).values
    center = 0.5 * (coord_min + coord_max)
    length_scale = (coord_max - coord_min).max().clamp_min(1e-8)
    return (coordinates - center) / length_scale, length_scale, center


def _node_inputs(common: dict[str, Tensor | int]) -> tuple[Tensor, Tensor, Tensor]:
    coordinates = common["coordinates"]
    applied_load = common["applied_load"]
    free_mask = common["free_mask"]
    valency = common["valency"]
    coordinates_norm, length_scale, _ = _normalize_coordinates(coordinates)
    load_scale = applied_load.norm(dim=-1).sum().clamp_min(1e-8)
    degree_norm = valency / valency.max().clamp_min(1.0)
    x = torch.cat(
        [
            coordinates_norm,
            applied_load / load_scale,
            free_mask.float().unsqueeze(-1),
            degree_norm.unsqueeze(-1),
        ],
        dim=-1,
    )
    return x, length_scale, load_scale


class _TrussJsonDataset(InMemoryDataset):
    node_feature_names: tuple[str, ...] = ()
    edge_feature_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    process_version = 1

    def __init__(
        self,
        root: str | Path = "data",
        *,
        typologies: Optional[Sequence[str]] = None,
        panels: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        max_seeds_per_panel: Optional[int] = None,
        max_designs_per_seed: Optional[int] = None,
        sampling_seed: int = 42,
        plane: str = "xz",
        cache_name: str,
        download_if_missing: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ) -> None:
        self.typologies = (
            tuple(sorted(_canonical_typology(name) for name in typologies))
            if typologies
            else None
        )
        self.panels = tuple(sorted(int(panel) for panel in panels)) if panels else None
        self.seeds = tuple(sorted(int(seed) for seed in seeds)) if seeds else None
        self.max_seeds_per_panel = _positive_or_none(
            max_seeds_per_panel, "max_seeds_per_panel"
        )
        self.max_designs_per_seed = _positive_or_none(
            max_designs_per_seed, "max_designs_per_seed"
        )
        self.sampling_seed = int(sampling_seed)
        self.plane = plane.lower()
        if self.plane not in _PLANE_AXES:
            raise ValueError(f"plane must be one of {sorted(_PLANE_AXES)}")
        self.download_if_missing = bool(download_if_missing)

        self._config = {
            "class": self.__class__.__name__,
            "process_version": self.process_version,
            "typologies": self.typologies,
            "panels": self.panels,
            "seeds": self.seeds,
            "max_seeds_per_panel": self.max_seeds_per_panel,
            "max_designs_per_seed": self.max_designs_per_seed,
            "sampling_seed": self.sampling_seed,
            "plane": self.plane,
        }
        digest = hashlib.sha1(
            json.dumps(self._config, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()[:12]
        self._cache_name = f"{cache_name}_{digest}.pt"
        self._metadata_name = f"{cache_name}_{digest}.json"

        super().__init__(
            root=str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self._load_processed(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.typologies or [])

    @property
    def processed_file_names(self) -> List[str]:
        return [self._cache_name]

    @property
    def metadata_path(self) -> Path:
        return Path(self.processed_dir) / self._metadata_name

    def download(self) -> None:
        if self.typologies is None:
            return
        raw_root = Path(self.raw_dir)
        raw_root.mkdir(parents=True, exist_ok=True)
        missing = [name for name in self.typologies if not (raw_root / name).is_dir()]
        if missing and not self.download_if_missing:
            raise FileNotFoundError(
                "Missing selected typology folders: "
                + ", ".join(str(raw_root / name) for name in missing)
            )
        for typology in missing:
            _download_typology(raw_root, typology)

    def process(self) -> None:
        raw_root = Path(self.raw_dir)
        selected = _select_json_paths(
            raw_root,
            typologies=self.typologies,
            panels=self.panels,
            seeds=self.seeds,
            max_seeds_per_panel=self.max_seeds_per_panel,
            max_designs_per_seed=self.max_designs_per_seed,
            sampling_seed=self.sampling_seed,
        )
        if not selected:
            raise FileNotFoundError(
                f"No matching JSON files found below {raw_root}. "
                "Check typologies, panels, and folder layout."
            )

        typology_names = sorted({item[1] for item in selected})
        typology_to_id = {name: idx for idx, name in enumerate(typology_names)}
        group_to_id: Dict[Tuple[str, int, int], int] = {}
        data_list: List[Data] = []
        skipped: List[dict[str, str]] = []

        for index, (path, typology, panel_count, seed) in enumerate(selected):
            group_key = (typology, panel_count, seed)
            group_id = group_to_id.setdefault(group_key, len(group_to_id))
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                common = _parse_common_graph(raw, self.plane)
                graph = self.build_data(
                    common,
                    typology_id=typology_to_id[typology],
                    panels=panel_count,
                    seed=seed,
                    group_id=group_id,
                )
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                data_list.append(graph)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(path), "error": str(exc)})

            if (index + 1) % 5_000 == 0:
                print(f"Processed {index + 1:,}/{len(selected):,} JSON files")

        if not data_list:
            raise RuntimeError("All selected JSON files failed conversion.")

        data, slices = self.collate(data_list)
        self.metadata = {
            "config": self._config,
            "num_selected_files": len(selected),
            "num_processed_graphs": len(data_list),
            "num_skipped_files": len(skipped),
            "typologies": typology_names,
            "typology_to_id": typology_to_id,
            "num_groups": len(group_to_id),
            "node_feature_names": self.node_feature_names,
            "edge_feature_names": self.edge_feature_names,
            "member_feature_names": getattr(self, "member_feature_names", ()),
            "target_names": self.target_names,
            "skipped_examples": skipped[:20],
        }
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices, self.metadata), self.processed_paths[0])
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )

    def _load_processed(self, path: str) -> None:
        try:
            payload = torch.load(path, weights_only=False)
        except TypeError:
            payload = torch.load(path)
        self.data, self.slices, self.metadata = payload

    def build_data(
        self,
        common: dict[str, Tensor | int],
        *,
        typology_id: int,
        panels: int,
        seed: int,
        group_id: int,
    ) -> Data:
        raise NotImplementedError


class TrussBaseDataset(_TrussJsonDataset):
    """Lean dataset for importing and manipulating raw truss results.

    Node features:
    ``valency, is_free, coord_1, coord_2, force_1, force_2, disp_1, disp_2``.

    Member features:
    ``length, axial_force``.
    """

    node_feature_names = (
        "valency",
        "is_free",
        "coord_1",
        "coord_2",
        "force_1",
        "force_2",
        "disp_1",
        "disp_2",
    )
    edge_feature_names = ("length", "axial_force")
    member_feature_names = ("length", "axial_force")
    target_names: tuple[str, ...] = ()
    process_version = 1

    def __init__(self, *args, cache_name: str = "truss_base", **kwargs) -> None:
        super().__init__(*args, cache_name=cache_name, **kwargs)

    def build_data(
        self,
        common: dict[str, Tensor | int],
        *,
        typology_id: int,
        panels: int,
        seed: int,
        group_id: int,
    ) -> Data:
        valency = common["valency"]
        free_mask = common["free_mask"]
        coordinates = common["coordinates"]
        applied_load = common["applied_load"]
        displacement = common["displacement"]
        member_index = common["member_index"]
        member_attr = torch.stack(
            [common["member_length"], common["axial_force"]],
            dim=-1,
        )

        return Data(
            x=torch.cat(
                [
                    valency.unsqueeze(-1),
                    free_mask.float().unsqueeze(-1),
                    coordinates,
                    applied_load,
                    displacement,
                ],
                dim=-1,
            ),
            edge_index=torch.cat([member_index, member_index.flip(0)], dim=1),
            edge_attr=torch.cat([member_attr, member_attr], dim=0),
            member_index=member_index,
            member_attr=member_attr,
            typology_id=torch.tensor([typology_id], dtype=torch.long),
            panels=torch.tensor([panels], dtype=torch.long),
            seed=torch.tensor([seed], dtype=torch.long),
            group_id=torch.tensor([group_id], dtype=torch.long),
            graph_id=torch.tensor([common["graph_id"]], dtype=torch.long),
            num_members=torch.tensor([common["num_members"]], dtype=torch.long),
        )


class AxialForceDataset(_TrussJsonDataset):
    """Edge-level axial-force dataset."""

    node_feature_names = (
        "coord_1_norm",
        "coord_2_norm",
        "load_1_norm",
        "load_2_norm",
        "is_free",
        "degree_norm",
    )
    edge_feature_names = ("delta_coord_1_norm", "delta_coord_2_norm", "length_norm")
    member_feature_names = ("length_norm", "cos2", "sin2", "cos_sin")
    target_names = ("axial_force_over_total_applied_load_magnitude",)
    process_version = 4

    def __init__(self, *args, cache_name: str = "truss_axial_force", **kwargs) -> None:
        super().__init__(*args, cache_name=cache_name, **kwargs)

    def build_data(
        self,
        common: dict[str, Tensor | int],
        *,
        typology_id: int,
        panels: int,
        seed: int,
        group_id: int,
    ) -> Data:
        coordinates = common["coordinates"]
        member_index = common["member_index"]
        member_length = common["member_length"]
        axial_force = common["axial_force"]
        x, length_scale, load_scale = _node_inputs(common)

        start, end = member_index
        delta = coordinates[end] - coordinates[start]
        delta_norm = delta / length_scale
        length_norm = (member_length / length_scale).unsqueeze(-1)
        forward_attr = torch.cat([delta_norm, length_norm], dim=-1)
        reverse_attr = forward_attr.clone()
        reverse_attr[:, :2] *= -1.0
        edge_index = torch.cat([member_index, member_index.flip(0)], dim=1)
        edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

        direction = delta / member_length.clamp_min(1e-8).unsqueeze(-1)
        member_attr = torch.cat(
            [
                length_norm,
                direction[:, :1] ** 2,
                direction[:, 1:2] ** 2,
                direction[:, :1] * direction[:, 1:2],
            ],
            dim=-1,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            member_index=member_index,
            member_attr=member_attr,
            y=axial_force / load_scale,
            axial_force=axial_force,
            member_length=member_length,
            load_scale=load_scale.view(1),
            static_action=_static_action(axial_force, member_length).view(1),
            pos_raw=coordinates,
            typology_id=torch.tensor([typology_id], dtype=torch.long),
            panels=torch.tensor([panels], dtype=torch.long),
            seed=torch.tensor([seed], dtype=torch.long),
            group_id=torch.tensor([group_id], dtype=torch.long),
            graph_id=torch.tensor([common["graph_id"]], dtype=torch.long),
            num_members=torch.tensor([common["num_members"]], dtype=torch.long),
        )


class StaticActionDataset(AxialForceDataset):
    """Graph-level static action evaluated from predicted member forces."""


class TrussDisplacementDataset(_TrussJsonDataset):
    """Node-level in-plane displacement dataset."""

    node_feature_names = (
        "coord_1_norm",
        "coord_2_norm",
        "load_1_norm",
        "load_2_norm",
        "is_free",
        "degree_norm",
    )
    edge_feature_names = ("delta_coord_1_norm", "delta_coord_2_norm")
    target_names = ("disp_1_norm", "disp_2_norm")
    process_version = 4

    def __init__(
        self,
        root: str | Path = "data",
        *,
        families: Optional[Sequence[str]] = None,
        typologies: Optional[Sequence[str]] = None,
        cache_name: str = "truss_displacement",
        **kwargs,
    ) -> None:
        if families is not None and typologies is not None:
            raise ValueError("Use either families or typologies, not both.")
        selected = typologies if typologies is not None else families
        super().__init__(root=root, typologies=selected, cache_name=cache_name, **kwargs)

    def build_data(
        self,
        common: dict[str, Tensor | int],
        *,
        typology_id: int,
        panels: int,
        seed: int,
        group_id: int,
    ) -> Data:
        coordinates = common["coordinates"]
        member_index = common["member_index"]
        displacement = common["displacement"]
        x, length_scale, force_scale = _node_inputs(common)
        displacement_scale = (force_scale * length_scale).clamp_min(1e-8)

        start, end = member_index
        delta_norm = (coordinates[end] - coordinates[start]) / length_scale
        edge_index = torch.cat([member_index, member_index.flip(0)], dim=1)
        edge_attr = torch.cat([delta_norm, -delta_norm], dim=0).to(torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=displacement / displacement_scale,
            free_mask=common["free_mask"],
            displacement=displacement,
            displacement_scale=displacement_scale.view(1),
            member_index=member_index,
            pos_raw=coordinates,
            typology_id=torch.tensor([typology_id], dtype=torch.long),
            panels=torch.tensor([panels], dtype=torch.long),
            seed=torch.tensor([seed], dtype=torch.long),
            group_id=torch.tensor([group_id], dtype=torch.long),
            graph_id=torch.tensor([common["graph_id"]], dtype=torch.long),
            num_members=torch.tensor([common["num_members"]], dtype=torch.long),
        )
