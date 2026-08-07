from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim]
        if num_layers > 1:
            dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(out_dim)

        layers = []
        for idx, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(dim_in, dim_out))
            is_last = idx == len(dims) - 2
            if not is_last or final_activation:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TrussMessageBlock(MessagePassing):
    """Residual edge-aware node message-passing block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(aggr="mean", node_dim=0)
        self.message_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.update_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, h: Tensor, edge_index: Tensor, edge_embedding: Tensor) -> Tensor:
        aggregated = self.propagate(edge_index=edge_index, h=h, edge_embedding=edge_embedding)
        update = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        h = h + F.dropout(update, p=self.dropout, training=self.training)
        return self.norm(h)

    def message(self, h_i: Tensor, h_j: Tensor, edge_embedding: Tensor) -> Tensor:
        return self.message_mlp(torch.cat([h_i, h_j - h_i, edge_embedding], dim=-1))


class TrussAxialForceGNN(nn.Module):
    """Edge-level GNN surrogate for axial forces prediction."""

    def __init__(
        self,
        node_dim: int = 6,
        directed_edge_dim: int = 3,
        member_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_encoder = MLP(directed_edge_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.member_encoder = MLP(member_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TrussMessageBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.member_decoder = MLP(
            3 * hidden_dim,
            hidden_dim,
            1,
            num_layers=3,
            dropout=dropout,
        )

    @staticmethod
    def _member_batch(data) -> Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch[data.member_index[0]]
        return torch.zeros(
            data.member_index.size(1), dtype=torch.long, device=data.member_index.device
        )

    def forward(self, data, compute_static_action: bool = True) -> Dict[str, Tensor]:
        h = self.node_encoder(data.x)
        edge_embedding = self.edge_encoder(data.edge_attr)
        for block in self.blocks:
            h = block(h, data.edge_index, edge_embedding)

        start, end = data.member_index
        h_start = h[start]
        h_end = h[end]
        member_embedding = self.member_encoder(data.member_attr)

        # Symmetric endpoint representation: invariant to start/end ordering.
        pair_embedding = torch.cat(
            [h_start + h_end, torch.abs(h_start - h_end), member_embedding], dim=-1
        )
        axial_norm = self.member_decoder(pair_embedding).squeeze(-1)

        member_batch = self._member_batch(data)
        output = {
            "axial_norm": axial_norm,
            "member_batch": member_batch,
        }

        # Physical forces and static action are not needed by the training loss.
        if compute_static_action:
            graph_load_scale = data.load_scale.view(-1)
            axial_force = axial_norm * graph_load_scale[member_batch]
            num_graphs = int(graph_load_scale.numel())
            static_action = torch.zeros(
                num_graphs, dtype=axial_force.dtype, device=axial_force.device
            )
            static_action.index_add_(
                0, member_batch, axial_force.abs() * data.member_length
            )
            output["axial_force"] = axial_force
            output["static_action"] = static_action

        return output
