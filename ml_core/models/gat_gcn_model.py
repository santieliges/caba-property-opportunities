import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .baseModel import BaseModel


class _SegmentOps:
    """Minimal scatter-based helpers to stay dependency free."""

    @staticmethod
    def softmax_per_dst(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Numerically stable softmax of scores grouped by destination node.

        Args:
            scores: Tensor [E, d] attention logits.
            dst: Tensor [E] destination node indices.
            num_nodes: total number of nodes (N).
        Returns:
            Tensor [E, d] with softmax values per destination node.
        """
        max_per_dst = torch.full((num_nodes, scores.size(1)), -math.inf, device=scores.device, dtype=scores.dtype)
        max_per_dst = max_per_dst.scatter_reduce(0, dst.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True)

        stabilized = scores - max_per_dst[dst]
        exp_scores = torch.exp(stabilized)

        denom = torch.zeros_like(max_per_dst)
        denom = denom.scatter_add(0, dst.unsqueeze(-1).expand_as(scores), exp_scores)

        return exp_scores / (denom[dst] + 1e-9)

    @staticmethod
    def scatter_sum(values: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = torch.zeros(num_nodes, values.size(1), device=values.device, dtype=values.dtype)
        return out.scatter_add(0, dst.unsqueeze(-1).expand_as(values), values)


class GraphAttentionLayer(nn.Module):
    """GCN-style layer with explicit Q/K/V/U attention over edges.

    Follows the description in the cited paper's Chapter 3:
    - Each node provides queries (Q) and keys/values (K, V).
    - Each edge contributes features U (edge context) and an attention bias.
    - Attention scores are computed for every incoming neighbor, softmaxed per
      destination node, then used for a weighted aggregation of V + U.
    - The new state mixes the aggregated representation with the original one
      through a learnable gate (residual-like update).
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden ({hidden}) debe ser divisible por num_heads ({num_heads})")

        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.out_dim = hidden  # alias para evitar confusiones

        self.W_q = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_u = nn.Linear(edge_dim, self.out_dim, bias=False)

        # Bias term modulating attention with edge attributes
        self.edge_bias = nn.Linear(edge_dim, self.num_heads, bias=False)

        # Projects original features into hidden space for the gated residual
        self.skip_proj = nn.Linear(in_dim, self.out_dim)

        # Gate controls how much of the aggregated message is mixed with the skip
        self.gate = nn.Linear(self.out_dim * 2, self.out_dim)

        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index  # [E]

        q = self.W_q(x).view(x.size(0), self.num_heads, self.head_dim)  # [N, H, Dh]
        k = self.W_k(x).view(x.size(0), self.num_heads, self.head_dim)
        v = self.W_v(x).view(x.size(0), self.num_heads, self.head_dim)
        u = self.W_u(edge_attr).view(edge_attr.size(0), self.num_heads, self.head_dim)  # [E, H, Dh]

        q_dst = q[dst]
        k_src = k[src]
        v_src = v[src]

        # Attention logits with edge trans (U) + optional bias per cabeza
        att_logits = (q_dst * (k_src + u)).sum(dim=-1) / math.sqrt(self.head_dim)
        att_logits = att_logits + self.edge_bias(edge_attr)  # [E, H]

        alphas = _SegmentOps.softmax_per_dst(att_logits, dst, x.size(0))  # [E, H]

        # Weighted aggregation of value + edge context per cabeza
        messages = alphas.unsqueeze(-1) * (v_src + u)
        agg = _SegmentOps.scatter_sum(messages.view(edge_attr.size(0), -1), dst, x.size(0))  # [N, H*Dh]
        agg = agg.view(x.size(0), self.num_heads, self.head_dim)

        agg_flat = agg.view(x.size(0), self.out_dim)
        skip = self.skip_proj(x)

        # Gated fusion between aggregated message and skip connection
        gate = torch.sigmoid(self.gate(torch.cat([agg_flat, skip], dim=-1)))
        h_next = gate * agg_flat + (1 - gate) * skip
        h_next = self.activation(self.dropout(h_next))
        return h_next


class GraphAttentionGCN(BaseModel, nn.Module):
    """End-to-end regression model using stacked GraphAttentionLayer blocks."""

    def __init__(
        self,
        feature_names: Sequence[str],
        edge_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        patience: int = 100,
        min_delta: float = 1e-4,
    ):
        BaseModel.__init__(self)
        nn.Module.__init__(self)

        self.feature_names_ = list(feature_names)
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.patience = patience
        self.min_delta = min_delta

        layers = []
        in_dim = len(self.feature_names_)
        for _ in range(num_layers):
            layers.append(GraphAttentionLayer(in_dim, edge_dim, hidden, num_heads=num_heads, dropout=dropout))
            in_dim = hidden
        self.layers = nn.ModuleList(layers)

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.to(self.device)

        # cache used during predict
        self.edge_index_ = None
        self.edge_attr_ = None

    # --- data helpers ---------------------------------------------------
    def _prepare_tensors(
        self,
        X,
        y: Optional[Iterable] = None,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if edge_index is None or edge_attr is None:
            if self.edge_index_ is None or self.edge_attr_ is None:
                raise ValueError("edge_index y edge_attr deben pasarse la primera vez")
            edge_index = self.edge_index_
            edge_attr = self.edge_attr_

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index debe tener shape (2, E)")

        x_tensor = torch.as_tensor(np.asarray(X[self.feature_names_].values), dtype=torch.float32, device=self.device)
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)

        y_tensor = None
        if y is not None:
            y_tensor = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32, device=self.device)

        return x_tensor, y_tensor, edge_index_tensor, edge_attr_tensor

    # --- core forward ---------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.readout(h).squeeze(-1)

    # --- public API -----------------------------------------------------
    def fit(
        self,
        X,
        y,
        coords=None,
        *,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        epochs: int = 200,
    ):
        x_tensor, y_tensor, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X, y=y, edge_index=edge_index, edge_attr=edge_attr
        )

        self.edge_index_ = edge_index
        self.edge_attr_ = edge_attr

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        patience_left = self.patience

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
            loss = loss_fn(preds.view(-1, 1), y_tensor)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss + self.min_delta < best_loss:
                best_loss = current_loss
                patience_left = self.patience
            else:
                patience_left -= 1

            if patience_left <= 0:
                print(f"Early stopping en epoch {epoch + 1}, best loss={best_loss:.6f}")
                break

        self.is_fitted_ = True
        self.X_train_ = X
        self.y_train_ = np.asarray(y)
        return self

    def predict(
        self,
        X,
        coords=None,
        *,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
    ):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        x_tensor, _, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X, y=None, edge_index=edge_index, edge_attr=edge_attr
        )

        self.eval()
        with torch.no_grad():
            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
        return preds.cpu().numpy()


__all__ = ["GraphAttentionGCN", "GraphAttentionLayer"]
