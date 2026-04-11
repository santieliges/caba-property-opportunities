import math
from typing import Iterable, Optional, Sequence, Tuple

import warnings

import numpy as np
import torch
import torch.nn as nn
import numpy as np
from .baseModel import BaseModel
from sklearn.neighbors import BallTree

class _GraphBlock(nn.Module):
    """Single message-passing block with edge MLP + multihead attention aggregation + node update."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int,
        heads: int = 1,
        dropout: float = 0.0,
        use_sar: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        self.use_sar = use_sar

        sar_edge_dim = 1 if use_sar else 0
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim + sar_edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Atención sobre la representación oculta de la arista
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, heads, bias=False),
        )

        self.head_merge = nn.Sequential(
            nn.Linear(hidden * heads, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Softmax por nodo destino para varias cabezas.

        scores: [E, H]
        return: [E, H]
        """
        max_per_dst = torch.full((num_nodes, scores.size(1)), -math.inf, device=scores.device, dtype=scores.dtype)
        max_per_dst = max_per_dst.scatter_reduce(0, dst.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True)

        stabilized = scores - max_per_dst[dst]
        exp_scores = torch.exp(stabilized)

        denom = torch.zeros_like(max_per_dst)
        denom = denom.scatter_add(0, dst.unsqueeze(-1).expand_as(scores), exp_scores)

        return exp_scores / (denom[dst] + 1e-9)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y_node: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index  # [E]
        h_src, h_dst = x[src], x[dst]

        edge_input = [h_src, h_dst, edge_attr]
        if self.use_sar:
            if y_node is None:
                y_src = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
            else:
                y_src = y_node[src]
            edge_input.append(y_src)
        edge_input = torch.cat(edge_input, dim=-1)
        e_msg = self.edge_mlp(edge_input)  # [E, hidden]

        att_scores = self.att_mlp(e_msg)  # [E, H]
        alphas = self._segment_softmax(att_scores, dst, x.size(0))  # [E, H]

        # Agregamos por cabeza de forma vectorizada
        agg_heads = torch.zeros(x.size(0), self.heads, e_msg.size(1), device=x.device, dtype=x.dtype)
        weighted = alphas.unsqueeze(-1) * e_msg.unsqueeze(1)  # [E, H, hidden]
        dst_exp = dst.unsqueeze(1).expand(-1, self.heads)  # [E, H]
        h_exp = torch.arange(self.heads, device=x.device).unsqueeze(0).expand(len(dst), -1)  # [E, H]
        agg_heads[dst_exp, h_exp] += weighted
        agg = agg_heads.reshape(x.size(0), self.heads * self.hidden)  # [N, H*hidden]
        agg = self.head_merge(agg)  # [N, hidden]

        node_input = torch.cat([x, agg], dim=-1)
        return self.node_mlp(node_input)


class RentalPriceGNN(BaseModel, nn.Module):
    """
    GNN de prueba para precio de alquiler (log-precio) usando message passing con atención.

    - Nodos: departamentos con features numéricas.
    - Aristas: conectan departamentos; `edge_attr` incluye al menos la distancia.
    - Mensajes: MLP sobre [h_src, h_dst, distancia].
    - Agregación: atención por nodo destino.
    - Update: MLP sobre [h_dst, agregación].
    - Readout: MLP a escalar (log precio).

    Se entrena con MSE sobre `y` (se asume ya log-transformado).
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        edge_dim: int = 1,
        hidden: int = 64,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        patience: int = 200,  # Early stopping patience
        min_delta: float = 1e-4,  # Minimum improvement
        lr_patience: int = 50,  # LR scheduler patience
        lr_factor: float = 0.5,  # LR reduction factor
        use_sar: bool = False,
        use_layernorm: bool = False,
    ):
        BaseModel.__init__(self)
        nn.Module.__init__(self)

        self.feature_names_ = list(feature_names)
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.patience = patience
        self.min_delta = min_delta
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.use_sar = use_sar
        self.use_layernorm = use_layernorm

        self.blocks = nn.ModuleList()
        in_dim = len(self.feature_names_)
        for _ in range(num_layers):
            block = _GraphBlock(
                in_dim=in_dim,
                edge_dim=edge_dim,
                hidden=hidden,
                heads=heads,
                dropout=dropout,
                use_sar=self.use_sar,
            )
            self.blocks.append(block)
            in_dim = hidden

        # LayerNorm después de cada bloque (opcional)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)]) if self.use_layernorm else None

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.to(self.device)

        # Persisted during fit
        self.edge_index_ = None
        self.edge_attr_ = None
        self.n_train_ = None

    def _prepare_tensors(
        self,
        X,
        y: Optional[Iterable] = None,
        y_node: Optional[Iterable] = None,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if edge_index is None or edge_attr is None:
            if self.edge_index_ is None or self.edge_attr_ is None:
                raise ValueError("Debe proveer edge_index y edge_attr la primera vez.")
            edge_index = self.edge_index_
            edge_attr = self.edge_attr_

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index debe tener shape (2, num_edges).")

        x_tensor = torch.as_tensor(np.asarray(X[self.feature_names_].values), dtype=torch.float32, device=self.device)
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)

        y_tensor = None
        if y is not None:
            y_tensor = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32, device=self.device)

        y_node_tensor = None
        if y_node is not None:
            y_node_tensor = torch.as_tensor(np.asarray(y_node).reshape(-1, 1), dtype=torch.float32, device=self.device)

        return x_tensor, y_tensor, y_node_tensor, edge_index_tensor, edge_attr_tensor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y_node: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h, edge_index, edge_attr, y_node=y_node)
            if self.use_layernorm:
                h = self.layer_norms[i](h)
        return self.readout(h).squeeze(-1)

    def fit(
        self,
        X,
        y,
        coords=None,
        *,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        epochs: int = 300,
    ):
        x_tensor, y_tensor, y_node_tensor, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X, y, y_node=y if self.use_sar else None, edge_index=edge_index, edge_attr=edge_attr
        )

        self.edge_index_ = edge_index
        self.edge_attr_ = edge_attr
        self.n_train_ = len(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, verbose=True
        )

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            optimizer.zero_grad()

            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor, y_node=y_node_tensor)
            loss = loss_fn(preds.view(-1, 1), y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())


            # Early stopping
            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
                break

            # Log loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.6f}")

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
        y_node: Optional[Iterable] = None,
        inductive: bool = False,
        sar_iterations: int = 1,
        train_mask: Optional[Iterable[bool]] = None,
    ):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        # Si inductive=True y coords proporcionados, construir grafo solo con nodos de test
        if inductive and coords is not None:            
            coords_rad = np.deg2rad(np.array(coords))
            tree = BallTree(coords_rad, metric='haversine')
            dist, idx = tree.query(coords_rad, k=self.k if hasattr(self, 'k') else 8 + 1)  # Usar k similar al train
            src, dst, edge_feat = [], [], []
            for i in range(len(coords)):
                for j, d in zip(idx[i, 1:], dist[i, 1:]):
                    src.append(i)
                    dst.append(int(j))
                    edge_feat.append([d * 6371.0])
            edge_index = np.vstack([src, dst])
            edge_attr = np.asarray(edge_feat, dtype=np.float32)

        x_tensor, _, y_node_tensor, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X, y=None, y_node=y_node, edge_index=edge_index, edge_attr=edge_attr
        )

        # Para transductive con SAR, usar y_train reales para nodos de train
        if self.use_sar and not inductive and hasattr(self, 'y_train_') and y_node_tensor is None:
            n_train = self.n_train_ if self.n_train_ is not None else len(self.y_train_)

            # Prefer explicit mask to evitar supuestos de orden
            if train_mask is not None:
                mask = np.asarray(train_mask, dtype=bool)
                if mask.shape[0] != len(X):
                    raise ValueError("train_mask debe tener la misma longitud que X en predict")
                y_node_combined = np.zeros((len(X), 1), dtype=np.float32)
                if mask.sum() != n_train:
                    warnings.warn(
                        f"train_mask tiene {mask.sum()} verdaderos pero se esperaban {n_train}; se usarán los que haya",
                        RuntimeWarning,
                    )
                idx_train = np.where(mask)[0][: len(self.y_train_)]
                y_node_combined[idx_train, 0] = self.y_train_[: len(idx_train)]
            else:
                if len(X) < n_train:
                    raise ValueError(
                        "En modo transductive con SAR, X debe contener al menos los nodos de train; "
                        "pase train_mask para no depender del orden."
                    )
                y_node_combined = np.concatenate([self.y_train_, np.zeros(len(X) - n_train)], axis=0).reshape(-1, 1)

            y_node_tensor = torch.as_tensor(y_node_combined, dtype=torch.float32, device=self.device)

        if self.use_sar and y_node_tensor is None:
            y_node_tensor = torch.zeros((x_tensor.size(0), 1), device=self.device, dtype=x_tensor.dtype)

        self.eval()
        with torch.no_grad():
            preds = None
            for _ in range(sar_iterations):
                preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor, y_node=y_node_tensor)
                if self.use_sar:
                    y_node_tensor = preds.unsqueeze(-1)
        return preds.cpu().numpy()
