import math
from copy import deepcopy
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.neighbors import BallTree

from ..preprocessing.knhs import KNHS
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

        # Final aggregation matches the paper: linear_aggr([h_i, h_hat_i])
        self.linear_aggr = nn.Linear(in_dim + self.out_dim, self.out_dim)

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

        # Attention logits following the paper: Q_i · (K_j + U_ij)
        att_logits = (q_dst * (k_src + u)).sum(dim=-1) / math.sqrt(self.head_dim)

        alphas = _SegmentOps.softmax_per_dst(att_logits, dst, x.size(0))  # [E, H]

        # Weighted aggregation of value + edge context per cabeza
        messages = alphas.unsqueeze(-1) * (v_src + u)
        agg = _SegmentOps.scatter_sum(messages.view(edge_attr.size(0), -1), dst, x.size(0))  # [N, H*Dh]
        agg = agg.view(x.size(0), self.num_heads, self.head_dim)

        agg_flat = agg.view(x.size(0), self.out_dim)
        h_next = self.linear_aggr(torch.cat([x, agg_flat], dim=-1))
        h_next = self.activation(self.dropout(h_next))
        return h_next


class GraphAttentionGCN(BaseModel, nn.Module):
    """End-to-end regression model using stacked GraphAttentionLayer blocks.

    El flujo esperado del modelo es espacial: `coords` debe pasarse en `fit()`
    para que el modelo pueda cachear el estado del grafo de train y luego
    construir grafos cross-split en `predict()`. Si además se proveen
    `edge_index` y `edge_attr`, esos grafos manuales tienen prioridad. Cuando
    se pasa `knhs_builder`, ese objeto se usa como plantilla para construir y
    fittear internamente el grafo de train, incluyendo el scaler de aristas.
    """

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
        loss_name: str = "mse",
        huber_delta: float = 1.0,
        grad_clip_norm: Optional[float] = None,
        k_neighbors: int = 15,
        radius_km: float = 3.0,
        bandwidth_km: float = 2.0,
        graph_distance: str = "euclidean",
        add_reverse_edges: bool = True,
        coord_feature_names: Sequence[str] = ("longitud", "latitud"),
        knhs_builder: Optional[KNHS] = None,
    ):
        BaseModel.__init__(self)
        nn.Module.__init__(self)

        self.feature_names_ = list(feature_names)
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.patience = patience
        self.min_delta = min_delta
        self.loss_name = loss_name
        self.huber_delta = huber_delta
        self.grad_clip_norm = grad_clip_norm
        self.k_neighbors = k_neighbors
        self.radius_km = radius_km
        self.bandwidth_km = bandwidth_km
        self.graph_distance = graph_distance
        self.add_reverse_edges = add_reverse_edges
        self.coord_feature_names = tuple(coord_feature_names)
        self.weight_cols_ = [f"w_{col}" for col in self.feature_names_]
        self.knhs_builder_template = deepcopy(knhs_builder) if knhs_builder is not None else None
        if self.knhs_builder_template is not None:
            self.k_neighbors = int(self.knhs_builder_template.k)
            self.radius_km = float(self.knhs_builder_template.radius_km)
            self.graph_distance = str(self.knhs_builder_template.distance)
            self.add_reverse_edges = bool(self.knhs_builder_template.add_reverse)

        self._build_network()

        # cache used during predict
        self.edge_index_ = None
        self.edge_attr_ = None
        self.tuning_results_ = None
        self.coords_train_ = None
        self._graph_state_ = None
        self.knhs_builder_ = None
        self.history_ = None

    @staticmethod
    def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(
            100
            * np.mean(
                2 * np.abs(y_pred - y_true)
                / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
            )
        )

    @staticmethod
    def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true_arr = np.asarray(y_true).reshape(-1)
        y_pred_arr = np.asarray(y_pred).reshape(-1)
        mape = float(
            100
            * np.mean(
                np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), 1e-8, None))
            )
        )
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "mape": mape,
            "median_ae": float(median_absolute_error(y_true_arr, y_pred_arr)),
            "r2": float(r2_score(y_true_arr, y_pred_arr)),
            "smape": GraphAttentionGCN._smape(y_true_arr, y_pred_arr),
        }

    def _build_network(self) -> None:
        layers = []
        in_dim = len(self.feature_names_)
        for _ in range(self.num_layers):
            layers.append(
                GraphAttentionLayer(
                    in_dim,
                    self.edge_dim,
                    self.hidden,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
            in_dim = self.hidden
        self.layers = nn.ModuleList(layers)

        self.readout = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

        self.to(self.device)

    def _apply_config(self, config: dict) -> None:
        self.hidden = int(config.get("hidden", self.hidden))
        self.num_layers = int(config.get("num_layers", self.num_layers))
        self.num_heads = int(config.get("num_heads", self.num_heads))
        self.dropout = float(config.get("dropout", self.dropout))
        self.lr = float(config.get("lr", self.lr))
        self.weight_decay = float(config.get("weight_decay", self.weight_decay))
        self.loss_name = str(config.get("loss_name", self.loss_name))
        self.huber_delta = float(config.get("huber_delta", self.huber_delta))
        grad_clip_norm = config.get("grad_clip_norm", self.grad_clip_norm)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.k_neighbors = int(config.get("k_neighbors", self.k_neighbors))
        self.radius_km = float(config.get("radius_km", self.radius_km))
        self.bandwidth_km = float(config.get("bandwidth_km", self.bandwidth_km))
        self._build_network()

    @staticmethod
    def _set_random_state(random_state: Optional[int]) -> None:
        if random_state is None:
            return
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _build_loss_fn(self) -> nn.Module:
        if self.loss_name == "mse":
            return nn.MSELoss()
        if self.loss_name == "huber":
            return nn.HuberLoss(delta=self.huber_delta)
        if self.loss_name == "smooth_l1":
            return nn.SmoothL1Loss(beta=self.huber_delta)
        raise ValueError(
            "loss_name debe ser 'mse', 'huber' o 'smooth_l1'. "
            f"Recibido: {self.loss_name!r}"
        )

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

        edge_index = np.asarray(edge_index)
        edge_attr = np.asarray(edge_attr)
        num_nodes = len(X)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index debe tener shape (2, E)")
        if edge_attr.ndim != 2:
            raise ValueError("edge_attr debe tener shape (E, edge_dim)")
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError(
                "edge_index y edge_attr deben tener la misma cantidad de aristas. "
                f"Recibido: {edge_index.shape[1]} y {edge_attr.shape[0]}."
            )
        if num_nodes == 0:
            raise ValueError("X no puede estar vacío")
        if edge_index.shape[1] > 0:
            min_idx = int(edge_index.min())
            max_idx = int(edge_index.max())
            if min_idx < 0 or max_idx >= num_nodes:
                raise ValueError(
                    "edge_index contiene nodos fuera de rango para el X provisto. "
                    f"Rango válido: [0, {num_nodes - 1}], recibido: [{min_idx}, {max_idx}]. "
                    "Si estás usando un grafo cross-split (train+target), pasá el dataframe "
                    "combinado correspondiente al hacer predict/tuning."
                )

        x_array = X[self.feature_names_].to_numpy(dtype=np.float32, copy=False)
        x_tensor = torch.as_tensor(x_array, dtype=torch.float32, device=self.device)
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)

        y_tensor = None
        if y is not None:
            if len(y) != num_nodes:
                raise ValueError(
                    "y y X deben tener la misma cantidad de filas. "
                    f"Recibido: len(X)={num_nodes}, len(y)={len(y)}."
                )
            y_tensor = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32, device=self.device)

        return x_tensor, y_tensor, edge_index_tensor, edge_attr_tensor

    @staticmethod
    def _concat_frames(X_left, X_right):
        if isinstance(X_left, pd.DataFrame) and isinstance(X_right, pd.DataFrame):
            return pd.concat([X_left.reset_index(drop=True), X_right.reset_index(drop=True)], ignore_index=True)
        return np.concatenate([np.asarray(X_left), np.asarray(X_right)], axis=0)

    def _coords_to_latlon_radians(self, coords) -> np.ndarray:
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(
                "coords debe tener shape (n, 2). "
                f"Recibido: {coords_arr.shape}."
            )

        names = tuple(name.lower() for name in self.coord_feature_names)
        if len(names) == 2 and "long" in names[0] and "lat" in names[1]:
            coords_arr = coords_arr[:, [1, 0]]

        if np.nanmax(np.abs(coords_arr)) > math.pi + 1e-6:
            coords_arr = np.deg2rad(coords_arr)

        return coords_arr

    def _compute_local_feature_weights(
        self,
        X_df,
        y_array,
        coords_latlon_rad,
    ) -> np.ndarray:
        X_arr = np.asarray(X_df[self.feature_names_].to_numpy(), dtype=float)
        y_arr = np.asarray(y_array, dtype=float).reshape(-1)
        n_rows, n_features = X_arr.shape
        tree = BallTree(coords_latlon_rad, metric="haversine")
        weights = np.zeros((n_rows, n_features), dtype=float)

        for i in range(n_rows):
            dist, idx = tree.query(
                coords_latlon_rad[i:i + 1],
                k=min(self.k_neighbors + 1, n_rows),
            )
            dist = dist.ravel() * 6371.0
            idx = idx.ravel()
            mask = dist > 0
            dist, idx = dist[mask], idx[mask]
            if len(idx) == 0:
                weights[i] = 1.0
                continue

            kernel = np.exp(-(dist ** 2) / (self.bandwidth_km ** 2))
            X_neighbors = X_arr[idx]
            A = X_neighbors * kernel[:, None]
            AtY = A.T @ y_arr[idx]

            beta = None
            ridge = 1e-6
            for _ in range(6):
                AtX = A.T @ X_neighbors + np.eye(n_features) * ridge
                try:
                    beta = np.linalg.solve(AtX, AtY)
                    break
                except np.linalg.LinAlgError:
                    ridge *= 10.0

            if beta is None:
                AtX = A.T @ X_neighbors + np.eye(n_features) * ridge
                beta = np.linalg.pinv(AtX) @ AtY

            beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
            weights[i] = np.maximum(np.abs(beta), 1e-8)

        return weights

    def _project_local_feature_weights(
        self,
        weights_source: np.ndarray,
        coords_source_latlon_rad: np.ndarray,
        coords_target_latlon_rad: np.ndarray,
    ) -> np.ndarray:
        tree = BallTree(coords_source_latlon_rad, metric="haversine")
        k_proj = min(self.k_neighbors, len(coords_source_latlon_rad))
        dist_proj, idx_proj = tree.query(coords_target_latlon_rad, k=k_proj)
        kernel_proj = np.exp(-((dist_proj * 6371.0) ** 2) / (self.bandwidth_km ** 2))
        kernel_proj = kernel_proj / (kernel_proj.sum(axis=1, keepdims=True) + 1e-9)
        return (kernel_proj[..., None] * weights_source[idx_proj]).sum(axis=1)

    def _build_weighted_graph_frame(
        self,
        X_df,
        coords_latlon_rad: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        graph = X_df[self.feature_names_].copy().reset_index(drop=True)
        graph["lat_deg"] = np.rad2deg(coords_latlon_rad[:, 0])
        graph["lon_deg"] = np.rad2deg(coords_latlon_rad[:, 1])
        if weights is not None:
            for col, vals in zip(self.weight_cols_, weights.T):
                graph[col] = vals
        return graph

    def _build_knhs_builder(self) -> KNHS:
        if self.knhs_builder_template is None:
            builder = KNHS(
                lat_col="lat_deg",
                lon_col="lon_deg",
                feature_cols=self.feature_names_,
                weight_cols=self.weight_cols_ if self.graph_distance == "local_weighted" else None,
                distance=self.graph_distance,
                radius_km=self.radius_km,
                k=self.k_neighbors,
                add_reverse=self.add_reverse_edges,
            )
        else:
            builder = deepcopy(self.knhs_builder_template)
            builder.lat_col = "lat_deg"
            builder.lon_col = "lon_deg"
            builder.feature_cols = list(self.feature_names_)
            builder.weight_cols = self.weight_cols_ if self.graph_distance == "local_weighted" else None
            builder.distance = self.graph_distance
            builder.radius_km = self.radius_km
            builder.k = self.k_neighbors
            builder.add_reverse = self.add_reverse_edges
            if builder.scale_edge_features and builder.edge_scaler_ is not None:
                builder.edge_scaler_ = deepcopy(builder.edge_scaler_)
            builder.edge_scaler_fitted_ = False
        return builder

    def _cache_graph_state(self, X, y, coords) -> None:
        coords_train_latlon_rad = self._coords_to_latlon_radians(coords)
        weights_train = None
        if self.graph_distance == "local_weighted":
            weights_train = self._compute_local_feature_weights(
                X,
                y,
                coords_train_latlon_rad,
            )
        graph_train = self._build_weighted_graph_frame(
            X,
            coords_train_latlon_rad,
            weights_train,
        )
        builder = self._build_knhs_builder()
        # Ajustamos el scaler de aristas solo con train para reutilizarlo en predict().
        builder.build(graph_train, fit_edge_scaler=True)
        self.knhs_builder_ = builder
        self._graph_state_ = {
            "coords_train_latlon_rad": coords_train_latlon_rad,
            "weights_train": weights_train,
            "graph_train": graph_train,
            "builder": builder,
        }

    def _build_cross_graph_for_predict(self, X_target, coords_target):
        if self._graph_state_ is None or self.X_train_ is None or self.y_train_ is None:
            raise ValueError(
                "No hay estado de grafo cacheado de train. "
                "Entrena el modelo con coords o pasa edge_index/edge_attr explicitamente."
            )

        coords_target_latlon_rad = self._coords_to_latlon_radians(coords_target)
        weights_target = None
        if self.graph_distance == "local_weighted":
            weights_target = self._project_local_feature_weights(
                self._graph_state_["weights_train"],
                self._graph_state_["coords_train_latlon_rad"],
                coords_target_latlon_rad,
            )
        graph_target = self._build_weighted_graph_frame(
            X_target,
            coords_target_latlon_rad,
            weights_target,
        )
        X_eval = self._concat_frames(self.X_train_, X_target)
        _, edge_index, edge_attr, target_mask = self._graph_state_["builder"].build_cross_split(
            self._graph_state_["graph_train"],
            graph_target,
        )
        return X_eval, edge_index, edge_attr, target_mask

    def _build_graph_for_fit(self, X, y, coords):
        coords_latlon_rad = self._coords_to_latlon_radians(coords)
        weights = None
        if self.graph_distance == "local_weighted":
            weights = self._compute_local_feature_weights(
                X,
                y,
                coords_latlon_rad,
            )
        graph = self._build_weighted_graph_frame(
            X,
            coords_latlon_rad,
            weights,
        )
        builder = self._build_knhs_builder()
        edge_index, edge_attr = builder.build(graph, fit_edge_scaler=True)
        return edge_index, edge_attr

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
        coords,
        *,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
        epochs: int = 200,
    ):
        if coords is None:
            raise ValueError(
                "GraphAttentionGCN requiere `coords` en fit(). "
                "El modelo usa las coordenadas para cachear el estado del "
                "grafo de train y construir grafos cross-split en predict()."
            )

        if (edge_index is None) != (edge_attr is None):
            raise ValueError(
                "edge_index y edge_attr deben pasarse juntos o ambos omitirse."
            )

        self._cache_graph_state(X, y, coords)
        graph_builder = self._graph_state_["builder"]
        graph_train = self._graph_state_["graph_train"]

        if edge_index is None and edge_attr is None:
            edge_index, edge_attr = graph_builder.build(graph_train)
        else:
            edge_attr = graph_builder.transform_edge_attr(edge_attr)

        self.train()
        x_tensor, y_tensor, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X, y=y, edge_index=edge_index, edge_attr=edge_attr
        )

        self.edge_index_ = edge_index
        self.edge_attr_ = edge_attr

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = self._build_loss_fn()

        best_loss = float("inf")
        patience_left = self.patience
        self.history_ = {
            "epoch": [],
            "loss": [],
            "best_loss": [],
            "patience_left": [],
        }

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
            loss = loss_fn(preds.view(-1, 1), y_tensor)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
            optimizer.step()

            current_loss = loss.item()
            if current_loss + self.min_delta < best_loss:
                best_loss = current_loss
                patience_left = self.patience
            else:
                patience_left -= 1

            self.history_["epoch"].append(epoch + 1)
            self.history_["loss"].append(float(current_loss))
            self.history_["best_loss"].append(float(best_loss))
            self.history_["patience_left"].append(int(patience_left))

            if patience_left <= 0:
                print(f"Early stopping en epoch {epoch + 1}, best loss={best_loss:.6f}")
                break

        self.is_fitted_ = True
        self.X_train_ = X
        self.y_train_ = np.asarray(y)
        self.coords_train_ = np.asarray(coords)
        return self

    def tune_hyperparameters(
        self,
        X,
        y,
        coords,
        *,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
        X_val=None,
        y_val=None,
        coords_val=None,
        val_edge_index: Optional[np.ndarray] = None,
        val_edge_attr: Optional[np.ndarray] = None,
        param_grid: Optional[dict] = None,
        search_type: str = "grid",
        n_iter: Optional[int] = None,
        epochs: int = 600,
        refit: bool = True,
        refit_epochs: Optional[int] = None,
        optimize_metric: str = "mae",
        maximize_metric: bool = False,
        sort_by: Tuple[str, ...] = ("mae", "rmse"),
        eval_on_exp_scale: bool = True,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        if param_grid is None:
            param_grid = {
                "hidden": [64, 96, 128],
                "num_heads": [2, 4],
                "dropout": [0.05, 0.10],
                "lr": [1e-3, 5e-4],
            }

        if coords is None:
            raise ValueError(
                "GraphAttentionGCN requiere `coords` en tune_hyperparameters(). "
                "El tuning necesita mantener un flujo espacial coherente entre "
                "fit() y predict()."
            )

        if (edge_index is None) != (edge_attr is None):
            raise ValueError(
                "edge_index y edge_attr deben pasarse juntos o ambos omitirse."
            )

        train_graph_is_explicit = edge_index is not None and edge_attr is not None

        if X_val is None:
            X_val = X
        if y_val is None:
            y_val = y
        if coords_val is None and X_val is X:
            coords_val = coords
        if (val_edge_index is None) != (val_edge_attr is None):
            raise ValueError(
                "val_edge_index y val_edge_attr deben pasarse juntos o ambos omitirse."
            )
        if (
            val_edge_index is None
            and val_edge_attr is None
            and X_val is not X
            and coords_val is None
        ):
            raise ValueError(
                "Si X_val es distinto de X, pasá coords_val o val_edge_index/val_edge_attr "
                "para que el modelo pueda evaluar sobre el split de validación."
            )

        grid_keys = list(param_grid.keys())
        grid_values = [param_grid[key] for key in grid_keys]
        valid_metric_names = {"rmse", "mae", "mape", "median_ae", "r2", "smape"}
        valid_search_types = {"grid", "random"}

        if optimize_metric not in valid_metric_names:
            raise ValueError(
                f"optimize_metric debe ser una de {sorted(valid_metric_names)}. "
                f"Recibido: {optimize_metric!r}"
            )
        if search_type not in valid_search_types:
            raise ValueError(
                f"search_type debe ser uno de {sorted(valid_search_types)}. "
                f"Recibido: {search_type!r}"
            )

        results = []
        best_result = None
        best_config = None

        all_configs = [dict(zip(grid_keys, values)) for values in product(*grid_values)]
        if search_type == "random":
            if n_iter is None:
                raise ValueError("Si search_type='random', tenés que pasar n_iter.")
            if n_iter <= 0:
                raise ValueError(f"n_iter debe ser mayor a 0. Recibido: {n_iter}")

            rng = np.random.default_rng(random_state)
            sample_size = min(int(n_iter), len(all_configs))
            sampled_idx = rng.choice(len(all_configs), size=sample_size, replace=False)
            configs_to_run = [all_configs[int(i)] for i in sampled_idx]
        else:
            configs_to_run = all_configs

        if verbose:
            print(
                f"Ejecutando {len(configs_to_run)} configuraciones "
                f"(search_type={search_type}, total_grid={len(all_configs)})."
            )

        for cfg in configs_to_run:
            hidden = int(cfg.get("hidden", self.hidden))
            num_heads = int(cfg.get("num_heads", self.num_heads))
            if hidden % num_heads != 0:
                if verbose:
                    print(f"Saltando config inválida {cfg}: hidden debe ser divisible por num_heads.")
                continue

            run_seed = random_state if random_state is None else int(random_state)
            self._set_random_state(run_seed)

            candidate_cfg = {
                "hidden": hidden,
                "num_layers": int(cfg.get("num_layers", self.num_layers)),
                "num_heads": num_heads,
                "dropout": float(cfg.get("dropout", self.dropout)),
                "lr": float(cfg.get("lr", self.lr)),
                "weight_decay": float(cfg.get("weight_decay", self.weight_decay)),
                "loss_name": str(cfg.get("loss_name", self.loss_name)),
                "huber_delta": float(cfg.get("huber_delta", self.huber_delta)),
                "grad_clip_norm": cfg.get("grad_clip_norm", self.grad_clip_norm),
            }

            if verbose:
                print(f"Entrenando config {candidate_cfg}")

            candidate = None
            try:
                candidate = GraphAttentionGCN(
                    feature_names=self.feature_names_,
                    edge_dim=self.edge_dim,
                    hidden=candidate_cfg["hidden"],
                    num_layers=candidate_cfg["num_layers"],
                    num_heads=candidate_cfg["num_heads"],
                    dropout=candidate_cfg["dropout"],
                    lr=candidate_cfg["lr"],
                    weight_decay=candidate_cfg["weight_decay"],
                    device=str(self.device),
                    patience=self.patience,
                    min_delta=self.min_delta,
                    loss_name=candidate_cfg["loss_name"],
                    huber_delta=candidate_cfg["huber_delta"],
                    grad_clip_norm=candidate_cfg["grad_clip_norm"],
                    k_neighbors=self.k_neighbors,
                    radius_km=self.radius_km,
                    bandwidth_km=self.bandwidth_km,
                    graph_distance=self.graph_distance,
                    add_reverse_edges=self.add_reverse_edges,
                    coord_feature_names=self.coord_feature_names,
                    knhs_builder=self.knhs_builder_template,
                )

                fit_kwargs = {
                    "epochs": epochs,
                }
                if train_graph_is_explicit:
                    fit_kwargs["edge_index"] = edge_index
                    fit_kwargs["edge_attr"] = edge_attr

                candidate.fit(
                    X,
                    y,
                    coords,
                    **fit_kwargs,
                )

                predict_kwargs = {}
                X_val_eval = X_val
                val_target_slice = slice(None)

                if val_edge_index is not None and val_edge_attr is not None:
                    val_edge_index_arr = np.asarray(val_edge_index)
                    val_uses_combined_graph = (
                        val_edge_index_arr.size > 0 and int(val_edge_index_arr.max()) >= len(X_val)
                    )
                    if val_uses_combined_graph:
                        expected_nodes = len(X) + len(X_val)
                        max_idx = int(val_edge_index_arr.max())
                        if max_idx >= expected_nodes:
                            raise ValueError(
                                "val_edge_index referencia más nodos que los disponibles en train+val. "
                                f"Máximo índice recibido: {max_idx}, nodos esperados: {expected_nodes}."
                            )
                        X_val_eval = self._concat_frames(X, X_val)
                        val_target_slice = slice(len(X), expected_nodes)
                        if verbose:
                            print(
                                "Detectado grafo de validación cross-split explícito; la evaluación usará "
                                "el grafo combinado train+val y luego recortará solo los nodos target."
                            )

                    predict_kwargs["edge_index"] = val_edge_index
                    predict_kwargs["edge_attr"] = val_edge_attr
                elif coords_val is not None:
                    predict_kwargs["coords"] = coords_val

                preds = candidate.predict(
                    X_val_eval,
                    **predict_kwargs,
                )
                preds = np.asarray(preds).reshape(-1)[val_target_slice]
            finally:
                if candidate is not None:
                    del candidate
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            y_true_eval = np.asarray(y_val).reshape(-1)
            y_pred_eval = np.asarray(preds).reshape(-1)
            if eval_on_exp_scale:
                y_true_eval = np.exp(y_true_eval)
                y_pred_eval = np.exp(y_pred_eval)

            metrics = self._metrics_dict(y_true_eval, y_pred_eval)
            result = {**candidate_cfg, **metrics}
            results.append(result)

            if best_result is None:
                best_result = result
                best_config = candidate_cfg.copy()
                continue

            current_metric = result[optimize_metric]
            best_metric = best_result[optimize_metric]
            is_better = current_metric > best_metric if maximize_metric else current_metric < best_metric

            if not is_better and current_metric == best_metric:
                current_key = tuple(result[key] for key in sort_by)
                best_key = tuple(best_result[key] for key in sort_by)
                is_better = current_key < best_key

            if is_better:
                best_result = result
                best_config = candidate_cfg.copy()

        if not results:
            raise ValueError("No se encontraron configuraciones válidas durante el tuning.")

        self.tuning_results_ = sorted(results, key=lambda row: tuple(row[key] for key in sort_by))
        self.best_params_ = best_config.copy()

        if refit:
            self._apply_config(best_config)
            self._set_random_state(random_state)
            fit_kwargs = {
                "epochs": refit_epochs or epochs,
            }
            if train_graph_is_explicit:
                fit_kwargs["edge_index"] = edge_index
                fit_kwargs["edge_attr"] = edge_attr

            self.fit(
                X,
                y,
                coords,
                **fit_kwargs,
            )

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

        target_mask = None
        X_eval = X

        if edge_index is None or edge_attr is None:
            if coords is not None:
                if self._graph_state_ is None:
                    raise ValueError(
                        "predict() recibio coords pero el modelo no tiene estado "
                        "de grafo cacheado de train. Reentrena pasando coords a "
                        "fit(), o pasa edge_index/edge_attr explicitamente."
                    )
                X_eval, edge_index, edge_attr, target_mask = self._build_cross_graph_for_predict(
                    X_target=X,
                    coords_target=coords,
                )
            else:
                x_tensor, _, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
                    X, y=None, edge_index=edge_index, edge_attr=edge_attr
                )
                self.eval()
                with torch.no_grad():
                    preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
                return preds.cpu().numpy()
        elif self._graph_state_ is not None:
            edge_attr = self._graph_state_["builder"].transform_edge_attr(edge_attr)

        x_tensor, _, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X_eval, y=None, edge_index=edge_index, edge_attr=edge_attr
        )

        self.eval()
        with torch.no_grad():
            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
        preds_np = preds.cpu().numpy()
        if target_mask is not None:
            preds_np = preds_np[np.asarray(target_mask)]
        return preds_np


__all__ = ["GraphAttentionGCN", "GraphAttentionLayer"]
