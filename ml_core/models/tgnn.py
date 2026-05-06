import math
from copy import deepcopy
from itertools import product
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
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


class TGNN(BaseModel, nn.Module):
    """End-to-end regression model using stacked GraphAttentionLayer blocks.

    El modelo no construye grafos por su cuenta. Depende de un `graph_builder`
    externo que debe encargarse de:
    - preparar el grafo de train,
    - construir el cross-graph train->target,
    - y mantener el scaler de edge_attr.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        edge_dim: int,
        graph_builder: Any,
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
    ):
        BaseModel.__init__(self)
        nn.Module.__init__(self)

        self.feature_names_ = list(feature_names)
        self.edge_dim = edge_dim
        self.graph_builder_template = deepcopy(graph_builder)
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
        if self.graph_builder_template is None:
            raise ValueError("TGNN requiere un graph_builder explícito.")

        self._build_network()

        # runtime state
        self.X_train_ = None
        self.y_train_ = None
        self.edge_index_ = None
        self.edge_attr_ = None
        self.tuning_results_ = None
        self._graph_state_ = None
        self.graph_builder_ = None
        self.prepared_train_graph_ = None
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
            "smape": TGNN._smape(y_true_arr, y_pred_arr),
        }

    @staticmethod
    def _sort_metric_value(
        metric_name: str,
        metric_value: float,
        *,
        optimize_metric: str,
        maximize_metric: bool,
    ) -> float:
        if metric_name == optimize_metric and maximize_metric:
            return -float(metric_value)
        return float(metric_value)

    @classmethod
    def _results_sort_key(
        cls,
        row: dict,
        *,
        sort_by: Sequence[str],
        optimize_metric: str,
        maximize_metric: bool,
    ) -> tuple:
        return tuple(
            cls._sort_metric_value(
                metric_name=metric_name,
                metric_value=row[metric_name],
                optimize_metric=optimize_metric,
                maximize_metric=maximize_metric,
            )
            for metric_name in sort_by
        )

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
    def _feature_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names_ if col not in X.columns]
            if missing:
                raise ValueError(
                    "X no contiene todas las columnas de feature_names. "
                    f"Faltan: {missing}."
                )
            return X.loc[:, self.feature_names_].copy()

        x_array = np.asarray(X)
        if x_array.ndim != 2:
            raise ValueError("X debe ser un DataFrame o un array 2D.")
        if x_array.shape[1] != len(self.feature_names_):
            raise ValueError(
                "X no coincide con feature_names del modelo. "
                f"Esperadas {len(self.feature_names_)} columnas y llegaron {x_array.shape[1]}."
            )
        return pd.DataFrame(x_array, columns=self.feature_names_)

    def _build_node_feature_matrix(self, X) -> np.ndarray:
        return self._feature_dataframe(X).to_numpy(dtype=np.float32, copy=True)

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

        self._validate_edge_arrays(
            edge_index,
            edge_attr,
            num_nodes=num_nodes,
        )
        if num_nodes == 0:
            raise ValueError("X no puede estar vacío")

        x_array = self._build_node_feature_matrix(X)
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

    @staticmethod
    def _concat_coords(coords_left, coords_right):
        return np.concatenate(
            [
                np.asarray(coords_left, dtype=float),
                np.asarray(coords_right, dtype=float),
            ],
            axis=0,
        )

    def _validate_edge_arrays(
        self,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        *,
        num_nodes: Optional[int] = None,
    ) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index debe tener shape (2, E)")
        if edge_attr.ndim != 2:
            raise ValueError("edge_attr debe tener shape (E, edge_dim)")
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError(
                "edge_index y edge_attr deben tener la misma cantidad de aristas. "
                f"Recibido: {edge_index.shape[1]} y {edge_attr.shape[0]}."
            )
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                "edge_attr no coincide con edge_dim del modelo. "
                f"Recibido edge_attr.shape[1]={edge_attr.shape[1]} y "
                f"edge_dim={self.edge_dim}."
            )
        if num_nodes is None or edge_index.shape[1] == 0:
            return

        min_idx = int(edge_index.min())
        max_idx = int(edge_index.max())
        if min_idx < 0 or max_idx >= num_nodes:
            raise ValueError(
                "edge_index contiene nodos fuera de rango para el X provisto. "
                f"Rango válido: [0, {num_nodes - 1}], recibido: [{min_idx}, {max_idx}]. "
                "Si estás usando un grafo cross-split (train+target), pasá el dataframe "
                "combinado correspondiente al hacer predict/tuning."
            )

    def _instantiate_graph_builder(self):
        return deepcopy(self.graph_builder_template)

    def _transform_edge_attr_with_graph_builder(self, edge_attr: np.ndarray) -> np.ndarray:
        edge_attr_arr = np.asarray(edge_attr, dtype=float)
        if edge_attr_arr.ndim != 2:
            raise ValueError(
                "edge_attr debe tener shape (E, edge_dim) para transformar aristas. "
                f"Recibido: {edge_attr_arr.shape}."
            )
        if edge_attr_arr.shape[0] == 0:
            return edge_attr_arr.astype(np.float32, copy=False)
        if self.graph_builder_ is None:
            raise ValueError(
                "No hay graph_builder ajustado para transformar edge_attr."
            )
        if hasattr(self.graph_builder_, "transform_edge_attr"):
            return self.graph_builder_.transform_edge_attr(edge_attr_arr)
        return edge_attr_arr.astype(np.float32, copy=False)

    def _is_training_input(self, X) -> bool:
        if self.X_train_ is None:
            return False
        candidate = self._feature_dataframe(X)
        if candidate is self.X_train_:
            return True
        if isinstance(candidate, pd.DataFrame) and isinstance(self.X_train_, pd.DataFrame):
            try:
                return bool(candidate.equals(self.X_train_))
            except Exception:
                return False
        try:
            return bool(
                np.array_equal(
                    candidate.to_numpy(copy=False),
                    self.X_train_.to_numpy(copy=False),
                )
            )
        except Exception:
            return False

    def _cache_graph_state(self, X, coords) -> None:
        train_features = self._feature_dataframe(X)
        graph_builder = self._instantiate_graph_builder()
        prepared_train_graph = graph_builder.prepare(
            train_features,
            coords=coords,
            expected_feature_cols=list(self.feature_names_),
        )
        edge_index_train, edge_attr_train = prepared_train_graph.build_graph(
            fit_edge_scaler=True,
            scale_edge_attr=True,
        )

        self.graph_builder_ = graph_builder
        self.prepared_train_graph_ = prepared_train_graph
        self._graph_state_ = {
            "graph_builder": graph_builder,
            "prepared_train_graph": prepared_train_graph,
            "edge_attr_train": edge_attr_train,
            "edge_index_train": edge_index_train,
        }
        self.X_train_ = train_features

    def _build_cross_graph_for_predict(self, X_target, coords_target):
        if self._graph_state_ is None or self.X_train_ is None:
            raise ValueError(
                "No hay estado de grafo cacheado de train. Entrená el modelo antes de predecir."
            )
        target_features = self._feature_dataframe(X_target)
        target_prepared_graph = self._graph_state_["graph_builder"].prepare(
            target_features,
            coords=coords_target,
            expected_feature_cols=list(self.feature_names_),
        )
        combined_features, edge_index, edge_attr, target_mask = self.prepared_train_graph_.build_cross_graph(
            target_prepared_graph,
            fit_edge_scaler_on_source=False,
            scale_edge_attr=True,
        )
        return combined_features, edge_index, edge_attr, target_mask

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
        epochs: int = 200,
    ):
        if coords is None:
            raise ValueError(
                "TGNN requiere `coords` en fit() para que el graph_builder "
                "pueda construir el grafo de train."
            )

        self._cache_graph_state(X, coords)
        X_features = self.X_train_
        edge_index = self._graph_state_["edge_index_train"]
        edge_attr = self._graph_state_["edge_attr_train"]

        self.train()
        x_tensor, y_tensor, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X_features,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        self.edge_index_ = edge_index
        self.edge_attr_ = edge_attr

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = self._build_loss_fn()

        best_loss = float("inf")
        best_state_dict = deepcopy(self.state_dict())
        best_epoch = 0
        patience_left = self.patience
        self.history_ = {
            "epoch": [],
            "loss": [],
            "best_loss": [],
            "patience_left": [],
            "best_epoch": [],
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
                best_state_dict = deepcopy(self.state_dict())
                best_epoch = epoch + 1
                patience_left = self.patience
            else:
                patience_left -= 1

            self.history_["epoch"].append(epoch + 1)
            self.history_["loss"].append(float(current_loss))
            self.history_["best_loss"].append(float(best_loss))
            self.history_["patience_left"].append(int(patience_left))
            self.history_["best_epoch"].append(int(best_epoch))

            if patience_left <= 0:
                print(f"Early stopping en epoch {epoch + 1}, best loss={best_loss:.6f}")
                break

        self.load_state_dict(best_state_dict)
        self.best_loss_ = float(best_loss)
        self.best_epoch_ = int(best_epoch)
        self.is_fitted_ = True
        self.y_train_ = np.asarray(y)
        return self

    def tune_hyperparameters(
        self,
        X,
        y,
        coords,
        X_val=None,
        y_val=None,
        coords_val=None,
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
                "TGNN requiere `coords` en tune_hyperparameters(). "
                "El tuning necesita mantener un flujo espacial coherente entre "
                "fit() y predict()."
            )

        if X_val is None:
            X_val = X
        if y_val is None:
            y_val = y
        if X_val is not X and coords_val is None:
            raise ValueError(
                "Si X_val es distinto de X, pasá coords_val para que el "
                "graph_builder pueda construir el cross-graph de validación."
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
                candidate = TGNN(
                    feature_names=self.feature_names_,
                    edge_dim=self.edge_dim,
                    graph_builder=deepcopy(self.graph_builder_template),
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
                )

                candidate.fit(
                    X,
                    y,
                    coords,
                    epochs=epochs,
                )

                if X_val is X:
                    preds = candidate.predict(X_val)
                elif coords_val is not None:
                    preds = candidate.predict(X_val, coords_val)
                else:
                    raise ValueError(
                        "coords_val es obligatorio cuando X_val es distinto de X."
                    )
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
                current_key = self._results_sort_key(
                    result,
                    sort_by=sort_by,
                    optimize_metric=optimize_metric,
                    maximize_metric=maximize_metric,
                )
                best_key = self._results_sort_key(
                    best_result,
                    sort_by=sort_by,
                    optimize_metric=optimize_metric,
                    maximize_metric=maximize_metric,
                )
                is_better = current_key < best_key

            if is_better:
                best_result = result
                best_config = candidate_cfg.copy()

        if not results:
            raise ValueError("No se encontraron configuraciones válidas durante el tuning.")

        self.tuning_results_ = sorted(
            results,
            key=lambda row: self._results_sort_key(
                row,
                sort_by=sort_by,
                optimize_metric=optimize_metric,
                maximize_metric=maximize_metric,
            ),
        )
        self.best_params_ = best_config.copy()

        if refit:
            self._apply_config(best_config)
            self._set_random_state(random_state)
            self.fit(
                X,
                y,
                coords,
                epochs=refit_epochs or epochs,
            )

        return self

    def predict(
        self,
        X,
        coords=None,
    ):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        target_mask = None
        X_eval = X
        if coords is not None:
            X_eval, edge_index, edge_attr, target_mask = self._build_cross_graph_for_predict(
                X_target=X,
                coords_target=coords,
            )
        elif self._is_training_input(X):
            X_eval = self.X_train_
            edge_index = self._graph_state_["edge_index_train"]
            edge_attr = self._graph_state_["edge_attr_train"]
        else:
            raise ValueError(
                "predict() requiere `coords` salvo que estés prediciendo "
                "exactamente sobre el set de train."
            )

        x_tensor, _, edge_index_tensor, edge_attr_tensor = self._prepare_tensors(
            X_eval,
            y=None,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        self.eval()
        with torch.no_grad():
            preds = self.forward(x_tensor, edge_index_tensor, edge_attr_tensor)
        preds_np = preds.cpu().numpy()
        if target_mask is not None:
            preds_np = preds_np[np.asarray(target_mask)]
        return preds_np


__all__ = ["TGNN", "GraphAttentionLayer"]
