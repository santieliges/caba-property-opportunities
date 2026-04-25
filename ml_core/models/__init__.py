from .baseModel import BaseModel
from .gwrmodel import GWRModel
from .rfrkModel import RegressionKrigingModel
from .sarModel import SpatialAutoregressiveModel

__all__ = [
    "BaseModel",
    "GWRModel",
    "RegressionKrigingModel",
    "SpatialAutoregressiveModel",
]


def _optional_imports() -> None:
    try:
        from .gnn_price_model import RentalPriceGNN
    except Exception:
        pass
    else:
        globals()["RentalPriceGNN"] = RentalPriceGNN
        __all__.append("RentalPriceGNN")

    try:
        from .gat_gcn_model import GraphAttentionGCN, GraphAttentionLayer
    except Exception:
        pass
    else:
        globals()["GraphAttentionGCN"] = GraphAttentionGCN
        globals()["GraphAttentionLayer"] = GraphAttentionLayer
        __all__.extend(["GraphAttentionGCN", "GraphAttentionLayer"])


_optional_imports()
del _optional_imports
