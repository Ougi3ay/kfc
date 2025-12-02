from typing import Any, Dict, Optional, TypedDict, Union
import numpy.typing as npt
import numpy as np

ArrayLike2D = npt.NDArray[np.floating]
ArrayLike1D = npt.NDArray[np.floating]

DivergenceSpec = Union[str, Dict[str, Any]]
class DivergenceConfig(TypedDict, total=False):
    name: str
    name: str
    n_clusters: int
    max_iter: int
    n_init: int
    tol: float
    random_state: Optional[int]
    deg: Optional[int]