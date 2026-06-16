"""Medical dataset loaders. Binary classification, statevector-tractable sizes.

Five datasets:
  - WDBC               (sklearn built-in)         30 feat, 569 samples
  - Parkinson's voice  (UCI fetch + cache)        22 feat, 195 samples
  - Heart Cleveland    (UCI fetch + cache)        13 feat, 297 samples
  - PneumoniaMNIST     (MedMNIST + PCA-12)        12 feat, 5856 samples
  - BreastMNIST        (MedMNIST + PCA-12)        12 feat, 780 samples

UCI datasets cache to `data/` after first fetch. MedMNIST uses the medmnist
Python package; first call downloads ~10MB per dataset.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

CACHE_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: tuple[str, str]

    def summary(self) -> str:
        c = np.bincount(self.y)
        return (f"{self.name}: X={self.X.shape}  "
                f"y={self.target_names}={c.tolist()}  "
                f"n_feat={self.X.shape[1]}")


# ---------- WDBC ----------

def load_wdbc() -> Dataset:
    d = load_breast_cancer()
    return Dataset(
        name="wdbc",
        X=np.asarray(d.data, dtype=float),
        y=np.asarray(d.target, dtype=int),
        feature_names=list(d.feature_names),
        target_names=("malignant", "benign"),
    )


# ---------- Parkinson's ----------

_PARKINSONS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/parkinsons.data"
)


def _fetch(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    with urlopen(url, timeout=30) as r:
        blob = r.read()
    cache_path.write_bytes(blob)
    return blob


def load_parkinsons() -> Dataset:
    """UCI Parkinson's voice. status=1 → PD, status=0 → healthy."""
    blob = _fetch(_PARKINSONS_URL, CACHE_DIR / "parkinsons.csv")
    df = pd.read_csv(io.BytesIO(blob))
    feat_cols = [c for c in df.columns if c not in ("name", "status")]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["status"].to_numpy(dtype=int)
    return Dataset(
        name="parkinsons",
        X=X, y=y,
        feature_names=feat_cols,
        target_names=("healthy", "pd"),
    )


# ---------- Heart disease (Cleveland) ----------

_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

_CLEVELAND_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load_heart_cleveland() -> Dataset:
    """UCI processed Cleveland. Binarise: num > 0 → disease (1), else healthy (0).

    Drops rows with missing values ('?'). 13 features, ~297 samples after dropna.
    """
    blob = _fetch(_CLEVELAND_URL, CACHE_DIR / "cleveland.csv")
    df = pd.read_csv(io.BytesIO(blob), header=None, names=_CLEVELAND_COLS, na_values=["?"])
    df = df.dropna().reset_index(drop=True)
    feat_cols = _CLEVELAND_COLS[:-1]
    X = df[feat_cols].to_numpy(dtype=float)
    y = (df["num"].to_numpy(dtype=int) > 0).astype(int)
    return Dataset(
        name="heart_cleveland",
        X=X, y=y,
        feature_names=feat_cols,
        target_names=("healthy", "disease"),
    )


# ---------- registry ----------

# ---------- MedMNIST (Q1 benchmark) ----------

_MEDMNIST_BIN = {
    # name → (medmnist class name, 0-class index, 1-class index)
    "pneumonia_mnist": ("PneumoniaMNIST", 0, 1),
    "breast_mnist":    ("BreastMNIST", 0, 1),
    # Multi-class datasets binarized as class-0 vs rest
    "path_mnist":      ("PathMNIST", 0, -1),       # adipose vs rest (9-class)
    "derma_mnist":     ("DermaMNIST", 0, -1),      # 7-class skin lesion
    "blood_mnist":     ("BloodMNIST", 0, -1),      # 8-class blood cells
    "oct_mnist":       ("OCTMNIST", 0, -1),        # 4-class retinal OCT
    "tissue_mnist":    ("TissueMNIST", 0, -1),     # 8-class kidney tissue
    "organ_a_mnist":   ("OrganAMNIST", 0, -1),     # 11-class organ axial
}


def load_medmnist(name: str, n_components: int = 12) -> Dataset:
    """MedMNIST binary subset. Loads via medmnist package, flattens 28x28 → PCA n_components.

    Requires: pip install medmnist
    First call downloads ~10 MB per dataset to ~/.medmnist/
    """
    try:
        import medmnist
    except ImportError as e:
        raise ImportError("medmnist not installed. pip install medmnist") from e

    cls_name, c0, c1 = _MEDMNIST_BIN[name]
    cls = getattr(medmnist, cls_name)
    train = cls(split="train", download=True)
    val   = cls(split="val",   download=True)
    test  = cls(split="test",  download=True)

    def _flat(ds):
        X = ds.imgs.reshape(len(ds.imgs), -1).astype(float) / 255.0
        y = ds.labels.flatten().astype(int)
        if c1 == -1:
            # multi-class → binarize as class-c0 vs rest
            y = (y == c0).astype(int)
            # Don't filter — keep all samples
        return X, y

    X_tr, y_tr = _flat(train)
    X_va, y_va = _flat(val)
    X_te, y_te = _flat(test)
    X = np.concatenate([X_tr, X_va, X_te], axis=0)
    y = np.concatenate([y_tr, y_va, y_te])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    X = pca.fit_transform(X)
    explained = float(pca.explained_variance_ratio_.sum())

    return Dataset(
        name=f"medmnist_{name}",
        X=X, y=y,
        feature_names=[f"pc{i+1}" for i in range(n_components)],
        target_names=("class0", "class1"),
    )


LOADERS = {
    "wdbc": load_wdbc,
    "parkinsons": load_parkinsons,
    "heart": load_heart_cleveland,
    "pneumonia_mnist": lambda: load_medmnist("pneumonia_mnist"),
    "breast_mnist":    lambda: load_medmnist("breast_mnist"),
    "path_mnist":      lambda: load_medmnist("path_mnist"),
    "derma_mnist":     lambda: load_medmnist("derma_mnist"),
    "blood_mnist":     lambda: load_medmnist("blood_mnist"),
    "oct_mnist":       lambda: load_medmnist("oct_mnist"),
    "tissue_mnist":    lambda: load_medmnist("tissue_mnist"),
    "organ_a_mnist":   lambda: load_medmnist("organ_a_mnist"),
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; pick from {list(LOADERS)}")
    return LOADERS[name]()


def load_all() -> list[Dataset]:
    return [load(n) for n in LOADERS]
