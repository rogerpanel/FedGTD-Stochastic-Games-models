"""
ICS3D (Integrated Cloud Security 3-Datasets) handler.

Loads and preprocesses the three component datasets:
  1. Edge-IIoT  – 2,219,201 samples, 14 attack families   (Ferrag et al.)
  2. Container  – 234,560 samples, 11 CVE exploit classes  (Caprolu et al.)
  3. SOC/GUIDE  – 13M+ triage events, 33 entity types      (Microsoft GUIDE)

Dataset DOI: https://doi.org/10.34740/kaggle/dsv/12483891
Kaggle slug : rogernickanaedevha/integrated-cloud-security-3datasets-ics3d

Reference: Paper Section 6.1–6.3 (Dataset Description and Federated Partitioning)
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fedgtd.config import GameConfig


class ICS3DDataHandler:
    """Download, preprocess, and partition the ICS3D dataset."""

    KAGGLE_SLUG = "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d"

    # Expected filenames inside the downloaded dataset directory
    EDGE_FILE = "DNN-EdgeIIoT-dataset.csv"
    CONTAINER_FILE = "Containers_Dataset.csv"
    SOC_FILE = "Microsoft_GUIDE_Train.csv"

    def __init__(self, config: GameConfig):
        self.config = config
        self.scalers = {
            "edge": StandardScaler(),
            "container": StandardScaler(),
            "soc": MinMaxScaler(),
        }
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # ── Download ─────────────────────────────────────────────────────────

    def download(self) -> Optional[Path]:
        """Download ICS3D from Kaggle using kagglehub."""
        try:
            import kagglehub
            path = kagglehub.dataset_download(self.KAGGLE_SLUG)
            print(f"Dataset downloaded to: {path}")
            return Path(path)
        except Exception as e:
            print(f"Warning: could not download dataset ({e}). "
                  "Falling back to synthetic data.")
            return None

    # ── Load individual components ───────────────────────────────────────

    def load_edge_iiot(self, data_path: Optional[Path] = None,
                       max_samples: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Load Edge-IIoT component (Section 6.1).

        Binary task: Normal vs Attack.
        Protocol-specific features are label-encoded; missing values filled with 0.
        """
        fp = data_path / self.EDGE_FILE if data_path else None
        if fp and fp.exists():
            df = pd.read_csv(fp, low_memory=False)
            if max_samples and len(df) > max_samples:
                df = df.sample(max_samples, random_state=self.config.seed)

            if "Attack_type" in df.columns:
                y = (df["Attack_type"] != "Normal").astype(int).values
                X = df.drop(columns=["Attack_type"])
            else:
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1]

            for col in X.select_dtypes(include="object").columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            X = X.fillna(0).values.astype(np.float32)
        else:
            print("Generating synthetic Edge-IIoT data (real data not found)...")
            n = max_samples or 50_000
            rng = np.random.RandomState(self.config.seed)
            X = rng.randn(n, self.config.edge_features).astype(np.float32)
            X[:, :10] = np.abs(X[:, :10]) * 100          # flow statistics
            X[:, 10:20] = rng.randint(0, 256, (n, 10))   # protocol fields
            y = rng.choice([0, 1], size=n, p=[0.721, 0.279])

        X = self.scalers["edge"].fit_transform(X)
        self._report("Edge-IIoT", X, y)
        return X, y

    def load_container(self, data_path: Optional[Path] = None,
                       max_samples: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Load Container component (Section 6.1).

        Binary task: Benign (label 0) vs Attack (label > 0).
        CVE string labels are encoded then binarised.
        """
        fp = data_path / self.CONTAINER_FILE if data_path else None
        if fp and fp.exists():
            df = pd.read_csv(fp, low_memory=False)
            if max_samples and len(df) > max_samples:
                df = df.sample(max_samples, random_state=self.config.seed)

            if "Label" in df.columns:
                y_raw = df["Label"]
                X = df.drop(columns=["Label"])
            else:
                y_raw = df.iloc[:, -1]
                X = df.iloc[:, :-1]

            if y_raw.dtype == object:
                le = LabelEncoder()
                y_raw = le.fit_transform(y_raw)
                self.label_encoders["container"] = le

            y = (np.asarray(y_raw) > 0).astype(int)

            for col in X.select_dtypes(include="object").columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            X = X.fillna(0).values.astype(np.float32)
        else:
            print("Generating synthetic Container data (real data not found)...")
            n = max_samples or 20_000
            rng = np.random.RandomState(self.config.seed + 1)
            X = rng.randn(n, self.config.container_features).astype(np.float32)
            X[:, :20] = np.abs(X[:, :20]) * 1000
            X[:, 20:40] = rng.exponential(0.1, (n, min(20, self.config.container_features - 20)))
            y = rng.choice([0, 1], size=n, p=[0.94, 0.06])

        X = self.scalers["container"].fit_transform(X)
        self._report("Container", X, y)
        return X, y

    def load_soc(self, data_path: Optional[Path] = None,
                 max_samples: Optional[int] = 100_000
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Load Microsoft GUIDE / SOC component (Section 6.1).

        Binary task: TruePositive vs rest (BenignPositive + FalsePositive).
        High-cardinality string features are hashed to bounded integers.
        """
        fp = data_path / self.SOC_FILE if data_path else None
        if fp and fp.exists():
            df = pd.read_csv(fp, nrows=max_samples, low_memory=False)

            if "IncidentGrade" in df.columns:
                grade_map = {"TruePositive": 2, "BenignPositive": 1, "FalsePositive": 0}
                y = df["IncidentGrade"].map(grade_map).fillna(0).astype(int).values
                X = df.drop(columns=["IncidentGrade", "Id"], errors="ignore")
            else:
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1]

            y = (y == 2).astype(int)

            for col in X.select_dtypes(include="object").columns:
                X[col] = X[col].astype(str).apply(
                    lambda v: int(hashlib.md5(v.encode()).hexdigest()[:8], 16) % 10_000
                )

            X = X.fillna(0).values.astype(np.float32)
        else:
            print("Generating synthetic SOC data (real data not found)...")
            n = max_samples or 30_000
            rng = np.random.RandomState(self.config.seed + 2)
            X = rng.randn(n, self.config.soc_features).astype(np.float32)
            X[:, :10] = rng.poisson(5, (n, 10))
            X[:, 10:20] = rng.uniform(0, 1, (n, 10))
            y = rng.choice([0, 1], size=n, p=[0.992, 0.008])

        X = self.scalers["soc"].fit_transform(X)
        self._report("SOC/GUIDE", X, y)
        return X, y

    # ── Federated partitioning (Section 6.3) ─────────────────────────────

    def create_federated_splits(
        self, X: np.ndarray, y: np.ndarray,
        n_clients: int, alpha: float = 0.3,
    ) -> List[Dict[str, np.ndarray]]:
        """Non-IID partition using Dirichlet(α) distribution.

        Following Hsu et al. (2019) as described in Section 6.3:
        for each class, sample proportions ~ Dir(α · 1_K), then assign
        data to clients proportionally.
        """
        rng = np.random.RandomState(self.config.seed)
        n_classes = len(np.unique(y))
        class_indices = {c: np.where(y == c)[0] for c in range(n_classes)}

        # Initialise per-client index lists
        client_indices: List[list] = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            indices = class_indices[c].copy()
            rng.shuffle(indices)

            proportions = rng.dirichlet(np.ones(n_clients) * alpha)
            counts = (proportions * len(indices)).astype(int)
            counts[-1] = len(indices) - counts[:-1].sum()  # absorb rounding

            start = 0
            for k in range(n_clients):
                end = start + counts[k]
                if counts[k] > 0:
                    client_indices[k].extend(indices[start:end].tolist())
                start = end

        splits = []
        for k in range(n_clients):
            idx = np.array(client_indices[k])
            if len(idx) > 0:
                splits.append({"X": X[idx], "y": y[idx]})
            else:
                # Ensure every client has at least a small amount of data
                fallback = rng.choice(len(X), size=max(10, len(X) // n_clients), replace=False)
                splits.append({"X": X[fallback], "y": y[fallback]})

        return splits

    # ── DataLoader factories ─────────────────────────────────────────────

    @staticmethod
    def splits_to_loaders(splits: List[Dict[str, np.ndarray]],
                          batch_size: int, shuffle: bool = True
                          ) -> List[DataLoader]:
        loaders = []
        for split in splits:
            ds = TensorDataset(
                torch.from_numpy(split["X"]).float(),
                torch.from_numpy(split["y"]).long(),
            )
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle))
        return loaders

    @staticmethod
    def arrays_to_loader(X: np.ndarray, y: np.ndarray,
                         batch_size: int, shuffle: bool = False) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _report(name: str, X: np.ndarray, y: np.ndarray):
        counts = np.bincount(y)
        print(f"  {name}: {X.shape[0]:,} samples, {X.shape[1]} features | "
              f"class counts = {dict(enumerate(counts))}")
