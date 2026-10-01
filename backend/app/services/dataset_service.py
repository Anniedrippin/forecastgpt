"""
Dataset service.

Owns everything about *storing* and *loading* uploaded time-series CSVs.
This is intentionally simple (files on disk + a JSON registry) rather than
a full database -- for a portfolio project this keeps the "no paid infra"
requirement trivially true, and it is easy to explain in an interview:
"datasets are content-addressed by UUID, persisted as CSV, with a small
metadata registry so the API can list/describe them without re-reading
every file."

Swap point: replace this with Postgres/SQLite without touching agents,
forecasting, or API route signatures -- they only depend on the
`DatasetService` interface below.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_DATE_CANDIDATES = ["date", "ds", "timestamp", "time", "period", "month", "day"]
_VALUE_CANDIDATES = ["value", "y", "target", "sales", "demand", "count", "quantity", "revenue"]


@dataclass
class LoadedDataset:
    dataset_id: str
    df: pd.DataFrame  # columns: ["date", "value"], sorted, indexed by date
    date_column_original: str
    value_column_original: str
    filename: str


class DatasetService:
    def __init__(self) -> None:
        settings = get_settings()
        self.data_dir = Path(settings.data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "registry.json"
        if not self.registry_path.exists():
            self.registry_path.write_text("{}")

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def _read_registry(self) -> dict:
        return json.loads(self.registry_path.read_text() or "{}")

    def _write_registry(self, registry: dict) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2))

    # ------------------------------------------------------------------
    # Column inference
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_date_column(df: pd.DataFrame) -> str:
        for col in df.columns:
            if col.lower().strip() in _DATE_CANDIDATES:
                return col
        # fall back: first column that parses as datetime for >90% of rows
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col
        # last resort: first column
        return df.columns[0]

    @staticmethod
    def _infer_value_column(df: pd.DataFrame, date_col: str) -> str:
        numeric_cols = [
            c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in numeric_cols:
            if col.lower().strip() in _VALUE_CANDIDATES:
                return col
        if numeric_cols:
            return numeric_cols[0]
        raise ValueError("Could not find a numeric value column in the uploaded CSV.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_csv(
        self,
        file_bytes: bytes,
        filename: str,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
    ) -> LoadedDataset:
        import io

        raw = pd.read_csv(io.BytesIO(file_bytes))
        if raw.empty:
            raise ValueError("Uploaded CSV is empty.")

        date_col = date_column or self._infer_date_column(raw)
        value_col = value_column or self._infer_value_column(raw, date_col)

        df = raw[[date_col, value_col]].copy()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < 10:
            raise ValueError(
                "Need at least 10 valid rows of (date, value) after cleaning to be useful."
            )

        dataset_id = str(uuid.uuid4())[:8]
        csv_path = self.upload_dir / f"{dataset_id}.csv"
        df.to_csv(csv_path, index=False)

        freq = pd.infer_freq(df["date"])

        registry = self._read_registry()
        registry[dataset_id] = {
            "dataset_id": dataset_id,
            "filename": filename,
            "n_rows": len(df),
            "date_column": date_col,
            "value_column": value_col,
            "start_date": str(df["date"].min().date()),
            "end_date": str(df["date"].max().date()),
            "inferred_frequency": freq,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(csv_path),
        }
        self._write_registry(registry)
        logger.info(f"Ingested dataset {dataset_id} ({filename}, {len(df)} rows)")

        return LoadedDataset(
            dataset_id=dataset_id,
            df=df,
            date_column_original=date_col,
            value_column_original=value_col,
            filename=filename,
        )

    def load(self, dataset_id: str) -> LoadedDataset:
        registry = self._read_registry()
        if dataset_id not in registry:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        meta = registry[dataset_id]
        df = pd.read_csv(meta["csv_path"], parse_dates=["date"])
        return LoadedDataset(
            dataset_id=dataset_id,
            df=df,
            date_column_original=meta["date_column"],
            value_column_original=meta["value_column"],
            filename=meta["filename"],
        )

    def get_metadata(self, dataset_id: str) -> dict:
        registry = self._read_registry()
        if dataset_id not in registry:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        return registry[dataset_id]

    def list_datasets(self) -> list[dict]:
        return list(self._read_registry().values())


_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
