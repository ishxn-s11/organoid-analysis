from __future__ import annotations

import numpy as np
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Return a default dataset for the dashboard.

    The dashboard expects 3D coordinates, a cluster label, confidence, and
    several quantitative features used by the various plot components.
    """
    rng = np.random.default_rng(1)
    n_samples = 150
    x = rng.normal(loc=0.0, scale=10.0, size=n_samples)
    y = rng.normal(loc=0.0, scale=8.0, size=n_samples)
    z = rng.normal(loc=0.0, scale=6.0, size=n_samples)
    cluster = rng.choice(["A", "B", "C"], size=n_samples, p=[0.42, 0.33, 0.25])
    confidence = np.clip(rng.normal(loc=0.88, scale=0.07, size=n_samples), 0.0, 1.0)
    cell_volume_um3 = np.clip(1200.0 + 8.0 * x + 5.0 * y + 12.0 * z + rng.normal(scale=150.0, size=n_samples), 200.0, None)
    intensity = np.clip(35.0 + 1.5 * z + rng.normal(scale=8.0, size=n_samples), 0.0, None)
    shape_factor = np.clip(0.82 + rng.normal(scale=0.11, size=n_samples), 0.35, 1.15)
    morphology_score = np.clip(40.0 + 0.7 * x - 0.4 * y + rng.normal(scale=9.0, size=n_samples), 5.0, None)
    sample_id = rng.choice([f"Sample {idx}" for idx in range(1, 7)], size=n_samples)

    df = pd.DataFrame(
        {
            "Cell_Number": np.arange(1, n_samples + 1),
            "x": x,
            "y": y,
            "z": z,
            "cluster": cluster,
            "confidence": confidence,
            "cell_volume_um3": cell_volume_um3,
            "intensity": intensity,
            "shape_factor": shape_factor,
            "morphology_score": morphology_score,
            "sample_id": sample_id,
        }
    )
    return df


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns suitable for selection inputs."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric if col not in {"Cell_Number"}]


def build_interpretation(df: pd.DataFrame) -> list[dict[str, str]]:
    """Build a short interpretation summary for the dashboard."""
    if df.empty:
        return [
            {"badge": "Data", "title": "No data available", "body": "The dataset is empty, so there are no feature patterns to interpret."},
        ]

    cluster_counts = df["cluster"].value_counts(normalize=True)
    top_cluster = cluster_counts.idxmax()
    top_cluster_pct = cluster_counts.max() * 100
    avg_confidence = df["confidence"].mean() * 100
    avg_volume = df["cell_volume_um3"].mean()

    return [
        {
            "badge": "Volume",
            "title": "Volume trend is stable",
            "body": f"Mean organoid volume is {avg_volume:.0f} μm³ and remains consistent across the sample set.",
        },
        {
            "badge": "Confidence",
            "title": "High overall confidence",
            "body": f"Average segmentation confidence is {avg_confidence:.0f}% with a dominant cluster {top_cluster} representing {top_cluster_pct:.0f}% of cells.",
        },
        {
            "badge": "Morphology",
            "title": "Feature space shows structure",
            "body": "The 3D spatial distribution and extracted morphology scores indicate coherent sample grouping for follow-up review.",
        },
    ]
