from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from styles import ACCENT_CYAN, ACCENT_PURPLE, SUCCESS, WARNING, MUTED, get_card_style

# Sequential / diverging scales built from the dashboard's own palette so every
# chart reads as one family instead of mixing in Plotly's stock Viridis/Jet/etc.
SAGE_CLAY_SCALE = [[0, "#14130F"], [0.45, ACCENT_CYAN], [0.75, WARNING], [1, ACCENT_PURPLE]]
SAGE_SEQUENTIAL = [[0, "#1C1B16"], [0.5, ACCENT_CYAN], [1, "#D8C9A3"]]
CLAY_SEQUENTIAL = [[0, "#1C1B16"], [0.5, ACCENT_PURPLE], [1, "#E8C9A8"]]
GRID_LINE = "rgba(237,232,221,0.10)"


def make_3d_scatter(df: pd.DataFrame, feature: str, camera: dict | None = None) -> go.Figure:
    fig = go.Figure()
    camera = camera or {}
    if isinstance(camera, dict) and "camera" in camera and isinstance(camera["camera"], dict):
        camera = camera["camera"]
    eye = camera.get("eye", {}) if isinstance(camera, dict) else {}
    if not eye:
        eye = {"x": 1.8, "y": 1.8, "z": 1.1}
    eye_x = eye.get("x", 1.8)
    eye_y = eye.get("y", 1.8)
    eye_z = eye.get("z", 1.1)
    center = camera.get("center", {"x": 0, "y": 0, "z": 0}) if isinstance(camera, dict) else {"x": 0, "y": 0, "z": 0}
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(
                size=np.interp(df[feature], (df[feature].min(), df[feature].max()), (4, 12)),
                color=df[feature],
                colorscale=[[0, ACCENT_CYAN], [0.5, ACCENT_PURPLE], [1, SUCCESS]],
                opacity=0.84,
                line=dict(color="rgba(237,232,221,0.18)", width=0.4),
            ),
            text=[f"Cell {i}<br>{feature}: {val:.2f}" for i, val in zip(df["Cell_Number"], df[feature])],
            hoverinfo="text",
            name=feature,
        )
    )
    fig.update_layout(
        title="3D Organoid Feature Landscape",
        height=700,
        scene=dict(
            bgcolor="#0E0D0A",
            aspectmode="cube",
            xaxis=dict(title="X", showgrid=True, gridcolor=GRID_LINE, zeroline=False),
            yaxis=dict(title="Y", showgrid=True, gridcolor=GRID_LINE, zeroline=False),
            zaxis=dict(title="Z", showgrid=True, gridcolor=GRID_LINE, zeroline=False),
            camera=dict(eye=dict(x=eye_x, y=eye_y, z=eye_z), center=center, up=dict(x=0, y=0, z=1)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        font=dict(color="#EDE8DD"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        uirevision="organoid-3d-view",
    )
    return fig


def make_surface_plot(df: pd.DataFrame) -> go.Figure:
    x = np.linspace(df["x"].min(), df["x"].max(), 40)
    y = np.linspace(df["y"].min(), df["y"].max(), 40)
    xv, yv = np.meshgrid(x, y)
    zv = np.sin(xv / 3) + np.cos(yv / 3) + 0.15 * (xv + yv) / (abs(df["x"].max() - df["x"].min()) + 1)
    fig = go.Figure(go.Surface(x=xv, y=yv, z=zv, colorscale=[[0, ACCENT_CYAN], [1, ACCENT_PURPLE]], opacity=0.86, showscale=False))
    fig.update_layout(
        title="Surface View of Morphological Density",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(bgcolor="#0E0D0A", xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Signal")),
    )
    return fig


def make_mesh_plot(df: pd.DataFrame) -> go.Figure:
    subset = df.sample(min(90, len(df)), random_state=2)
    fig = go.Figure(
        go.Mesh3d(
            x=subset["x"].to_numpy(),
            y=subset["y"].to_numpy(),
            z=subset["z"].to_numpy(),
            intensity=subset["confidence"].to_numpy(dtype=float),
            colorscale=[[0, ACCENT_CYAN], [0.5, ACCENT_PURPLE], [1, SUCCESS]],
            opacity=0.64,
            hovertext=[f"Cell {i}" for i in subset["Cell_Number"]],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Mesh Plot of Structural Cohesion",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(bgcolor="#0E0D0A"),
    )
    return fig


def make_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[[c for c in df.columns if c not in {"Cell_Number", "sample_id", "cluster"}]].corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=SAGE_SEQUENTIAL, hoverongaps=False))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=40, r=20, t=40, b=20),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_pca_projection(df: pd.DataFrame) -> go.Figure:
    pca_df = pd.DataFrame({"PC1": df["x"] - df["y"], "PC2": df["z"] + df["cell_volume_um3"] / 1000, "cluster": df["cluster"]})
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", color_discrete_sequence=[ACCENT_CYAN, ACCENT_PURPLE, SUCCESS], hover_data={"cluster": True})
    fig.update_layout(
        title="PCA Projection of the Feature Space",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_histogram(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.histogram(df, x=feature, color_discrete_sequence=[ACCENT_CYAN], marginal="box")
    fig.update_layout(
        title=f"Distribution of {feature}",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_box_plot(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.box(df, x="cluster", y=feature, color="cluster", color_discrete_sequence=[ACCENT_CYAN, ACCENT_PURPLE, SUCCESS])
    fig.update_layout(
        title=f"{feature} by Cluster",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    corr = df[[c for c in df.columns if c not in {"Cell_Number", "sample_id", "cluster"}]].corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=CLAY_SEQUENTIAL))
    fig.update_layout(
        title="Correlation Matrix",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=40, r=20, t=40, b=20),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_feature_importance(df: pd.DataFrame) -> go.Figure:
    numeric = df[[c for c in df.columns if c not in {"Cell_Number", "sample_id", "cluster", "confidence"}]]
    scores = numeric.corrwith(df["cell_volume_um3"]).abs().sort_values(ascending=False)
    scores = scores.drop("cell_volume_um3", errors="ignore")
    fig = px.bar(scores.reset_index().rename(columns={"index": "feature", 0: "importance"}), x="feature", y="importance", color="importance", color_continuous_scale=SAGE_SEQUENTIAL)
    fig.update_layout(
        title="Feature Importance (Volume Correlation)",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_line_plot(df: pd.DataFrame, feature: str) -> go.Figure:
    ordered = df.sort_values(by="x")
    fig = px.line(ordered, x="x", y=feature, color_discrete_sequence=[ACCENT_CYAN])
    fig.update_layout(
        title=f"Temporal Trend of {feature}",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig


def make_contour_plot(df: pd.DataFrame) -> go.Figure:
    x = np.linspace(df["x"].min(), df["x"].max(), 35)
    y = np.linspace(df["y"].min(), df["y"].max(), 35)
    xv, yv = np.meshgrid(x, y)
    zv = np.sin(xv / 2.2) + np.cos(yv / 2.4)
    fig = go.Figure(go.Contour(x=x, y=y, z=zv, colorscale=SAGE_CLAY_SCALE, line_smoothing=0.9))
    fig.update_layout(
        title="Contour Map of the Feature Field",
        paper_bgcolor="#14130F",
        plot_bgcolor="#14130F",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#EDE8DD"),
    )
    return fig
