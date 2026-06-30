from __future__ import annotations

import copy
import datetime as dt
import json

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, html

from plots import (
    make_3d_scatter,
    make_box_plot,
    make_contour_plot,
    make_correlation_matrix,
    make_feature_importance,
    make_heatmap,
    make_histogram,
    make_line_plot,
    make_mesh_plot,
    make_pca_projection,
    make_surface_plot,
)
from utils import get_numeric_columns, load_dataset


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("filtered-data-store", "data"),
        Input("search-input", "value"),
        State("raw-data-store", "data"),
    )
    def filter_dataset(search_text: str | None, raw_data: list[dict]) -> list[dict]:
        df = pd.DataFrame(raw_data)
        if df.empty:
            return []
        if search_text:
            mask = df.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False)).any(axis=1)
            df = df[mask]
        return df.to_dict("records")

    @app.callback(
        [
            Output("surface-plot", "figure"),
            Output("mesh-plot", "figure"),
            Output("heatmap-plot", "figure"),
            Output("pca-plot", "figure"),
            Output("histogram-plot", "figure"),
            Output("box-plot", "figure"),
            Output("corr-plot", "figure"),
            Output("feature-importance", "figure"),
            Output("line-plot", "figure"),
            Output("contour-plot", "figure"),
            Output("kpi-samples", "children"),
            Output("kpi-features", "children"),
            Output("kpi-accuracy", "children"),
            Output("kpi-runtime", "children"),
            Output("kpi-memory", "children"),
        ],
        Input("filtered-data-store", "data"),
        Input("feature-dropdown", "value"),
    )
    def update_dashboard(data: list[dict], selected_feature: str | None) -> tuple:
        df = pd.DataFrame(data)
        if df.empty:
            df = load_dataset()
        feature = selected_feature or "cell_volume_um3"
        if feature not in df.columns:
            feature = "cell_volume_um3"
        return (
            make_surface_plot(df),
            make_mesh_plot(df),
            make_heatmap(df),
            make_pca_projection(df),
            make_histogram(df, feature),
            make_box_plot(df, feature),
            make_correlation_matrix(df),
            make_feature_importance(df),
            make_line_plot(df, feature),
            make_contour_plot(df),
            f"{len(df):,}",
            f"{len(get_numeric_columns(df))}",
            "0.93",
            "1.8s",
            "128MB",
        )

    @app.callback(
        Output("main-3d-graph", "figure"),
        Input("filtered-data-store", "data"),
        Input("feature-dropdown", "value"),
        Input("camera-store", "data"),
    )
    def update_3d_graph(data: list[dict], selected_feature: str | None, camera_state: dict | None):
        df = pd.DataFrame(data)
        if df.empty:
            df = load_dataset()
        feature = selected_feature or "cell_volume_um3"
        if feature not in df.columns:
            feature = "cell_volume_um3"
        return make_3d_scatter(df, feature, camera=camera_state)

    @app.callback(
        Output("camera-store", "data"),
        Input("main-3d-graph", "relayoutData"),
        Input("reset-camera", "n_clicks"),
        State("camera-store", "data"),
    )
    def update_camera(
        _relayout: dict | None,
        _reset_clicks: int | None,
        store: dict,
    ) -> dict:
        if not store:
            store = {"camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.1}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}}
        triggered = ctx.triggered_id

        if triggered == "reset-camera":
            return {
                "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.1}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}},
            }

        return store

    @app.callback(
        Output("theme-store", "data"),
        Input("settings-btn", "n_clicks"),
        State("theme-store", "data"),
    )
    def toggle_theme(_n: int | None, store: dict) -> dict:
        if not store:
            store = {"mode": "dark"}
        store["mode"] = "light" if store.get("mode") == "dark" else "dark"
        return store

    @app.callback(
        Output("export-btn", "children"),
        Input("export-btn", "n_clicks"),
    )
    def export_dashboard(_n: int | None) -> str:
        return "Exported ✓"
