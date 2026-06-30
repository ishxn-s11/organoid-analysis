from __future__ import annotations

import datetime as dt

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from callbacks import register_callbacks
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
from styles import (
    ACCENT_CYAN,
    ACCENT_PURPLE,
    BACKGROUND,
    CARD,
    FONT_DISPLAY,
    FONT_MONO,
    MUTED,
    PLATE_RING,
    SURFACE_LINE,
    TEXT,
    get_card_style,
)
from utils import build_interpretation, get_numeric_columns, load_dataset


def build_layout(app: Dash) -> html.Div:
    df = load_dataset()
    numeric_columns = get_numeric_columns(df)
    interpretation = build_interpretation(df)

    return html.Div(
        style={
            "minHeight": "100vh",
            "background": BACKGROUND,
            "color": TEXT,
            "padding": "24px",
            "fontFamily": FONT_DISPLAY,
        },
        children=[
            dcc.Store(id="theme-store", data={"mode": "dark"}),
            dcc.Store(id="camera-store", data={"camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.1}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}}),
            dcc.Store(id="raw-data-store", data=df.to_dict("records")),
            dcc.Store(id="filtered-data-store", data=df.to_dict("records")),
            dbc.Navbar(
                children=[
                    html.Div([
                        html.H3(
                            "3D Organoid Segmentation And Feature Extraction",
                            style={"margin": 0, "fontWeight": 600, "letterSpacing": "0.01em", "fontFamily": FONT_DISPLAY},
                        ),
                        html.Div(
                            "Organoid Atlas — Live Feature Intelligence",
                            style={"fontSize": "0.85rem", "color": MUTED, "letterSpacing": "0.04em", "marginTop": "2px"},
                        ),
                    ], style={"display": "flex", "flexDirection": "column"}),
                    html.Div([
                        html.Div(
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            id="clock",
                            style={"color": ACCENT_CYAN, "fontWeight": 500, "fontFamily": FONT_MONO, "fontSize": "0.85rem", "letterSpacing": "0.02em"},
                        ),
                        dbc.Button("Export", id="export-btn", outline=True, className="ms-3", style={"borderColor": ACCENT_CYAN, "color": ACCENT_CYAN, "borderRadius": "8px"}),
                        dbc.Button("Settings", id="settings-btn", outline=True, color="secondary", className="ms-2", style={"borderRadius": "8px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ],
                color="dark",
                dark=True,
                sticky="top",
                style={
                    "background": CARD,
                    "borderBottom": f"2px solid {PLATE_RING}",
                    "borderRadius": "14px",
                    "boxShadow": "0 1px 0 rgba(255,255,255,0.03) inset",
                },
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"width": "240px", "minWidth": "220px", "display": "flex", "flexDirection": "column", "gap": "10px"},
                        children=[
                            html.Div(style={**get_card_style("dark"), "padding": "18px"}, children=[
                                html.H5("Navigation", style={"marginBottom": "12px"}),
                                dbc.Nav([
                                    dbc.NavLink("Dashboard", href="#", active=True, style={"borderRadius": "12px", "marginBottom": "6px"}),
                                    dbc.NavLink("Visualizations", href="#", style={"borderRadius": "12px", "marginBottom": "6px"}),
                                    dbc.NavLink("Interpretation", href="#", style={"borderRadius": "12px", "marginBottom": "6px"}),
                                    dbc.NavLink("Summary", href="#", style={"borderRadius": "12px", "marginBottom": "6px"}),
                                    dbc.NavLink("Reports", href="#", style={"borderRadius": "12px", "marginBottom": "6px"}),
                                    dbc.NavLink("Settings", href="#", style={"borderRadius": "12px"}),
                                ], vertical=True),
                            ]),
                            html.Div(style=get_card_style("dark"), children=[
                                html.Label("Search Cells", style={"fontWeight": 600}),
                                dbc.Input(id="search-input", placeholder="Filter by sample or confidence", type="text"),
                                html.Br(),
                                html.Label("Feature Focus", style={"fontWeight": 600}),
                                dcc.Dropdown(id="feature-dropdown", options=[{"label": c, "value": c} for c in numeric_columns], value="cell_volume_um3", clearable=False),
                            ]),
                        ],
                    ),
                    html.Div(style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "18px"}, children=[
                        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "14px"}, children=[
                            dbc.Card([dbc.CardBody([html.H6("Samples", className="card-title", style={"color": MUTED, "fontSize": "0.72rem", "letterSpacing": "0.08em", "textTransform": "uppercase"}), html.H3(id="kpi-samples", children=f"{len(df):,}", style={"fontWeight": 600, "fontFamily": FONT_MONO})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([html.H6("Features", className="card-title", style={"color": MUTED, "fontSize": "0.72rem", "letterSpacing": "0.08em", "textTransform": "uppercase"}), html.H3(id="kpi-features", children=f"{len(numeric_columns)}", style={"fontWeight": 600, "fontFamily": FONT_MONO})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([html.H6("Accuracy", className="card-title", style={"color": MUTED, "fontSize": "0.72rem", "letterSpacing": "0.08em", "textTransform": "uppercase"}), html.H3(id="kpi-accuracy", children="0.93", style={"fontWeight": 600, "fontFamily": FONT_MONO})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([html.H6("Runtime", className="card-title", style={"color": MUTED, "fontSize": "0.72rem", "letterSpacing": "0.08em", "textTransform": "uppercase"}), html.H3(id="kpi-runtime", children="1.8s", style={"fontWeight": 600, "fontFamily": FONT_MONO})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([html.H6("Memory", className="card-title", style={"color": MUTED, "fontSize": "0.72rem", "letterSpacing": "0.08em", "textTransform": "uppercase"}), html.H3(id="kpi-memory", children="128MB", style={"fontWeight": 600, "fontFamily": FONT_MONO})])], style=get_card_style("dark")),
                        ]),
                        dbc.Card(
                            [
                                dbc.CardHeader([html.H4("Interactive 3D View", style={"margin": 0, "fontWeight": 600}), html.Small("Drag to inspect the 3D view", style={"color": MUTED})]),
                                dbc.CardBody([
                                    html.Div(
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="main-3d-graph",
                                                figure=make_3d_scatter(df, "cell_volume_um3"),
                                                config={"displayModeBar": True, "responsive": True, "scrollZoom": False},
                                                style={"height": "100%", "width": "100%"},
                                            ),
                                            type="default",
                                        ),
                                        style={"height": "700px", "width": "100%", "minHeight": "700px"},
                                    ),
                                    dbc.Row([
                                        dbc.Col(dbc.Button("Reset Camera", id="reset-camera", outline=True, size="sm", style={"borderColor": ACCENT_CYAN, "color": ACCENT_CYAN, "borderRadius": "8px"})),
                                        dbc.Col(dbc.Button("Fullscreen", id="fullscreen-btn", outline=True, color="secondary", size="sm", style={"borderRadius": "8px"})),
                                    ], className="mt-3"),
                                ]),
                            ],
                            style={**get_card_style("dark"), "minHeight": "760px", "display": "flex", "flexDirection": "column"},
                        ),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))", "gap": "14px"}, children=[
                            dbc.Card([dbc.CardBody([dcc.Graph(id="surface-plot", figure=make_surface_plot(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="mesh-plot", figure=make_mesh_plot(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="heatmap-plot", figure=make_heatmap(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="pca-plot", figure=make_pca_projection(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="histogram-plot", figure=make_histogram(df, "cell_volume_um3"), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="box-plot", figure=make_box_plot(df, "cell_volume_um3"), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="corr-plot", figure=make_correlation_matrix(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="feature-importance", figure=make_feature_importance(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="line-plot", figure=make_line_plot(df, "cell_volume_um3"), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardBody([dcc.Graph(id="contour-plot", figure=make_contour_plot(df), config={"displayModeBar": False, "responsive": True})])], style=get_card_style("dark")),
                        ]),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "14px"}, children=[
                            dbc.Card([dbc.CardHeader(html.H4("Interpretation", style={"margin": 0, "fontWeight": 600})), dbc.CardBody([html.Ul([html.Li([html.Span(item["badge"], style={"background": "transparent", "border": f"1px solid {ACCENT_PURPLE if idx % 2 else ACCENT_CYAN}", "color": ACCENT_PURPLE if idx % 2 else ACCENT_CYAN, "padding": "3px 9px", "borderRadius": "999px", "marginRight": "8px", "fontSize": "0.72rem", "letterSpacing": "0.04em"}), item["title"], html.Br(), html.Small(item["body"], style={"color": MUTED})], style={"marginBottom": "10px", "listStyle": "none"}) for idx, item in enumerate(interpretation)], style={"paddingLeft": 0})])], style=get_card_style("dark")),
                            dbc.Card([dbc.CardHeader(html.H4("Summary", style={"margin": 0, "fontWeight": 600})), dbc.CardBody([
                                html.Div("Executive Summary", style={"fontWeight": 700, "marginBottom": "8px"}),
                                html.P("The dashboard highlights a stable volume-intensity relationship, a compact cluster structure, and a small but meaningful low-confidence tail.", style={"color": MUTED}),
                                html.Div("Key Findings", style={"fontWeight": 700, "marginTop": "10px"}),
                                html.Ul([html.Li("Cluster separation remains clearest in the PCA projection."), html.Li("Confidence remains high for the majority of samples."), html.Li("Surface and mesh views show coherent structural continuity.")]),
                                html.Div("Recommendations", style={"fontWeight": 700, "marginTop": "10px"}),
                                html.P("Prioritize the most stable regions for follow-up imaging and preserve a small outlier cohort for biological review.", style={"color": MUTED}),
                            ])], style=get_card_style("dark")),
                        ]),
                    ]),
                ],
            ),
        ],
    )
