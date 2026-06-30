import dash
import dash_bootstrap_components as dbc
from layout import build_layout
from callbacks import register_callbacks
from utils import load_dataset

FONT_STYLESHEET = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP, FONT_STYLESHEET],
    suppress_callback_exceptions=True,
)
app.title = "3D Organoid Segmentation And Feature Extraction"
app.layout = build_layout(app)
register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
