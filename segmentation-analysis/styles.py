# Palette inspired by phase-contrast microscopy and lab notebooks:
# warm charcoal field, parchment text, desaturated sage as the primary
# data color, dusty clay as the secondary/highlight accent.
ACCENT_CYAN = "#8FA888"     # sage — primary data series (kept name for compat)
ACCENT_PURPLE = "#C97B5B"   # dusty clay — secondary data series
SUCCESS = "#7C9A82"         # muted moss — positive / tertiary signal
WARNING = "#C99A5B"         # ochre — caution / outlier signal
BACKGROUND = "#14130F"      # warm near-black, not pure neutral black
CARD = "#1C1B16"            # slightly lifted warm charcoal
TEXT = "#EDE8DD"            # parchment
MUTED = "#9B9587"           # warm grey-beige

# Extra tokens used by the restyled layout
SURFACE_LINE = "rgba(237,232,221,0.08)"
PLATE_RING = "rgba(143,168,136,0.35)"
FONT_DISPLAY = "'IBM Plex Sans', 'Helvetica Neue', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', monospace"


def get_card_style(theme_mode: str = "dark") -> dict:
    bg = CARD if theme_mode == "dark" else "#FAF8F2"
    text = TEXT if theme_mode == "dark" else "#262420"
    border = SURFACE_LINE if theme_mode == "dark" else "rgba(38,36,32,0.08)"
    return {
        "background": bg,
        "border": f"1px solid {border}",
        "borderRadius": "14px",
        "boxShadow": "0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 24px rgba(0,0,0,0.22)",
        "padding": "18px",
        "color": text,
        "fontFamily": FONT_DISPLAY,
        "transition": "all 220ms ease",
    }


def get_kpi_style(theme_mode: str = "dark") -> dict:
    style = get_card_style(theme_mode)
    style.update({
        "padding": "16px 20px",
        "minHeight": "112px",
        "borderTop": f"2px solid {PLATE_RING}",
    })
    return style
