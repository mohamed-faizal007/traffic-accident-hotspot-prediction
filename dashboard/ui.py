"""Presentation-only helpers for app.py: CSS injection, tier badges, a heatmap-style
confusion matrix, and themed charts. Nothing here computes or selects what is shown —
every value passed in comes from the caller, which reads it from results/metrics.json.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

TIER_COLOR = {
    "Critical": "#d03b3b",
    "High": "#ec835a",
    "Medium": "#fab219",
    "Low": "#0ca30c",
}

TIER_BG = {"Critical": "#fbe4e4", "High": "#fce4d9", "Medium": "#fef0d3", "Low": "#e0f5e0"}
TIER_FG = {"Critical": "#d03b3b", "High": "#b5541f", "Medium": "#8a6200", "Low": "#0a7a0a"}

SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

_GOOGLE_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap" '
    'rel="stylesheet">'
)


def inject_css(base_dir: Path) -> None:
    """Load dashboard/style.css once and inject it, plus the Inter font, via st.html.

    st.markdown(unsafe_allow_html=True) strips <style> tags, which leaves the raw CSS
    text visible on the page instead of applying it. st.html() is Streamlit's documented
    mechanism for injecting a stylesheet: passing a Path to a .css file auto-wraps it in
    <style> tags and, since the content is style-only, routes it to the event container
    so it applies without taking up layout space.
    """
    st.html(_GOOGLE_FONT_LINK)
    css_path = base_dir / "dashboard" / "style.css"
    if css_path.exists():
        st.html(css_path)


def tier_badge(tier: str) -> str:
    """Inline HTML for a single risk-tier pill. Caller embeds this in an st.markdown call."""
    return f'<span class="tier-pill tier-{tier}"><span class="tier-dot tier-{tier}"></span>{tier}</span>'


def tier_legend(tiers) -> str:
    items = "".join(f'<div>{tier_badge(t)}</div>' for t in tiers)
    return f'<div class="tier-legend">{items}</div>'


def hero(title: str, subtitle: str, badge_text: str | None = None) -> None:
    badge = f'<div class="hp-badge">{badge_text}</div>' if badge_text else ""
    st.markdown(
        f'<div class="hp-hero"><h1>{title}</h1><p>{subtitle}</p>{badge}</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(f'<div class="hp-section-title">{prefix}{text}</div>', unsafe_allow_html=True)


def _shade(count: int, total: int) -> str:
    """Pick a step of the sequential-blue ramp by this cell's share of the matrix total."""
    frac = 0 if total == 0 else count / total
    idx = min(int(frac * len(SEQUENTIAL_BLUE)), len(SEQUENTIAL_BLUE) - 1)
    return SEQUENTIAL_BLUE[idx]


def confusion_matrix_html(tn: int, fp: int, fn: int, tp: int) -> str:
    """A 2x2 heatmap-style table: cell fill intensity (sequential blue) encodes magnitude."""
    total = tn + fp + fn + tp
    cells = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    bg = {k: _shade(v, total) for k, v in cells.items()}
    # Only the two darkest ramp steps clear 4.5:1 with white text; the mid step (#3987e5)
    # measures ~3.6:1 with white but ~5.5:1 with dark text, so it stays dark-on-light too.
    fg = {k: ("#ffffff" if v in SEQUENTIAL_BLUE[3:] else "#0b0b0b") for k, v in bg.items()}
    return f"""
    <table class="cm-grid">
      <tr><td class="cm-corner"></td><th>predicted: no</th><th>predicted: yes</th></tr>
      <tr><th>actual: no hotspot</th>
          <td class="cm-cell" style="background:{bg['tn']};color:{fg['tn']}">{tn:,}</td>
          <td class="cm-cell" style="background:{bg['fp']};color:{fg['fp']}">{fp:,}</td></tr>
      <tr><th>actual: hotspot</th>
          <td class="cm-cell" style="background:{bg['fn']};color:{fg['fn']}">{fn:,}</td>
          <td class="cm-cell" style="background:{bg['tp']};color:{fg['tp']}">{tp:,}</td></tr>
    </table>
    """


def style_tier_column(df: pd.DataFrame, column: str = "risk_tier"):
    """Pandas Styler that renders the risk-tier column as coloured chips inside st.dataframe."""

    def _style(val: str) -> str:
        bg, fg = TIER_BG.get(val), TIER_FG.get(val)
        return f"background-color:{bg};color:{fg};font-weight:600;" if bg else ""

    return df.style.map(_style, subset=[column])


def _base_chart_config(chart: alt.Chart) -> alt.Chart:
    return chart.configure_axis(
        gridColor="#e1e0d9",
        domainColor="#c3c2b7",
        labelColor="#52514e",
        titleColor="#52514e",
        labelFont="Inter",
        titleFont="Inter",
    ).configure_view(strokeWidth=0).configure(background="#ffffff", font="Inter")


def themed_bar_chart(series: pd.Series, value_title: str, x_title: str = "") -> alt.Chart:
    """A single-hue (sequential blue) bar chart, themed to match the dashboard palette."""
    df = series.rename(value_title).rename_axis(x_title or "category").reset_index()
    x_col, y_col = df.columns[0], df.columns[1]
    chart = (
        alt.Chart(df)
        .mark_bar(color="#2a78d6", cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=18)
        .encode(
            x=alt.X(f"{x_col}:N", sort=None, title=x_title or None),
            y=alt.Y(f"{y_col}:Q", title=value_title),
            tooltip=[x_col, alt.Tooltip(f"{y_col}:Q", format=".4f")],
        )
        .properties(height=280)
    )
    return _base_chart_config(chart)
