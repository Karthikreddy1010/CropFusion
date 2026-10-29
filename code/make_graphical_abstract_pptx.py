"""Graphical abstract as an editable PowerPoint slide, built only from native shapes, text and charts drawn
by this project's code from the result files (no generated imagery).

1. Draws the chart panels with matplotlib from the result artefacts (manuscript/figures/ga_parts/).
2. Builds manuscript/figures/Graphical_Abstract.pptx with python-pptx.
3. Exports the slide to PNG and TIFF with PowerPoint (separate instance; the user's PowerPoint is not touched).

Run:  python code/make_graphical_abstract_pptx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import make_manuscript_figures as mf  # noqa: E402  (styles, palette, study-area map)

ROOT = CODE.parent
FIG = ROOT / "manuscript" / "figures"
PARTS = FIG / "ga_parts"
D = ROOT / "outputs_diagnostics" / "reports"

NAVY, INK, INK2 = "1F3A5F", "0B0B0B", "52514E"
BLUE, ORANGE, AQUA, YELLOW = "2A78D6", "EB6834", "1BAF7A", "EDA100"
COL_BG = ["EAF3FB", "EEF7EE", "FDF3E6", "FCECEC"]      # light panel fills
COL_HEAD = ["1F5F9E", "2E7D4F", "B9650B", "B23A3A"]    # panel title colours


# ─────────────────────────────────────────────── chart panels (from result files)
def draw_parts() -> dict:
    PARTS.mkdir(parents=True, exist_ok=True)
    out = {}

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    mf._draw_study_area(fig, ax)
    fig.axes[-1].set_ylabel("Share of seasons in the Extreme class", fontsize=6.5)
    fig.savefig(PARTS / "map.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out["map"] = PARTS / "map.png"

    rs = pd.read_csv(D / "rolling_origin_summary.csv").set_index(["method", "stratum"])
    st, gc = rs.loc[("rolling_static_cp", "Extreme")], rs.loc[("group_conditional_cp", "Extreme")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.6, 1.35), gridspec_kw={"wspace": 0.55})
    for ax, key, ref, title, lim in ((a1, "picp_weighted", 0.90, "Extreme-class coverage", (0.6, 1.0)),
                                     (a2, "unsafe_weighted", 0.05, "Unsafe miss rate", (0, 0.3))):
        vals = [st[key], gc[key]]
        bars = ax.bar([0, 1], vals, 0.6, color=[mf.BLUE, mf.ORANGE])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (lim[1] - lim[0]) * 0.03, f"{v:.3f}", ha="center",
                    fontsize=6.2, color=mf.INK)
        ax.axhline(ref, color=mf.INK2, lw=0.8, ls="--")
        ax.text(1.45, ref, f"{ref:.2f}", fontsize=5.4, color=mf.INK2, va="bottom", ha="right")
        ax.set_ylim(*lim); ax.set_xticks([0, 1]); ax.set_xticklabels(["Static", "Group-cond."], fontsize=6)
        ax.set_title(title, fontsize=6.8, loc="left"); ax.tick_params(labelsize=5.6, length=2)
    fig.savefig(PARTS / "group_conditional.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out["gc"] = PARTS / "group_conditional.png"
    out["gc_vals"] = dict(st_picp=st.picp_weighted, gc_picp=gc.picp_weighted,
                          gain_lo=None, st_umr=st.unsafe_weighted, gc_umr=gc.unsafe_weighted,
                          normal_gc=rs.loc[("group_conditional_cp", "Normal")].picp_weighted,
                          normal_st=rs.loc[("rolling_static_cp", "Normal")].picp_weighted)

    ro3 = json.loads((D / "objectives_RO_report.json").read_text(encoding="utf-8"))["RO3"]
    fig, ax = plt.subplots(figsize=(3.6, 0.95))
    labels = ["Coverage vs |bias|", "Coverage vs width"]
    vals = [ro3["corr_abs_bias_vs_coverage"], ro3["corr_width_vs_coverage"]]
    ax.barh([1, 0], vals, 0.5, color=[mf.ORANGE, mf.BLUE])
    for y, v in zip([1, 0], vals):
        ax.text(v + (-0.03 if v < 0 else 0.03), y, f"r = {v:.2f}", va="center", ha="right" if v < 0 else "left",
                fontsize=6.2, color=mf.INK)
    ax.axvline(0, color=mf.INK2, lw=0.6)
    ax.set_yticks([1, 0]); ax.set_yticklabels(labels, fontsize=6.2); ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.tick_params(labelsize=5.4, length=2)
    ax.set_xlabel(f"Pearson correlation, {ro3['loso_state_years']} leave-one-state-out state-years", fontsize=5.6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(PARTS / "bias_width.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out["bw"] = PARTS / "bias_width.png"
    out["locked"] = FIG / "fig_marginal_vs_extreme_coverage.png"      # Figure 2, same code and data
    return out


# ─────────────────────────────────────────────── slide helpers
def rgb(h):
    return RGBColor.from_string(h)


def box(sl, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def text(sl, x, y, w, h, runs, size=9, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, color=INK, bold=False,
         spacing=1.0):
    """runs: str, or list of paragraphs; each paragraph a str or list of (text, dict(bold, color, size))."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        for seg, fmt in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = seg
            r.font.name = "Times New Roman"; r.font.size = Pt(fmt.get("size", size))
            r.font.bold = fmt.get("bold", bold); r.font.color.rgb = rgb(fmt.get("color", color))
            r.font.italic = fmt.get("italic", False)
    return tb


def bullet_list(sl, x, y, w, items, size=8.6, marker=None, gap=0.30, sub_size=7.8):
    """items: (heading, detail). A small coloured square replaces icons."""
    for i, (head, detail) in enumerate(items):
        yy = y + i * gap
        if marker:
            box(sl, x, yy + 0.06, 0.09, 0.09, fill=marker, shape=MSO_SHAPE.RECTANGLE)
        text(sl, x + 0.14, yy, w - 0.14, gap, [[(head, {"bold": True, "size": size})],
                                               [(detail, {"size": sub_size, "color": INK2})]], spacing=0.9)


def panel(sl, k, x, y, w, h, title):
    box(sl, x, y, w, h, fill=COL_BG[k], line="D0D0D0", radius=0.03)
    text(sl, x + 0.08, y + 0.04, w - 0.16, 0.35, title, size=13, bold=True, color=COL_HEAD[k])


# ─────────────────────────────────────────────── slide
def build(parts: dict) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(6.667)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    box(sl, 0.08, 0.06, 13.17, 0.5, fill=NAVY, radius=0.15)
    text(sl, 0.15, 0.08, 13.0, 0.46, "Marginal coverage hides conditional failure: conformal prediction intervals "
         "for county maize yield under compound drought and heat", size=15.5, bold=True, color="FFFFFF",
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    top, h = 0.66, 5.93
    xs, ws = [0.08, 3.38, 6.88, 10.08], [3.22, 3.42, 3.12, 3.17]

    # 1. Study area and data
    x, w = xs[0], ws[0]
    panel(sl, 0, x, top, w, h, "1. Study area and data (1985–2023)")
    sl.shapes.add_picture(str(parts["map"]), Inches(x + 0.15), Inches(top + 0.42), width=Inches(w - 0.3))
    text(sl, x + 0.1, top + 3.05, w - 0.2, 0.5, [[("583 counties in six US Corn Belt states", {"bold": True})],
         [("Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio", {"size": 8, "color": INK2})]], size=9.5)
    text(sl, x + 0.1, top + 3.55, w - 0.2, 0.3, "Data sources (all public)", size=11, bold=True,
         color=COL_HEAD[0])
    bullet_list(sl, x + 0.12, top + 3.88, w - 0.2, [
        ("PRISM daily climate", "Maximum and minimum temperature, precipitation"),
        ("NLDAS-2 forcing", "Radiation, humidity, wind, vapor-pressure deficit, reference ET"),
        ("NOAA Storm Events", "Hail, thunderstorm wind, tornado"),
        ("USDA NASS", "County maize grain yield"),
        ("Static descriptors", "Soil properties, land cover (NLCD), elevation and slope")],
        marker=BLUE, gap=0.4)

    # 2. Modelling framework
    x, w = xs[1], ws[1]
    panel(sl, 1, x, top, w, h, "2. Modeling framework")
    text(sl, x + 0.1, top + 0.4, w - 0.2, 0.3, "Features: 94 per county-year", size=10.5, bold=True,
         color=COL_HEAD[1])
    bullet_list(sl, x + 0.12, top + 0.7, w - 0.2, [
        ("Weather aggregates", "Season and degree-day windows (vegetative, silking, grain fill)"),
        ("Drought indices", "SPI (30, 90, 180, 365 days); SPEI (30, 90 days)"),
        ("Heat and compound drought–heat", "Days above 35 °C; compound drought–heat days"),
        ("Yield lags", "Yields of earlier seasons only (no leakage)"),
        ("Storms and static descriptors", "Storm counts; soil, land cover, terrain")],
        marker="2E7D4F", gap=0.4)
    text(sl, x + 0.1, top + 2.75, w - 0.2, 0.3, "Point models (frozen after training)", size=10.5, bold=True,
         color=COL_HEAD[1])
    bw = (w - 0.35) / 4
    for i, (m, c) in enumerate((("LightGBM", AQUA), ("CatBoost", YELLOW), ("XGBoost", "8E7CC3"),
                                ("NeuralCQR", ORANGE))):
        b = box(sl, x + 0.12 + i * (bw + 0.035), top + 3.07, bw, 0.36, fill="FFFFFF", line=c, radius=0.15)
        text(sl, x + 0.12 + i * (bw + 0.035), top + 3.07, bw, 0.36, m, size=8.6, bold=True, color=INK,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(sl, x + 0.12, top + 3.5, w - 0.24, 0.3, fill="DCEBF8", radius=0.2)
    text(sl, x + 0.12, top + 3.5, w - 0.24, 0.3, "Ensemble: NeuralCQR + LightGBM", size=9, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(sl, x + 0.12, top + 3.9, w - 0.24, 1.05, fill="F1ECF8", line="C9BDE3", radius=0.06)
    text(sl, x + 0.18, top + 3.92, w - 0.3, 1.0, [[("Six conformal calibration methods", {"bold": True,
         "size": 9.5, "color": "5B3F99"})],
         "Static split · Standard ACI · Severity-aware ACI",
         "Width-scaled CQR · Weighted conformal · Phenology-stratified CQR",
         [("Rolling origin adds group-conditional (Mondrian) calibration", {"size": 7.6, "color": INK2})]],
         size=8.2, spacing=1.05)
    box(sl, x + 0.12, top + 5.05, w - 0.24, 0.78, fill="E3F1E3", line="BFDDBF", radius=0.06)
    text(sl, x + 0.18, top + 5.07, w - 0.3, 0.75, [[("Statistical baselines", {"bold": True, "size": 9.5,
         "color": COL_HEAD[1]})], "Linear trend + county mean residual",
         "County fixed-effects degree-day panel"], size=8.2)

    # 3. Evaluation design
    x, w = xs[2], ws[2]
    panel(sl, 2, x, top, w, h, "3. Evaluation design")
    text(sl, x + 0.1, top + 0.4, w - 0.2, 0.3, "(a) Locked temporal split", size=10, bold=True)
    segs = (("FIT", "1985–2013", 0.40, "B9D7F2"), ("DEV", "2014–15", 0.18, "B5E3CC"),
            ("CAL", "2016–18", 0.19, "F8DDA6"), ("TEST", "2019–23", 0.23, "F6B9A2"))
    cx = x + 0.12
    for lab, yrs, frac, c in segs:
        sw = (w - 0.24) * frac
        box(sl, cx, top + 0.72, sw, 0.5, fill=c, shape=MSO_SHAPE.RECTANGLE)
        text(sl, cx, top + 0.72, sw, 0.5, [[(lab, {"bold": True})], yrs], size=8, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, spacing=0.9)
        cx += sw
    text(sl, x + 0.1, top + 1.32, w - 0.2, 0.3, "(b) Leave-one-state-out transfer", size=10, bold=True)
    states = ["IL", "IN", "IA", "MN", "MO", "OH"]
    sw = (w - 0.24 - 5 * 0.05) / 6
    for i, s in enumerate(states):
        held = i == 3
        box(sl, x + 0.12 + i * (sw + 0.05), top + 1.64, sw, 0.36, fill=ORANGE if held else "B9D7F2",
            radius=0.12)
        text(sl, x + 0.12 + i * (sw + 0.05), top + 1.64, sw, 0.36, s, size=8.5, bold=True,
             color="FFFFFF" if held else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, x + 0.1, top + 2.03, w - 0.2, 0.5, [[("Train", {"bold": True, "color": "1F5F9E"}),
         (" on five states, ", {}), ("test", {"bold": True, "color": "C0502A"}),
         (" on the held-out state; repeated for all six (example: MN held out)", {})]], size=7.8)
    text(sl, x + 0.1, top + 2.52, w - 0.2, 0.3, "(c) Rolling origin: 16 test years", size=10, bold=True)
    x0, x1, ty = x + 0.25, x + w - 0.25, top + 2.92
    ln = sl.shapes.add_connector(1, Inches(x0), Inches(ty), Inches(x1), Inches(ty))
    ln.line.color.rgb = rgb("1F3A5F"); ln.line.width = Pt(1.5)
    years = list(range(2008, 2024))
    for i, yv in enumerate(years):
        cxp = x0 + (x1 - x0) * i / (len(years) - 1)
        box(sl, cxp - 0.045, ty - 0.045, 0.09, 0.09, fill="1F3A5F", shape=MSO_SHAPE.OVAL)
        if yv in (2008, 2011, 2014, 2017, 2020, 2023):
            text(sl, cxp - 0.3, ty + 0.05, 0.6, 0.2, str(yv), size=7.5, align=PP_ALIGN.CENTER)
    # one origin Y (Section 3.7): fit 1985 to Y-5, DEV Y-4 and Y-3 (unused by LightGBM), calibrate Y-2 and Y-1, test Y
    segs = ((("FIT", "1985 to Y−5"), 0.40, "B9D7F2"), (("DEV", "Y−4, Y−3"), 0.21, "B5E3CC"),
            (("CAL", "Y−2, Y−1"), 0.21, "F8DDA6"), (("TEST", "Y"), 0.18, "F6B9A2"))
    cx, by = x + 0.12, ty + 0.3
    for lab, frac, c in segs:
        sw = (w - 0.24) * frac
        box(sl, cx, by, sw, 0.36, fill=c, shape=MSO_SHAPE.RECTANGLE)
        text(sl, cx, by, sw, 0.36, [[(lab[0], {"bold": True})], lab[1]], size=7.4, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, spacing=0.9)
        cx += sw
    text(sl, x + 0.1, by + 0.38, w - 0.2, 0.25, "Each test year Y (2008–2023) is one origin; DEV is not used "
         "by LightGBM", size=7.2, color=INK2, align=PP_ALIGN.CENTER)
    text(sl, x + 0.1, top + 3.88, w - 0.2, 0.3, "(d) Other analyses", size=10, bold=True)
    text(sl, x + 0.14, top + 4.13, w - 0.25, 0.8, ["• Random 70/10/20 split (point accuracy only)",
         "• Sensitivity runs to a written plan", "• Year-clustered bootstrap (2,000 resamples)",
         "• Growth-stage exposure analysis (exploratory)"], size=8.0, spacing=1.0)
    box(sl, x + 0.12, top + 4.93, w - 0.24, 0.9, fill="FFF8EC", line="E6B66A", radius=0.06)
    text(sl, x + 0.18, top + 4.95, w - 0.3, 0.88, [[("Protocol transparency", {"bold": True, "size": 9.5,
         "color": COL_HEAD[2]})],
         "• Configuration locked and checked at the start of each main run",
         "• Not pre-registered; three analyses added after the locked-window failure was seen",
         "• Sensitivity runs followed a written plan"], size=7.4, spacing=0.95)

    # 4. Coverage diagnostics
    x, w = xs[3], ws[3]
    panel(sl, 3, x, top, w, h, "4. Coverage diagnostics")
    text(sl, x + 0.1, top + 0.38, w - 0.2, 0.25, "Locked window 2019–2023: six methods", size=9, bold=True)
    sl.shapes.add_picture(str(parts["locked"]), Inches(x + 0.12), Inches(top + 0.62), width=Inches(w - 0.24), height=Inches(1.36))
    box(sl, x + 0.12, top + 2.03, w - 0.24, 0.42, fill="FFFFFF", line="E3A0A0", radius=0.1)
    text(sl, x + 0.12, top + 2.03, w - 0.24, 0.42, "All six meet 0.90 on average, yet all six exceed the 0.05 "
         "reference rate in compound drought–heat county-years", size=7.8, bold=True, color="B23A3A",
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, x + 0.1, top + 2.52, w - 0.2, 0.25, "Group-conditional calibration (16 origins)", size=9,
         bold=True)
    sl.shapes.add_picture(str(parts["gc"]), Inches(x + 0.12), Inches(top + 2.78), width=Inches(w - 0.24), height=Inches(1.18))
    v = parts["gc_vals"]
    text(sl, x + 0.1, top + 3.98, w - 0.2, 0.3, f"Cost: Normal-class coverage falls from {v['normal_st']:.3f} to "
         f"{v['normal_gc']:.3f}, below the nominal 0.90", size=7.6, color=INK2, align=PP_ALIGN.CENTER)
    text(sl, x + 0.1, top + 4.36, w - 0.2, 0.25, "Failure follows bias, not width", size=9, bold=True)
    sl.shapes.add_picture(str(parts["bw"]), Inches(x + 0.12), Inches(top + 4.6), width=Inches(w - 0.24), height=Inches(0.76))
    box(sl, x + 0.12, top + 5.42, w - 0.24, 0.42, fill="B23A3A", radius=0.12)
    text(sl, x + 0.12, top + 5.42, w - 0.24, 0.42, "Marginal coverage alone cannot show whether yield "
         "intervals are reliable", size=8.6, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)

    out = FIG / "Graphical_Abstract.pptx"
    prs.save(out)
    return out


def export(pptx_path: Path) -> None:
    """PNG and TIFF via a separate PowerPoint instance; quits only if it had nothing open."""
    import pythoncom
    import win32com.client as w32
    from PIL import Image
    pythoncom.CoInitialize()
    app = w32.DispatchEx("PowerPoint.Application")
    n0 = app.Presentations.Count
    pres = app.Presentations.Open(str(pptx_path), True, False, False)   # ReadOnly, Untitled, WithWindow
    try:
        png = FIG / "Graphical_Abstract.png"
        pres.Slides(1).Export(str(png), "PNG", 4000, 2000)
        Image.open(png).convert("RGB").save(FIG / "Graphical_Abstract.tif", compression="tiff_lzw", dpi=(300, 300))
    finally:
        pres.Close()
        if n0 == 0 and app.Presentations.Count == 0:
            app.Quit()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parts = draw_parts()
    p = build(parts)
    export(p)
    print("wrote", p, "and Graphical_Abstract.png/.tif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
