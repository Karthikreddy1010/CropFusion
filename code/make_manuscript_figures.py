"""Render the manuscript figures from project artefacts.

Every figure reads the artefact that produced its numbers, never a markdown
summary of them, and the source of each is written to
manuscript/figures/figure_sources.json. Style follows the PaperOrchestra plotting
agent (300 dpi, exact aspect ratio, top and right spines hidden, serif type) with
the categorical palette replaced by one that passes a colour-vision-deficiency
check: the agent's default placed orange beside green at a protan separation of
4.3, which fails. Slots 2 and 3 fall below 3:1 contrast on white, so every series
carries a direct label or a legend.

Run:  python code/make_manuscript_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
for p in (str(CODE_DIR), str(CODE_DIR / "diagnostics_loso")):
    if p not in sys.path:
        sys.path.insert(0, p)

ROOT = CODE_DIR.parent
M = ROOT / "outputs_master" / "reports"
D = ROOT / "outputs_diagnostics" / "reports"
OUT = ROOT / "manuscript" / "figures"

# validated categorical slots (light surface); see module docstring
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SIZES = {"16:9": (5.5, 3.09), "4:3": (4.0, 3.0)}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,            # embed TrueType fonts in the vector files
})
SOURCES: dict = {}


def save(fig, fid: str, sources: list) -> None:
    """PNG at 300 dpi for the Word and LaTeX drafts, and a vector PDF with embedded fonts, which the
    EM&S Guide for Authors prefers for charts (a bitmap line drawing would need 1000 dpi)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fid}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{fid}.pdf", facecolor="white")
    plt.close(fig)
    SOURCES[fid] = sources
    print(f"  wrote {fid}.png/.pdf  <- {', '.join(sources)}")


def panel_label(ax, s: str) -> None:
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="right", color=INK)


# ───────────────────────────────────────────────────────────── figures

def _draw_study_area(fig, ax) -> None:
    import geopandas as gpd
    import _fold_lab as lab
    df = lab.engineered_df()
    share = (df.assign(ext=(df["Year_Type"].astype(str) == "Extreme").astype(float))
               .groupby("GEOID").ext.mean())
    counties = gpd.read_file(ROOT / "outputs_master" / "geo_cache" / "counties_fips.json")
    counties["GEOID"] = (counties["STATE"].astype(str).str.zfill(2) +
                         counties["COUNTY"].astype(str).str.zfill(3)).astype(int)
    states_fp = {"17": "Illinois", "18": "Indiana", "19": "Iowa", "27": "Minnesota",
                 "29": "Missouri", "39": "Ohio"}
    c = counties[counties["STATE"].astype(str).str.zfill(2).isin(states_fp)].copy()
    c["share"] = c["GEOID"].map(share)
    c = c.to_crs("EPSG:5070")
    st = c.dissolve(by="STATE")
    c.plot(column="share", ax=ax, cmap="Blues", vmin=0, vmax=float(np.nanmax(c["share"])),
           edgecolor="white", linewidth=0.15, missing_kwds={"color": "#eeeeec"})
    st.boundary.plot(ax=ax, color=INK2, linewidth=0.6)
    for code, name in states_fp.items():
        pt = st.loc[code].geometry.representative_point()
        ax.annotate(name, (pt.x, pt.y), ha="center", va="center", fontsize=6.3, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, float(np.nanmax(c["share"]))))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Share of seasons in the Extreme class, 1985–2023", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)


def _draw_designs(ax) -> None:
    years = np.arange(1985, 2024)
    role_col = {"FIT": BLUE, "DEV": AQUA, "CAL": YELLOW, "TEST": ORANGE}

    def band(y, a, b, role, h=0.62, label=None):
        ax.add_patch(Rectangle((a - 0.5, y - h / 2), b - a + 1, h, fc=role_col[role],
                               ec="white", lw=0.8))
        if label and b - a >= 2:
            ax.text((a + b) / 2, y, label, ha="center", va="center", fontsize=6,
                    color="white" if role in ("FIT", "TEST") else INK)

    rows = []
    y = 0
    # locked temporal
    band(y, 1985, 2013, "FIT", label="FIT 1985–2013"); band(y, 2014, 2015, "DEV", label="DEV")
    band(y, 2016, 2018, "CAL", label="CAL"); band(y, 2019, 2023, "TEST", label="TEST")
    rows.append((y, "Locked temporal\n(six states)"))
    # rolling origin, four illustrative origins of sixteen; LightGBM has fixed settings and does not use DEV
    for Y in (2008, 2012, 2018, 2023):
        y -= 0.72
        band(y, 1985, Y - 5, "FIT", h=0.5); band(y, Y - 4, Y - 3, "DEV", h=0.5)
        band(y, Y - 2, Y - 1, "CAL", h=0.5); band(y, Y, Y, "TEST", h=0.5)
        rows.append((y, f"Rolling origin, Y = {Y}"))
    ax.text(2023.9, y + 1.08, "16 origins,\nY = 2008–2023;\nDEV unused\nby LightGBM", fontsize=5.6,
            color=INK2, va="center")
    # LOSO
    y -= 1.0
    band(y, 1985, 2013, "FIT", label="FIT"); band(y, 2014, 2015, "DEV")
    band(y, 2016, 2023, "CAL", label="CAL 2016–2023")
    rows.append((y, "LOSO: five\ntraining states"))
    y -= 0.78
    band(y, 1985, 2023, "TEST", label="TEST: every year of the held-out state")
    rows.append((y, "LOSO: held-out\nstate"))
    # random: a ROW split, so every season contributes to all three partitions.
    # Colouring whole seasons would misstate the design as a year-blocked split.
    y -= 1.0
    h = 0.5
    for yr in years:
        base = y - h / 2
        for role, share in (("FIT", 0.7), ("DEV", 0.1), ("TEST", 0.2)):
            ax.add_patch(Rectangle((yr - 0.5, base), 1, h * share, fc=role_col[role], ec="white", lw=0.3))
            base += h * share
    rows.append((y, "Random 70/10/20 rows\n(interpolation only)"))

    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows], fontsize=6.3)
    ax.set_xlim(1984, 2029.5); ax.set_ylim(y - 0.6, 0.6)
    ax.set_xticks([1985, 1995, 2005, 2013, 2019, 2023])
    ax.set_xlabel("Season")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Patch(fc=role_col[r], ec="none", label=lbl) for r, lbl in
               (("FIT", "FIT: fits every learned step"),
                ("DEV", "DEV: model selection (early stopping, ensemble weight, LOSO configuration)"),
                ("CAL", "CAL: conformal calibration only"), ("TEST", "TEST: evaluation only"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.19), ncol=2,
              fontsize=6, handlelength=1.2, columnspacing=1.2)


def fig_study_area_and_designs() -> None:
    """Figure 1: (a) study area and Extreme-class share; (b) the four evaluation designs."""
    fig = plt.figure(figsize=(5.5, 6.6))
    a1 = fig.add_subplot(fig.add_gridspec(1, 1, left=0.10, right=0.86, top=0.99, bottom=0.53)[0])
    _draw_study_area(fig, a1)
    a2 = fig.add_subplot(fig.add_gridspec(1, 1, left=0.19, right=0.97, top=0.475, bottom=0.12)[0])
    _draw_designs(a2)
    fig.text(0.02, 0.985, "a", fontsize=8.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.495, "b", fontsize=8.5, fontweight="bold", color=INK, va="top")
    save(fig, "fig_study_area_and_designs",
         ["engineered frame via code/diagnostics_loso/_fold_lab.py (Year_Type)",
          "outputs_master/geo_cache/counties_fips.json",
          "outputs_master/splits/temporal_split_assignments.csv",
          "code/rolling_origin.py (YearWindow)", "code/main.py (_run_loso_cv)"])


def fig_rolling_origin_miss_rate() -> None:
    """Figure 3: per-origin unsafe miss rate (Extreme and Normal classes, static split), with the pooled and
    year-weighted Extreme-class rates and their year-clustered intervals. Per-origin rates have no interval:
    each origin is a single test year, so the year-clustered bootstrap cannot give one."""
    ro = pd.read_csv(D / "rolling_origin_raw.csv")
    st = ro[ro.method == "rolling_static_cp"]
    ext = st[st.stratum == "Extreme"].sort_values("test_year")
    nor = st[st.stratum == "Normal"].sort_values("test_year")
    rs = pd.read_csv(D / "rolling_origin_summary.csv")
    pooled = rs[(rs.method == "rolling_static_cp") & (rs.stratum == "Extreme")].iloc[0]
    yw = json.loads((D / "audit_checks.json").read_text(encoding="utf-8"))[
        "review_2026_09_26"]["rolling_extreme_unsafe_year_weighted"]

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.axvspan(2018.5, 2023.5, color=GRID, alpha=0.35, lw=0)
    ax.text(2021, 0.425, "locked test years", ha="center", fontsize=5.8, color=INK2)
    ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
    ax.text(2015.5, 0.056, "reference 0.05", fontsize=5.8, color=INK2, va="bottom")
    ax.plot(nor.test_year, nor.unsafe_miss_rate, "s", ms=3.6, mfc="white", mec=INK2, mew=0.8,
            label="Normal class", zorder=3)
    ax.scatter(ext.test_year, ext.unsafe_miss_rate, s=6 + ext.n * 0.55, color=ORANGE, edgecolor="white",
               linewidth=0.5, zorder=4, label="Extreme class")
    for _, r in ext[ext.n >= 80].iterrows():
        low = r.unsafe_miss_rate < 0.06                   # below the marker, clear of the reference line
        ax.annotate(f"n = {int(r.n)}", (r.test_year, r.unsafe_miss_rate), xytext=(0, -9 if low else 9),
                    textcoords="offset points", ha="center", va="top" if low else "baseline",
                    fontsize=5.6, color=INK)
    for yr in sorted(set(nor.test_year) - set(ext.test_year)):
        ax.text(yr, -0.03, "none", ha="center", fontsize=5.2, color=INK2)
    # pooled and year-weighted Extreme-class rates with year-clustered 95% intervals
    xs = {"Pooled": 2025.3, "Year-\nweighted": 2027.0}
    vals = ((pooled.unsafe_weighted, pooled.unsafe_ci_lo, pooled.unsafe_ci_hi),
            (yw["year_weighted_mean"], *yw["year_weighted_ci95"]))
    for x, (v, lo, hi) in zip(xs.values(), vals):
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="D", color=ORANGE, ms=4.2, mec="white", mew=0.5,
                    elinewidth=1.0, capsize=2.2, zorder=4)
        ax.text(x + 0.25, v, f"{v:.3f}", fontsize=5.8, color=INK, va="center")
    ax.axvline(2024.3, color=INK2, lw=0.5)
    ax.text(2026.15, 0.425, "Extreme class,\nall origins", ha="center", va="center", fontsize=5.8, color=INK2)
    ax.set_xticks(list(range(2008, 2024, 2)) + list(xs.values()))
    ax.set_xticklabels([str(y) for y in range(2008, 2024, 2)] + list(xs.keys()), fontsize=6.3)
    ax.set_xlim(2007.2, 2028.4); ax.set_ylim(-0.045, 0.45)
    ax.set_xlabel("Rolling-origin test year")
    ax.set_ylabel("Unsafe miss rate")
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.30, 0.86), fontsize=6.0, markerscale=0.7)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_rolling_origin_miss_rate",
         ["outputs_diagnostics/reports/rolling_origin_raw.csv", "outputs_diagnostics/reports/rolling_origin_summary.csv",
          "outputs_diagnostics/reports/audit_checks.json (review_2026_09_26)"])


def fig_point_accuracy_by_origin() -> None:
    fo = pd.read_csv(D / "fe_degree_day_baseline_by_origin.csv")
    ro = pd.read_csv(D / "rolling_origin_raw.csv")
    a = ro[(ro.stratum == "ALL") & (~ro.method.isin(["_group_cp_meta", "_error"]))].drop_duplicates("test_year")
    sw = pd.read_csv(D / "rolling_origin_seed_sweep_by_origin.csv")
    j = (a[["test_year", "model_r2", "baseline_trend_county_r2"]]
         .merge(fo[["test_year", "fe_r2"]], on="test_year").merge(sw[["test_year", "r2_min", "r2_max"]], on="test_year"))
    fig, ax = plt.subplots(figsize=SIZES["16:9"])
    ax.axhline(0, color=INK2, lw=0.5)
    ax.fill_between(j.test_year, j.r2_min, j.r2_max, color=BLUE, alpha=0.15, lw=0)
    for col, colr, lab, mk in (("model_r2", BLUE, "LightGBM (seed 42; band = five-seed range)", "o"),
                               ("fe_r2", ORANGE, "County fixed-effects degree-day panel", "s"),
                               ("baseline_trend_county_r2", AQUA, "Trend + county mean", "^")):
        ax.plot(j.test_year, j[col], color=colr, lw=1.4, marker=mk, ms=3.6, mec="white", mew=0.5, label=lab)
    r12 = j[j.test_year == 2012].iloc[0]
    ax.annotate("2012", (2012, r12.baseline_trend_county_r2), xytext=(2013.3, -1.0), fontsize=6.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_xlabel("Rolling-origin test year"); ax.set_ylabel("Test R$^2$ (yield, t ha$^{-1}$)")
    ax.set_xticks(range(2008, 2024, 2)); ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="lower right", fontsize=6.3)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_point_accuracy_by_origin",
         ["outputs_diagnostics/reports/rolling_origin_raw.csv", "outputs_diagnostics/reports/fe_degree_day_baseline_by_origin.csv",
          "outputs_diagnostics/reports/rolling_origin_seed_sweep_by_origin.csv"])


def fig_marginal_vs_extreme_coverage() -> None:
    cc = pd.read_csv(M / "conformal_comparison.csv")
    di = pd.read_csv(D / "decision_impact_by_stratum.csv")
    key = {"Static Conformal": "static_conformal", "Standard ACI": "standard_aci", "SA-ACI": "sa_aci",
           "Phenology-Stratified CQR": "phenology_stratified_cqr",
           "Weighted Conformal": "weighted_conformal", "Locally Adaptive": "locally_adaptive"}
    label = {"Static Conformal": "Static split", "Standard ACI": "Standard ACI", "SA-ACI": "Severity-aware ACI",
             "Phenology-Stratified CQR": "Phenology-stratified CQR", "Weighted Conformal": "Weighted",
             "Locally Adaptive": "Width-scaled CQR"}
    ex = di[di.Year_Type == "Extreme"].set_index("Conformal_Method")
    t = pd.DataFrame([{"m": label[k], "picp": float(cc[cc.Method == k].Coverage_PICP.iloc[0]),
                       "ace": float(cc[cc.Method == k].ACE.iloc[0]),
                       "umr": float(ex.loc[v, "unsafe_miss_rate"])} for k, v in key.items()])
    t = t.sort_values("ace", ascending=False).reset_index(drop=True)   # smallest ACE at the top
    y = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZES["16:9"], sharey=True, gridspec_kw={"wspace": 0.08})
    a1.axvline(0.90, color=INK2, lw=0.8, ls="--")
    a1.text(0.9005, len(t) - 0.35, "nominal 0.90", fontsize=6, color=INK2, va="bottom")
    a1.hlines(y, 0.90, t.picp, color=BLUE, lw=1.2, alpha=0.5)
    a1.plot(t.picp, y, "o", color=BLUE, ms=5.5, mec="white", mew=0.6)
    for yi, v in zip(y, t.picp):
        a1.text(v + 0.0015, yi + 0.18, f"{v:.3f}", fontsize=6, color=INK)
    a1.set_xlim(0.885, 0.965); a1.set_xlabel("Marginal PICP, all county-years")
    a1.set_yticks(y); a1.set_yticklabels(t.m); a1.set_title("Marginal: all six reach 0.90", loc="left")
    a2.axvline(0.05, color=INK2, lw=0.8, ls="--")
    a2.text(0.0505, len(t) - 0.35, "reference 0.05", fontsize=6, color=INK2, va="bottom")
    a2.hlines(y, 0.05, t.umr, color=ORANGE, lw=1.2, alpha=0.5)
    a2.plot(t.umr, y, "D", color=ORANGE, ms=4.8, mec="white", mew=0.6)
    for yi, v in zip(y, t.umr):
        a2.text(v + 0.002, yi + 0.18, f"{v:.3f}", fontsize=6, color=INK)
    a2.set_xlim(0.0, 0.16); a2.set_xlabel("Unsafe miss rate, Extreme class")
    a2.set_title("Extreme class: all six exceed it", loc="left")
    a2.tick_params(axis="y", length=0)
    for a in (a1, a2):
        a.grid(axis="x", alpha=0.6); a.set_ylim(-0.6, len(t) - 0.1)
    panel_label(a1, "a"); panel_label(a2, "b")
    # explicit margins: tight_layout clipped the longest method names
    fig.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.15, wspace=0.08)
    save(fig, "fig_marginal_vs_extreme_coverage",
         ["outputs_master/reports/conformal_comparison.csv", "outputs_diagnostics/reports/decision_impact_by_stratum.csv"])


def fig_rolling_coverage_by_class() -> None:
    rs = pd.read_csv(D / "rolling_origin_summary.csv")
    classes = ["Normal", "Moderate", "Extreme"]
    meth = [("rolling_static_cp", "Static split", BLUE, "o", -0.13),
            ("group_conditional_cp", "Group-conditional (Mondrian)", ORANGE, "D", 0.13)]
    x = np.arange(len(classes))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZES["16:9"], gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.32})
    a1.axhline(0.90, color=INK2, lw=0.8, ls="--"); a1.text(2.45, 0.902, "nominal", fontsize=6, color=INK2)
    for m, lab, colr, mk, dx in meth:
        s = rs[rs.method == m].set_index("stratum").loc[classes]
        err = np.vstack([s.picp_weighted - s.picp_ci_lo, s.picp_ci_hi - s.picp_weighted])
        a1.errorbar(x + dx, s.picp_weighted, yerr=err, fmt=mk, color=colr, ms=5, mec="white", mew=0.6,
                    elinewidth=1.0, capsize=2.2, label=lab)
        # static labels to the left of their point, group-conditional to the right,
        # so equal values (Moderate, 0.914 and 0.914) cannot collide
        side, ha = (-0.05, "right") if dx < 0 else (0.05, "left")
        for xi, v in zip(x + dx, s.picp_weighted):
            a1.text(xi + side, v, f"{v:.3f}", fontsize=5.8, color=INK, va="center", ha=ha)
    a1.set_xticks(x); a1.set_xticklabels(classes); a1.set_ylabel("PICP (95% year-clustered interval)")
    a1.set_ylim(0.62, 1.0); a1.set_xlim(-0.5, 2.8); a1.grid(axis="y", alpha=0.6)
    a1.legend(loc="lower left", fontsize=6.2)
    a1.set_title("Coverage by exposure class", loc="left")
    w = 0.34
    for i, (m, lab, colr, _, _) in enumerate(meth):
        s = rs[rs.method == m].set_index("stratum").loc[classes]
        b = a2.bar(x + (i - 0.5) * w, s.mpiw_weighted, w - 0.03, color=colr, label=lab)
        for rect, v in zip(b, s.mpiw_weighted):
            a2.text(rect.get_x() + rect.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=5.8, color=INK)
    a2.set_xticks(x); a2.set_xticklabels(classes); a2.set_ylabel("Mean interval width (t ha$^{-1}$)")
    a2.set_ylim(0, 6.0); a2.grid(axis="y", alpha=0.6); a2.set_title("Width moved between classes", loc="left")
    a2.legend(loc="upper left", fontsize=6.0)          # panel b must read on its own
    panel_label(a1, "a"); panel_label(a2, "b")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_rolling_coverage_by_class", ["outputs_diagnostics/reports/rolling_origin_summary.csv"])


def fig_bias_not_width() -> None:
    lp = pd.read_csv(ROOT / "outputs_master" / "predictions" / "loso_predictions.csv")
    lp["err"] = lp.y_true - lp.y_pred
    lp["width"] = lp.upper_bound - lp.lower_bound
    cell = lp.groupby(["held_out_state", "year"]).agg(bias=("err", "mean"), width=("width", "mean"),
                                                        coverage=("covered", "mean")).reset_index()
    cell["abs_bias"] = cell.bias.abs()
    ro_ = json.loads((D / "objectives_RO_report.json").read_text(encoding="utf-8"))["RO3"]
    fig, axes = plt.subplots(1, 3, figsize=SIZES["16:9"], gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.45})
    for ax, col, xl, r in ((axes[0], "abs_bias", "|mean error| (t ha$^{-1}$)", ro_["corr_abs_bias_vs_coverage"]),
                           (axes[1], "width", "Mean interval width (t ha$^{-1}$)", ro_["corr_width_vs_coverage"])):
        ax.scatter(cell[col], cell.coverage, s=7, color=BLUE, alpha=0.55, lw=0)
        ax.set_xlabel(xl); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.5)
        ax.text(0.97, 0.05, f"r = {r:.3f}", transform=ax.transAxes, ha="right", fontsize=6.8, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    axes[0].set_ylabel("Coverage, per LOSO state-year"); axes[1].tick_params(labelleft=False)
    axes[0].set_title("Bias", loc="left"); axes[1].set_title("Width", loc="left")
    c12 = ro_["case_2012"]["picp_by_method"]
    names = {"rolling_static_cp": "Static", "standard_aci": "ACI", "sa_aci": "SA-ACI",
             "group_conditional_cp": "Group-cond."}
    order = ["rolling_static_cp", "standard_aci", "sa_aci", "group_conditional_cp"]
    b = axes[2].bar(range(4), [c12[k] for k in order], 0.62,
                    color=[BLUE, BLUE, BLUE, ORANGE])
    for rect, k in zip(b, order):
        axes[2].text(rect.get_x() + rect.get_width() / 2, c12[k] + 0.02, f"{c12[k]:.3f}", ha="center", fontsize=5.6)
    axes[2].axhline(0.90, color=INK2, lw=0.8, ls="--")
    axes[2].text(3.45, 0.915, "nominal", fontsize=5.6, color=INK2, ha="right")
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels([names[k] for k in order], fontsize=5.6,
                                                          rotation=30, ha="right")
    axes[2].set_ylim(0, 1.0); axes[2].set_ylabel("Extreme-class PICP")
    axes[2].set_title("2012, Extreme class", loc="left")
    for a, s in zip(axes, "abc"):
        panel_label(a, s)
    # explicit margins: tight_layout clipped the x-axis labels of panels a and b
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.2, wspace=0.45)
    save(fig, "fig_bias_not_width", ["outputs_master/predictions/loso_predictions.csv",
                                     "outputs_diagnostics/reports/objectives_RO_report.json (RO3)"])


def fig_exposure_attribution() -> None:
    ph = json.loads((D / "phenology_coverage.json").read_text(encoding="utf-8"))
    rows = ph["by_exposure"]
    order = {"vegetative": 0, "silking": 1, "grain fill": 2, "season": 3}
    rows = sorted(rows, key=lambda r: (order[r["stage"]], r["label"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK2, lw=0.7)
    for yi, r in zip(y, rows):
        lo, hi = r["ci_95"]
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=1.1)
        sig = r["excludes_zero"]
        ax.plot(r["excess_unsafe"], yi, "o", ms=4.6, color=BLUE, mfc=BLUE if sig else "white", mew=1.0)
    labels = []
    for r in rows:
        base = r["label"].split(",")[0].replace("days >35C", "Days > 35 °C").replace("CDHW", "Compound")
        base = base.replace("SPEI-90 minimum (drought)", "SPEI-90 minimum").replace("season precipitation", "Precipitation")
        # the stratifier counts compound drought-heat days (cdhw_total_days_gs), not multi-day events
        base = base.replace("Compound event count (the paper's stratifier)", "Compound-day count")
        stage = {"vegetative": "vegetative window (<700 GDD)", "silking": "silking window (700–1400 GDD)",
                 "grain fill": "grain-fill window (>1400 GDD)", "season": "whole season"}[r["stage"]]
        labels.append(f"{base} — {stage}" + (" (lower tail)" if r.get("stressed_tail") == "lower" else ""))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlabel("Excess unsafe miss rate in the stressed tail, with 95% interval")
    for i in range(len(rows) - 1):                     # thin rules between growth stages
        if rows[i]["stage"] != rows[i + 1]["stage"]:
            ax.axhline(y[i] - 0.5, color=GRID, lw=0.6)
    ax.grid(axis="x", alpha=0.5); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.plot([], [], "o", color=BLUE, ms=4.6, label="interval excludes zero")
    ax.plot([], [], "o", color=BLUE, mfc="white", ms=4.6, label="interval includes zero")
    ax.legend(loc="upper right", fontsize=5.8)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_exposure_attribution", ["outputs_diagnostics/reports/phenology_coverage.json"])


def graphical_abstract() -> None:
    """Graphical abstract: 13 x 5.2 cm (the Guide asks for >= 531 x 1328 px, readable at 5 x 13 cm).
    Same sources as Figures 4 and 5; no number appears here that is not in those figures."""
    cc = pd.read_csv(M / "conformal_comparison.csv")
    di = pd.read_csv(D / "decision_impact_by_stratum.csv")
    rs = pd.read_csv(D / "rolling_origin_summary.csv")
    methods = ["Static Conformal", "Standard ACI", "SA-ACI", "Phenology-Stratified CQR", "Weighted Conformal",
               "Locally Adaptive"]
    key = {"Static Conformal": "static_conformal", "Standard ACI": "standard_aci", "SA-ACI": "sa_aci",
           "Phenology-Stratified CQR": "phenology_stratified_cqr", "Weighted Conformal": "weighted_conformal",
           "Locally Adaptive": "locally_adaptive"}
    ex = di[di.Year_Type == "Extreme"].set_index("Conformal_Method")
    picp = [float(cc[cc.Method == m].Coverage_PICP.iloc[0]) for m in methods]
    umr = [float(ex.loc[key[m], "unsafe_miss_rate"]) for m in methods]
    fig = plt.figure(figsize=(13 / 2.54, 5.2 / 2.54))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.70, bottom=0.2, wspace=0.55)
    fig.text(0.5, 0.955, "Conformal yield intervals: valid on average, unreliable in drought–heat seasons",
             ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    fig.text(0.5, 0.855, "583 US Corn Belt counties, 1985–2023 · six conformal methods · "
             "locked test, leave-one-state-out, 16 rolling origins", ha="center", va="top", fontsize=5.6, color=INK2)
    y = np.arange(len(methods))
    a1 = fig.add_subplot(gs[0])
    a1.axvline(0.90, color=INK2, lw=0.7, ls="--")
    a1.plot(picp, y, "o", color=BLUE, ms=3.6, mec="white", mew=0.4)
    a1.set_xlim(0.88, 0.96); a1.set_yticks([]); a1.set_xticks([0.90, 0.92, 0.94, 0.96])
    a1.set_xlabel("Coverage, all county-years", fontsize=5.6, labelpad=1.5)
    a1.set_title("1. All six meet 0.90 on average", loc="left", fontsize=6.2)
    a2 = fig.add_subplot(gs[1])
    a2.axvline(0.05, color=INK2, lw=0.7, ls="--")
    a2.plot(umr, y, "D", color=ORANGE, ms=3.2, mec="white", mew=0.4)
    a2.set_xlim(0.0, 0.16); a2.set_yticks([]); a2.set_xticks([0, 0.05, 0.10, 0.15])
    a2.set_xlabel("Share below lower bound, Extreme class", fontsize=5.6, labelpad=1.5)
    a2.set_title("2. All six exceed the 0.05 reference", loc="left", fontsize=6.2)
    a3 = fig.add_subplot(gs[2])
    classes = ["Normal", "Extreme"]
    for dx, (m, lab, colr, mk) in zip((-0.12, 0.12), (("rolling_static_cp", "Static", BLUE, "o"),
                                                       ("group_conditional_cp", "Group-conditional", ORANGE, "D"))):
        s = rs[rs.method == m].set_index("stratum").loc[classes]
        err = np.vstack([s.picp_weighted - s.picp_ci_lo, s.picp_ci_hi - s.picp_weighted])
        a3.errorbar(np.arange(2) + dx, s.picp_weighted, yerr=err, fmt=mk, color=colr, ms=3.2, mec="white",
                    mew=0.4, elinewidth=0.8, capsize=1.5, label=lab)
        for xi, v in zip(np.arange(2) + dx, s.picp_weighted):
            yv = v + (0.018 if abs(v - 0.90) < 0.012 else 0.0)   # keep labels off the nominal line
            a3.text(xi + (-0.06 if dx < 0 else 0.06), yv, f"{v:.3f}", fontsize=4.8, color=INK, va="center",
                    ha="right" if dx < 0 else "left")
    a3.axhline(0.90, color=INK2, lw=0.7, ls="--")
    a3.set_xticks([0, 1]); a3.set_xticklabels(classes); a3.set_xlim(-0.85, 1.6); a3.set_ylim(0.62, 1.0)
    a3.set_ylabel("Coverage, 16 origins", fontsize=5.6, labelpad=1.5)
    a3.set_title("3. Group-conditional: partial fix", loc="left", fontsize=6.2)
    a3.legend(loc="lower left", fontsize=4.8, handletextpad=0.3, borderaxespad=0.2)
    for a in (a1, a2, a3):
        a.tick_params(labelsize=5.2, length=2, pad=1.5)
        a.grid(axis="x" if a is not a3 else "y", alpha=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(OUT / "graphical_abstract.png", dpi=600, facecolor="white")
    fig.savefig(OUT / "graphical_abstract.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    SOURCES["graphical_abstract"] = ["outputs_master/reports/conformal_comparison.csv",
                                     "outputs_diagnostics/reports/decision_impact_by_stratum.csv",
                                     "outputs_diagnostics/reports/rolling_origin_summary.csv"]
    print("  wrote graphical_abstract.pdf/.png/.tif")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for fn in (fig_study_area_and_designs, fig_point_accuracy_by_origin, fig_marginal_vs_extreme_coverage,
               fig_rolling_origin_miss_rate, fig_rolling_coverage_by_class, fig_bias_not_width,
               fig_exposure_attribution, graphical_abstract):
        fn()
    (OUT / "figure_sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
