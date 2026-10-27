#!/usr/bin/env python3
"""Shared figure style: sizes, fonts and line weights.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import textwrap

TEXTWIDTH_IN = 7.0
PANEL = True
AUTOFIT = False

WIDE = dict(size=(7.0, 3.6), include=0.86)
FIG_DEFAULT = dict(size=(3.36, 2.1), include=0.48)

LINE_PT = 0.7
MARKER_PT = 2.2
EMPHASIS_ABOVE = 2.5
EMPHASIS_MULT = 2.0


FIG_CONFIG = {
    "sweep_sustained_dncc":   dict(size=(3.36, 2.1), include=0.48),
    "sweep_burst_dncc":       dict(size=(3.36, 2.1), include=0.48),
    "delay_ratios":           dict(size=(3.36, 2.1), include=0.48),
    "prereg":                 dict(size=(3.36, 2.1), include=0.48),

    "crossings":              dict(size=(3.36, 2.1), include=0.48),
    "channels_dncc":          dict(size=(3.36, 2.1), include=0.48),

    "information":            dict(size=(6.86, 2.3), include=0.98),
    "capture_curve":          dict(size=(2.52, 2.1), include=0.36),
    "survey_sweep":           dict(size=(4.34, 2.1), include=0.62),

    "budget_V_dncc":          dict(size=(3.36, 2.1), include=0.48),
    "budget_util_dncc":       dict(size=(3.36, 2.1), include=0.48),

    "protocol_matrix":        dict(size=(5.60, 2.6), include=0.80),
    "voi_curve":              dict(size=(5.60, 2.6), include=0.80),
    "channels_dncc10":        dict(size=(3.36, 2.1), include=0.48),
    "sweep_sustained_dncc10": dict(size=(3.36, 2.1), include=0.48),
    "sweep_burst_dncc10":     dict(size=(3.36, 2.1), include=0.48),
    "budget_V_dncc10":        dict(size=(3.36, 2.1), include=0.48),
    "budget_util_dncc10":     dict(size=(3.36, 2.1), include=0.48),
    "timing_dncc":            dict(size=(3.36, 2.1), include=0.48),
    "timing_dncc10":          dict(size=(3.36, 2.1), include=0.48),
    "utilisation_dncc":       dict(size=(3.36, 2.1), include=0.48),
    "utilisation_dncc10":     dict(size=(3.36, 2.1), include=0.48),
    "convergence_dncc":       dict(size=(3.36, 2.1), include=0.48),
    "convergence_dncc10":     dict(size=(3.36, 2.1), include=0.48),
}


def fig_cfg(name):
    if not PANEL:
        return dict(WIDE)
    cfg = dict(FIG_CONFIG.get(name, FIG_DEFAULT))
    if AUTOFIT:
        w, h = cfg["size"]
        want = cfg["include"] * TEXTWIDTH_IN
        cfg["size"] = (want, h * want / w)
    return cfg


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]
DASHES = [(0, ()), (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.2, 1, 1.2)),
          (0, (3, 1, 1, 1, 1, 1)), (0, (7, 2)), (0, (2, 1)),
          (0, (6, 1.5, 2, 1.5)), (0, (1, 2))]
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def apply_style(panel=True):
    base = 8.0 if panel else 9.0
    label = base
    plt.rcParams.update({
        "figure.figsize": FIG_DEFAULT["size"] if panel else WIDE["size"],
        "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.45,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "font.size": base, "axes.labelsize": label, "axes.titlesize": label,
        "xtick.labelsize": base, "ytick.labelsize": base,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1 if panel else 1.4,
        "lines.markersize": 3.2 if panel else 4.0,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False, "legend.fontsize": base - 1.0,
        "legend.handlelength": 2.2, "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.25, "legend.handletextpad": 0.5,
        "axes.prop_cycle": cycler(color=COLOURS),
    })


def panel_legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels)
    if not n:
        return None
    if not PANEL or n <= 4:
        ax._panel_legend_kw = dict(kw)
        return ax.legend(**kw)
    ncol = 2 if n <= 6 else 3
    ax._panel_legend_kw = dict(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                               ncol=ncol, **kw)
    return ax.legend(**ax._panel_legend_kw)


def panel_title(ax, text, width=34):
    if not PANEL:
        return ax.set_title(text)
    wrapped = textwrap.wrap(str(text), width=width)
    if len(wrapped) > 2:
        return None
    return ax.set_title("\n".join(wrapped))


SHORT_LABEL = {
    "objective $V$ (billions BDT, lower is better)": "$V$ (bn BDT, lower better)",
    "objective $V$, billions BDT": "$V$ (bn BDT)",
    "peak relative to a week-1 start": "peak vs week-1 start",
    "week the campaign starts": "campaign start week",
    "weeks of full coverage from week 1": "weeks at full coverage",
    "training steps (millions)": "training steps (M)",
    "gap as \\% of untreated burden": "gap, \\% untreated",
    "fraction of the gap captured": "fraction of gap captured",
    "wards in the instance": "wards",
    "surveillance cap": "surveillance cap",
}


def panel_xlabel(ax, text):
    ax.set_xlabel(SHORT_LABEL.get(text, text) if PANEL else text)


def panel_ylabel(ax, text):
    ax.set_ylabel(SHORT_LABEL.get(text, text) if PANEL else text)


def differentiate(ax, markevery=None):
    i = 0
    for ln in ax.get_lines():
        lab = ln.get_label() or ""
        if lab.startswith("_"):
            ln.set_linewidth(min(ln.get_linewidth(), 0.7))
            continue
        m = ln.get_marker()
        if m in (None, "None", "", " ", "o"):
            ln.set_marker(MARKERS[i % len(MARKERS)])
        w = ln.get_linewidth()
        ln.set_linewidth(LINE_PT * EMPHASIS_MULT if w > EMPHASIS_ABOVE
                         else LINE_PT)
        ln.set_markersize(MARKER_PT)
        n = len(ln.get_xdata())
        if markevery is not None:
            ln.set_markevery(markevery)
        elif n > 24:
            ln.set_markevery(max(1, n // 12))
        i += 1

    if ax.get_legend() is not None:
        kw = getattr(ax, "_panel_legend_kw", {})
        ax.legend(**kw)


def thin_ticks(ax, nx=5, ny=5):
    if not PANEL:
        return
    from matplotlib.ticker import MaxNLocator
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))


def save_figure(name, path):
    cfg = fig_cfg(name)
    w, h = cfg["size"]
    fig = plt.gcf()
    fig.set_size_inches(w, h)

    on_page = cfg["include"] * TEXTWIDTH_IN
    scale = on_page / w
    if abs(scale - 1.0) > 0.08:
        print(f"  NOTE {name}: drawn {w:.2f}in, placed at {on_page:.2f}in "
              f"(x{scale:.2f}); {plt.rcParams['font.size']:.1f}pt type will "
              f"render as {plt.rcParams['font.size'] * scale:.1f}pt. "
              f"Set size to ({on_page:.2f}, {h * scale:.2f}) or pass --autofit.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    return cfg["include"]


def add_geometry_args(ap):
    ap.add_argument("--wide", action="store_true",
                    help="draw at the old full text width instead of panels")
    ap.add_argument("--autofit", action="store_true",
                    help="derive each drawn width from its include width, so "
                         "every figure renders 1:1")
    return ap


def configure(args):
    global PANEL, AUTOFIT
    PANEL = not getattr(args, "wide", False)
    AUTOFIT = getattr(args, "autofit", False)
    apply_style(PANEL)
    mode = "panel" if PANEL else "wide"
    print(f"figure geometry: {mode}"
          + (", autofit on" if AUTOFIT else "")
          + f", text width {TEXTWIDTH_IN} in")
