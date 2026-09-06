#!/usr/bin/env python
"""Generate the proxy--repair mismatch overview used by the manuscript.

v3 redesign: every box is sized from the measured pixel extent of its text,
including column headers, evaluation labels, and the headline.  Safe margins
and adaptive typography keep edge annotations inside the canvas in both
languages.  Layout follows an ICLR-style aesthetic: Arial/Helvetica type,
flat muted fills with saturated edges, rounded corners, uniform stage heights,
and column-aligned comparison rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#23282D"
MUTED = "#69737C"
LINE = "#B9C1C8"
PANEL = "#F7F8F9"
BLUE = "#277DA1"
AMBER = "#DD9C27"
GREEN = "#3A7D44"
CORAL = "#C2543A"
ARROW_C = "#707A83"

TOKEN_STEP = 0.0155
N_TOKENS = 4


TEXT = {
    "zh": {
        "noisy": "带噪训练集",
        "observables": "查询前可观测量",
        "observable_list": "loss · entropy · posterior\n训练动态 · 表示覆盖 · influence",
        "ranking": "冻结排序",
        "query": "预算查询集  Q",
        "verify": "人工核验并修正",
        "retrain": "同一初始化\n完整重训",
        "evaluate": "隐藏组离线评估",
        "delta": "修复效用  ΔWGA",
        "legal": "选择阶段：不可访问组标签",
        "private": "评估阶段：组标签仅用于计算 WGA",
        "proxy_title": "查询前代理",
        "repair_title": "重训后修复价值",
        "set_a": "查询集 A",
        "set_b": "查询集 B",
        "score": "检测精度 / 验证代理",
        "weak_low": "弱组修正少",
        "weak_high": "弱组修正多",
        "low_value": "较低/负 ΔWGA",
        "high_value": "较高 ΔWGA",
        "weak_col": "隐藏弱组组成",
    },
    "en": {
        "noisy": "Noisy\ntraining set",
        "observables": "Pre-query signals",
        "observable_list": "loss · entropy · posterior\ndynamics · coverage · influence",
        "ranking": "Frozen ranking",
        "query": "Budgeted\nquery set  Q",
        "verify": "Verify labels\n& correct",
        "retrain": "Full retraining\n(shared init.)",
        "evaluate": "Group evaluation",
        "delta": "Repair utility  ΔWGA",
        "legal": "Acquisition: no group labels",
        "private": "Evaluation only: groups compute WGA",
        "proxy_title": "Query-time proxy",
        "repair_title": "Post-retrain ΔWGA",
        "set_a": "Query set A",
        "set_b": "Query set B",
        "score": "Detection / validation score",
        "weak_low": "Fewer weak-group fixes",
        "weak_high": "More weak-group fixes",
        "low_value": "Low / negative ΔWGA",
        "high_value": "Higher ΔWGA",
        "weak_col": "Weak-group fixes",
        "headline_left": "Proxy improvement",
        "headline_right": "certified WGA repair",
    },
}

ZH_EXTRAS = {"headline_left": "代理改善", "headline_right": "最差组修复证书"}


class Meter:
    """Measure rendered text sizes in axes data coordinates."""

    def __init__(self, ax: plt.Axes) -> None:
        self.ax = ax
        self.fig = ax.figure

    def __call__(self, s: str, fontsize: float, weight: str = "normal") -> tuple[float, float]:
        probe = self.ax.text(
            0.0,
            -10.0,
            s,
            fontsize=fontsize,
            weight=weight,
            ha="left",
            va="baseline",
            linespacing=1.25,
        )
        self.fig.canvas.draw()
        bb = probe.get_window_extent(renderer=self.fig.canvas.get_renderer())
        inv = self.ax.transData.inverted()
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        probe.remove()
        return abs(x1 - x0), abs(y1 - y0)


def rounded_box(ax, xy, w, h, *, fc, ec, lw=1.0, radius=0.012, zorder=1) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=zorder,
        )
    )


def arrow(ax, start, end, color=ARROW_C, lw=1.0, scale=8.0, zorder=2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            capstyle="butt",
            zorder=zorder,
        )
    )


def draw_dataset_icon(ax, cx: float, cy: float, w: float = 0.10, h: float = 0.11) -> None:
    """Stacked-document icon centred at (cx, cy)."""
    sheet = w * 0.66
    sheet_h = h * 0.62
    x0 = cx - sheet / 2 - 0.014
    y0 = cy - sheet_h / 2
    layers = [
        (0.024, "#DCEAF1"),
        (0.012, "#F3E7CB"),
        (0.0, "#FFFFFF"),
    ]
    for off, color in layers:
        ax.add_patch(
            Rectangle(
                (x0 + off, y0 + off),
                sheet,
                sheet_h,
                facecolor=color,
                edgecolor=MUTED,
                linewidth=0.7,
                zorder=3,
            )
        )
    for frac in (0.72, 0.48, 0.24):
        ax.plot(
            [x0 + sheet * 0.15, x0 + sheet * 0.85],
            [y0 + sheet_h * frac, y0 + sheet_h * frac],
            color=MUTED,
            lw=0.65,
            zorder=4,
        )


def draw_query_tokens(ax, x_left: float, cy: float, step: float = TOKEN_STEP, size: int = 26) -> None:
    colors = [BLUE, AMBER, BLUE, CORAL]
    for i, color in enumerate(colors):
        ax.scatter(
            x_left + step * i,
            cy + (0.007 if i % 2 else -0.002),
            s=size,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )


TOKEN_SPAN = TOKEN_STEP * (N_TOKENS - 1)


def draw_bracket(ax, x0: float, x1: float, y: float, color: str, tick: float = 0.010) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
    for xx in (x0, x1):
        ax.plot([xx, xx], [y, y - tick], color=color, lw=2.0, solid_capstyle="round", zorder=2)


def draw_pipeline(ax, m: Meter, lb: dict[str, str]) -> None:
    # Keep a visible safety margin around every annotation.  The previous
    # layout relied on ``bbox_inches='tight'`` to remove whitespace, which
    # made labels at the right edge look clipped after rasterization.
    L, R = 0.045, 0.955
    top, bot = 0.885, 0.55
    h = top - bot
    gap_min = 0.030
    n_stages = 5

    fills = [
        (PANEL, INK),
        ("#EBF2F7", BLUE),
        ("#FBF2DC", AMBER),
        ("#F9ECE8", CORAL),
        ("#EAF1EC", GREEN),
    ]

    # Measure every stage; shrink all type together until the row fits.
    f = 1.0
    while True:
        title_fs = 8.3 * f
        content_fs = 6.5 * f
        widths = []
        for i, key in enumerate(["noisy", "observables", "query", "verify", "retrain"]):
            tw, _ = m(lb[key], title_fs, weight="bold")
            cw = 0.0
            if key == "observables":
                cw, _ = m(lb["observable_list"], content_fs)
            elif key == "query":
                cw = max(TOKEN_SPAN, m(lb["ranking"], 6.4 * f)[0])
            elif key == "verify":
                cw, _ = m(r"$\tilde{y}_i\;\rightarrow\;y_i$", 10.5 * f)
            elif key == "retrain":
                cw, _ = m("$f_Q$", 11.5 * f, weight="bold")
            widths.append(max(tw, cw) + 0.046)
        total = sum(widths)
        if total + gap_min * (n_stages - 1) <= (R - L) or f <= 0.62:
            break
        f -= 0.03

    gap = max(gap_min, ((R - L) - total) / (n_stages - 1))
    used = total + gap * (n_stages - 1)
    x = L + ((R - L) - used) / 2

    keys = ["noisy", "observables", "query", "verify", "retrain"]
    title_fs = 8.3 * f
    specs = []
    for i, (key, w) in enumerate(zip(keys, widths)):
        tw, th = m(lb[key], title_fs, weight="bold")
        fc, ec = fills[i]
        rounded_box(ax, (x, bot), w, h, fc=fc, ec=ec, lw=1.15, radius=0.014, zorder=1)
        cx = x + w / 2
        ax.text(
            cx,
            top - 0.026 - th / 2,
            lb[key],
            ha="center",
            va="center",
            color=INK,
            fontsize=title_fs,
            weight="bold",
            linespacing=1.25,
            zorder=3,
        )
        mid_y = bot + (h - th) / 2 - 0.008
        if key == "noisy":
            draw_dataset_icon(ax, cx, mid_y)
        elif key == "observables":
            ax.text(cx, mid_y, lb["observable_list"], ha="center", va="center",
                    color=MUTED, fontsize=6.5 * f, linespacing=1.5, zorder=3)
        elif key == "query":
            draw_query_tokens(ax, cx - TOKEN_SPAN / 2, mid_y + 0.014)
            ax.text(cx, mid_y - 0.027, lb["ranking"], ha="center", va="center",
                    color=MUTED, fontsize=6.4 * f, zorder=3)
        elif key == "verify":
            ax.text(cx, mid_y + 0.004, r"$\tilde{y}_i\;\rightarrow\;y_i$",
                    ha="center", va="center", color=CORAL, fontsize=10.5 * f, zorder=3)
        else:
            ax.text(cx, mid_y + 0.004, "$f_Q$", ha="center", va="center",
                    color=GREEN, fontsize=11.5 * f, weight="bold", zorder=3)
        specs.append({"key": key, "cx": cx, "left": x, "right": x + w})
        x += w + gap

    # --- flow arrows between adjacent stages -------------------------------
    cursor = specs[0]["right"]
    for spec in specs[1:]:
        arrow(ax, (cursor + 0.005, bot + h / 2), (cursor + gap - 0.005, bot + h / 2))
        cursor += gap + (spec["right"] - spec["left"])

    # --- downstream evaluation branch --------------------------------------
    cx5 = specs[-1]["cx"]
    delta_fs = 8.0
    ev_text_w = max(m(lb["evaluate"], 7.4)[0], m(lb["delta"], delta_fs, weight="bold")[0])
    ev_w, ev_h = ev_text_w + 0.050, 0.062
    ev_top = 0.460
    ev_bot = ev_top - ev_h
    arrow(ax, (cx5, bot - 0.004), (cx5, ev_top + 0.004))
    rounded_box(ax, (cx5 - ev_w / 2, ev_bot), ev_w, ev_h, fc="#F1F2F3", ec=MUTED,
                lw=0.95, radius=0.012)
    ax.text(cx5, ev_bot + ev_h / 2, lb["evaluate"], ha="center", va="center",
            color=INK, fontsize=7.4, zorder=3)
    ax.text(cx5, ev_bot - 0.027, lb["delta"], ha="center", va="center",
            color=GREEN, fontsize=delta_fs, weight="bold", zorder=3)

    # --- phase brackets -----------------------------------------------------
    by = 0.915
    label_fs = 7.4

    draw_bracket(ax, specs[0]["left"], specs[3]["right"], by, BLUE)
    draw_bracket(ax, specs[4]["left"], specs[4]["right"], by, MUTED)

    def centered_label(text, target_cx, color):
        tw, _ = m(text, label_fs)
        cx = min(max(target_cx, L + tw / 2 + 0.005), R - tw / 2 - 0.005)
        ax.text(cx, by + 0.031, text, ha="center", va="center", color=color, fontsize=label_fs)

    centered_label(lb["legal"], (specs[0]["left"] + specs[3]["right"]) / 2, BLUE)
    centered_label(lb["private"], cx5, MUTED)


def draw_mismatch(ax, m: Meter, lb: dict[str, str]) -> None:
    extras = ZH_EXTRAS if lb["set_a"].startswith("查询集") else None
    P_L, P_R = 0.045, 0.955
    P_T, P_B = 0.330, 0.042
    rounded_box(ax, (P_L, P_B), P_R - P_L, P_T - P_B, fc="#FBFCFC", ec=LINE, lw=0.95,
                radius=0.016, zorder=0)

    # ---- headline -----------------------------------------------------------
    hy = P_T - 0.028
    if lb["set_a"].startswith("查询集"):
        left_txt, right_txt = ZH_EXTRAS["headline_left"], ZH_EXTRAS["headline_right"]
    else:
        left_txt, right_txt = lb["headline_left"], lb["headline_right"]
    # Size the headline to the available width before drawing it.  This is
    # especially important for the longer English phrase on narrow exports.
    headline_fs = 9.0
    while headline_fs > 7.0:
        hw0 = m(left_txt, headline_fs, weight="bold")[0]
        hwm = m("≠", headline_fs + 1.0)[0]
        hw1 = m(right_txt, headline_fs, weight="bold")[0]
        if hw0 + hwm + hw1 + 2 * 0.014 <= (P_R - P_L) - 0.04:
            break
        headline_fs -= 0.2
    hw0 = m(left_txt, headline_fs, weight="bold")[0]
    hwm = m("≠", headline_fs + 1.0)[0]
    hw1 = m(right_txt, headline_fs, weight="bold")[0]
    sep = 0.014
    total = hw0 + hwm + hw1 + 2 * sep
    x = (P_L + P_R) / 2 - total / 2
    ax.text(x, hy, left_txt, ha="left", va="center", color=CORAL, fontsize=headline_fs,
            weight="bold", zorder=3)
    ax.text(x + hw0 + sep, hy, "≠", ha="left", va="center", color=INK, fontsize=headline_fs + 1.0,
            weight="bold", zorder=3)
    ax.text(x + hw0 + hwm + 2 * sep, hy, right_txt, ha="left", va="center", color=GREEN,
            fontsize=headline_fs, weight="bold", zorder=3)

    # ---- shared column layout ------------------------------------------------
    y_headers = hy - 0.054
    pad_x, pad_y = 0.017, 0.011
    margin = 0.035

    cells_fs = 6.8
    header_fs = 7.0
    f_cell = 1.0
    while True:
        lab_w = max(m(lb["set_a"], 8.0, weight="bold")[0], m(lb["set_b"], 8.0, weight="bold")[0])
        w1 = lab_w + 0.013 + TOKEN_SPAN + 0.008
        w2 = max(m(lb["score"], cells_fs * f_cell)[0], m(lb["proxy_title"], header_fs)[0]) + 2 * pad_x
        w3 = max(m(lb["weak_low"], cells_fs * f_cell)[0],
                 m(lb["weak_high"], cells_fs * f_cell)[0],
                 m(lb["weak_col"], header_fs)[0]) + 2 * pad_x
        w4 = max(m(lb["low_value"], cells_fs * f_cell, weight="bold")[0],
                 m(lb["high_value"], cells_fs * f_cell, weight="bold")[0],
                 m(lb["repair_title"], header_fs)[0]) + 2 * pad_x
        cell_h = max(
            m(lb["score"], cells_fs * f_cell)[1],
            m(lb["weak_low"], cells_fs * f_cell)[1],
            m(lb["high_value"], cells_fs * f_cell, weight="bold")[1],
        )
        avail = (P_R - P_L) - 2 * margin
        need = w1 + w2 + w3 + w4 + 3 * 0.018
        if need <= avail or f_cell <= 0.62:
            break
        f_cell -= 0.04

    az = min(0.034, max(0.018, (avail - (w1 + w2 + w3 + w4)) / 3))
    block = w1 + w2 + w3 + w4 + 3 * az
    x_start = P_L + margin + (avail - block) / 2

    c1_l = x_start
    c2_l = c1_l + w1 + az
    c3_l = c2_l + w2 + az
    c4_l = c3_l + w3 + az
    centers = {
        "proxy": c2_l + w2 / 2,
        "fixes": c3_l + w3 / 2,
        "value": c4_l + w4 / 2,
    }

    hs = header_fs
    ax.text(centers["proxy"], y_headers, lb["proxy_title"], ha="center", va="center",
            color=BLUE, fontsize=hs, weight="bold", zorder=3)
    ax.text(centers["fixes"], y_headers, lb["weak_col"], ha="center", va="center",
            color=MUTED, fontsize=hs, weight="bold", zorder=3)
    ax.text(centers["value"], y_headers, lb["repair_title"], ha="center", va="center",
            color=GREEN, fontsize=hs, weight="bold", zorder=3)

    row_h = max(cell_h + 2 * pad_y, 0.056)
    y_r1 = y_headers - 0.016 - cell_h - row_h / 2
    y_r2 = y_r1 - row_h - 0.028

    rows = [
        (y_r1, lb["set_a"], lb["weak_low"], lb["low_value"], CORAL, "#FBEEEB"),
        (y_r2, lb["set_b"], lb["weak_high"], lb["high_value"], GREEN, "#EBF3EC"),
    ]
    for yc, name, fixes, value, accent, tint in rows:
        ax.text(c1_l, yc, name, ha="left", va="center", color=INK, fontsize=8.0,
                weight="bold", zorder=3)
        draw_query_tokens(ax, c1_l + lab_w + 0.013, yc)
        rounded_box(ax, (c2_l, yc - row_h / 2), w2, row_h, fc="#EEF4F8", ec=BLUE,
                    lw=1.0, radius=0.011, zorder=1)
        ax.text(centers["proxy"], yc, lb["score"], ha="center", va="center", color=BLUE,
                fontsize=cells_fs * f_cell, zorder=3)
        rounded_box(ax, (c3_l, yc - row_h / 2), w3, row_h, fc="#F2F3F4", ec=MUTED,
                    lw=1.0, radius=0.011, zorder=1)
        ax.text(centers["fixes"], yc, fixes, ha="center", va="center", color="#49535C",
                fontsize=cells_fs * f_cell, zorder=3)
        rounded_box(ax, (c4_l, yc - row_h / 2), w4, row_h, fc=tint, ec=accent,
                    lw=1.3, radius=0.011, zorder=1)
        ax.text(centers["value"], yc, value, ha="center", va="center", color=accent,
                fontsize=cells_fs * f_cell, weight="bold", zorder=3)
        arrow(ax, (c1_l + w1 - 0.002, yc), (c2_l - 0.004, yc), scale=7.5, lw=0.95)
        arrow(ax, (c2_l + w2 + 0.004, yc), (c3_l - 0.004, yc), scale=7.5, lw=0.95)
        arrow(ax, (c3_l + w3 + 0.004, yc), (c4_l - 0.004, yc), scale=7.5, lw=0.95)


def generate(language: str, output_dir: Path, suffix: str = "") -> tuple[Path, Path]:
    labels = TEXT[language]
    output_dir.mkdir(parents=True, exist_ok=True)

    font_family = "Microsoft YaHei" if language == "zh" else "Arial"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # A slightly wider canvas gives the pipeline and the comparison panel
    # enough horizontal breathing room at paper-width exports.
    fig, ax = plt.subplots(figsize=(8.2, 3.45))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    m = Meter(ax)
    draw_pipeline(ax, m, labels)
    draw_mismatch(ax, m, labels)

    stem = f"fig1_proxy_repair_overview{suffix}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.045)
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suffix", default="", help="filename suffix, e.g. '_v2' for drafts")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(
        "paper/figures_diagnostic_zh" if args.language == "zh" else "paper/figures_diagnostic"
    )
    pdf_path, png_path = generate(args.language, output_dir, args.suffix)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
