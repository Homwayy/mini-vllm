"""The README's figures, drawn from results/ and saved next to this script.

Run from the repo root: python figures/plot.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__))

# a categorical palette checked for color-vision deficiencies, and gray ink for everything that is not data
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"  # also: changes that were measured but not kept
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def load(name):
    with open(os.path.join(RESULTS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def w1(*names):
    """W1 output throughput: the mean of the runs' medians (a control run before and after the change)."""
    return float(np.mean([load(n)["median_output_throughput"] for n in names]))


def spec(*names):
    """W4 output throughput per concurrency, the mean over the given runs."""
    levels = [load(n)["levels"] for n in names]
    return {l["concurrency"]: float(np.mean([lv[i]["output_throughput"] for lv in levels])) for i, l in enumerate(levels[0])}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figures/{name}")


def bar(ax, pos, length, thickness, color, horizontal=False, radius_px=5):
    """A bar from 0 to length, rounded at its data end and square at the baseline. Call it once the axes'
    limits and layout are final: the radius is in pixels."""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    px_along, px_across = (abs(x1 - x0), abs(y1 - y0)) if horizontal else (abs(y1 - y0), abs(x1 - x0))
    ra = min(radius_px / px_along, abs(length) / 2)  # the corner radius along the bar and across it, in data units
    rc = min(radius_px / px_across, thickness / 2)
    s, e, lo, hi, k = (1 if length >= 0 else -1), length, pos - thickness / 2, pos + thickness / 2, 0.5523  # k: cubic arc
    pts = [(0, lo), (e - s * ra, lo), (e - s * ra * (1 - k), lo), (e, lo + rc * (1 - k)), (e, lo + rc), (e, hi - rc),
           (e, hi - rc * (1 - k)), (e - s * ra * (1 - k), hi), (e - s * ra, hi), (0, hi), (0, lo)]
    codes = [Path.MOVETO, Path.LINETO, *[Path.CURVE4] * 3, Path.LINETO, *[Path.CURVE4] * 3, Path.LINETO, Path.CLOSEPOLY]
    verts = pts if horizontal else [(c, a) for a, c in pts]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, linewidth=0))


def pct(v):
    s = f"{v:+.1%}" if abs(v) >= 0.01 else f"{v:+.2%}"
    return s.replace("-", "−")


def plot_w1_gains():
    """Each change's effect on W1 throughput, against the control run before and after it in the same session."""
    rows = [
        ("CUDA graph（E5，消融）", w1("e5_w1_graph_before", "e5_w1_graph_after") / w1("e5_w1_eager") - 1, True),
        ("分页 KV（E2）", w1("e2_paged_static") / w1("e2_ctrl_before", "e2_ctrl_after") - 1, True),
        ("融合逐元素的 kernel（E9）", w1("e9_w1") / w1("e9_w1_unfused_before", "e9_w1_unfused_after") - 1, True),
        ("连续批处理（E3）", w1("e3_continuous") / w1("e3_ctrl_before", "e3_ctrl_after") - 1, True),
        ("Triton 分页 decode 注意力（E4）", w1("e4_w1_triton") / w1("e4_w1_flex_before", "e4_w1_flex_after") - 1, True),
        ("prefill 也用 Triton（E4b）", w1("e4b_w1_triton") / w1("e4b_w1_triton_decode_before", "e4b_w1_triton_decode_after") - 1, True),
        ("短 prefill 也录图（E10）", w1("e10_w1") / w1("e10_w1_eager_before", "e10_w1_eager_after") - 1, True),
        ("乐观准入 + 余量（E3c，TPOT p99 × 2.4）", w1("e3c_headroom") / w1("e3b_reserve_before", "e3b_reserve_after") - 1, False),
        ("乐观准入（E3b，反复抢占）", w1("e3b_optimistic_before", "e3b_optimistic_after") / w1("e3b_reserve_before", "e3b_reserve_after") - 1, False),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(rows))[::-1]
    ax.set_xlim(-0.06, 0.92)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    fig.canvas.draw()  # fixes the axes' size, so bars can be rounded in pixels
    for yi, (label, v, kept) in zip(y, rows):
        bar(ax, yi, v, 0.5, BLUE if kept else GRAY, horizontal=True)
        ax.text(max(v, 0) + 0.008, yi, pct(v), va="center", ha="left", fontsize=9, color=INK2)  # right of 0 for a loss too
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0%}".replace("-", "−") if v else "0"))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("W1 输出吞吐的变化（相对同一次测试里的对照组）")
    ax.set_title("每项改动对离线吞吐的影响（Qwen3-0.6B，W1：256 条请求）")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=GRAY)], labels=["采用", "测过，没有采用"],
              loc="lower right", fontsize=9)
    ax.text(0, -0.2, "对照组在改动的前后各跑一次，取平均；每组 3 次取中位数。CUDA graph 是消融：关掉 decode 的录图，吞吐低 44%。",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    save(fig, "w1_gains.png")


def plot_attention_bandwidth():
    """Effective K/V read bandwidth of one decode attention layer, per batch size (context 1024)."""
    d = load("e4_attention")
    shapes = [s for s in d["shapes"] if s["shape"].endswith("context 1024")]
    batch = [s["batch"] for s in shapes]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    peak = d["read_bandwidth_gbs"]
    ax.axhline(peak, color=INK2, linewidth=1)
    ax.text(1.0, peak + 12, f"显存读带宽 {peak:.0f} GB/s（实测）", fontsize=8.5, color=INK2, va="bottom")
    lines = [("triton", "自己写的 Triton 分页 kernel", BLUE), ("flex", "FlexAttention", ORANGE), ("gather", "gather：拷出 K/V 再调用 SDPA", AQUA)]
    for key, label, color in lines:
        v = [s[key]["effective_gbs"] for s in shapes]
        ax.plot(batch, v, color=color, linewidth=1.8, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                label=label, solid_capstyle="round", solid_joinstyle="round")
        ax.text(batch[0] / 1.12, v[0], f"{v[0]:.0f}", ha="right", va="center", fontsize=9, color=INK2)
    flex1, tri1 = shapes[0]["flex"]["us_per_layer"], shapes[0]["triton"]["us_per_layer"]
    ax.text(1.12, 240, f"batch 1：每层 Triton {tri1:.1f} µs，\nflex {flex1:.1f} µs（慢 {flex1 / tri1:.1f} 倍）", fontsize=8.5, color=INK2, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch, [str(b) for b in batch])
    ax.minorticks_off()
    ax.set_xlim(0.75, 80)
    ax.set_ylim(0, 680)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("batch（每条序列 1024 个 token 的上下文）")
    ax.set_ylabel("有效带宽（GB/s）")
    ax.set_title("单层 decode 注意力读 K/V 的有效带宽（Qwen3 的形状，E4）")
    ax.legend(loc="center right", fontsize=9)
    ax.text(0, -0.2, "有效带宽 = 这一层要读的 K/V 字节数 ÷ 耗时。FlexAttention 有完整块时按整个 KV 池的长度切分工作，"
            "batch 小时短序列全落在同一段里，并行度不够。", transform=ax.transAxes, fontsize=8, color=GRAY)
    save(fig, "attention_bandwidth.png")


def plot_spec_decoding():
    """Speculative decoding on GPT-2 (W4): throughput over no speculation, the best number of drafts per
    concurrency, for temperature 1.0 (E8c's session) and greedy (E8e's session)."""
    t1_off = spec("e8c_t1_off")
    t1 = {"EAGLE": {3: spec("e8c_t1_eagle3"), 4: spec("e8c_t1_eagle4_before", "e8c_t1_eagle4_after")},
          "DSpark": {3: spec("e8c_t1_dspark3"), 4: spec("e8c_t1_dspark4")}}
    g_off = spec("e8e_greedy_off_before", "e8e_greedy_off_after")
    greedy = {"n-gram": {4: spec("e8e_greedy_ngram4"), 8: spec("e8e_greedy_ngram8_before", "e8e_greedy_ngram8_after")},
              "EAGLE": {g: spec(f"e8e_greedy_eagle{g}") for g in (3, 4, 6, 8)}}
    colors = {"EAGLE": BLUE, "DSpark": ORANGE, "n-gram": AQUA}
    conc = [1, 8, 32, 64]
    panels = [("temperature 1.0（E8c）", t1, t1_off), ("贪心（E8e）", greedy, g_off)]
    when = {"temperature 1.0（E8c）": "temperature 1.0 时", "贪心（E8e）": "贪心时"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (title, runs, _) in zip(axes, panels):
        ax.set_xlim(-0.6, len(conc) - 0.4)
        ax.set_ylim(0, 4.3)
        ax.axhline(1, color=INK2, linewidth=1)
        ax.set_xticks(range(len(conc)), [f"并发 {c}" for c in conc])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5)
        ax.legend(handles=[Rectangle((0, 0), 1, 1, color=colors[n]) for n in runs], labels=list(runs), loc="upper right", fontsize=9)
    axes[0].set_ylabel("吞吐 ÷ 不投机的吞吐")
    axes[1].text(1.01, 1, "不投机", transform=blended_transform_factory(axes[1].transAxes, axes[1].transData), ha="left", va="center",
                 fontsize=8.5, color=INK2)
    fig.suptitle("投机解码的加速（GPT-2 124M，W4：真实文本），每个并发取最好的草稿数 γ", x=0.065, y=0.99, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.canvas.draw()  # the layout is final: bars can be rounded in pixels
    notes, width = [], 0.36
    for ax, (title, runs, off) in zip(axes, panels):
        part = []
        for k, (name, by_gamma) in enumerate(runs.items()):
            best = [max(by_gamma, key=lambda g: by_gamma[g][c]) for c in conc]
            for i, (g, c) in enumerate(zip(best, conc)):
                x, r = i + (k - 0.5) * (width + 0.04), by_gamma[g][c] / off[c]
                bar(ax, x, r, width, colors[name])
                ax.text(x, r + 0.05, f"{r:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
            part.append(f"{name} {'、'.join(map(str, best))}")
        notes.append(when[title] + " " + "，".join(part))
    fig.text(0.065, -0.01, "最好的 γ（并发 1、8、32、64）：" + "；".join(notes) + "。\ntemperature 1.0 时 n-gram 只有 7–8% 的草稿被接受，"
             "并发 8–64 反而更慢（E8a），所以左图没有它。", fontsize=8, color=GRAY, va="top")
    save(fig, "spec_decoding.png")


if __name__ == "__main__":
    plot_w1_gains()
    plot_attention_bandwidth()
    plot_spec_decoding()
