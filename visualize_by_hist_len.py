"""入力系列長 (hist_len) ごとに context あり / なしを比較する可視化スクリプト。

visualize_results.py の fig2 を掘り下げたもので、details CSV があれば
ばらつき・勝率・対応のある検定まで出す。

使い方:

    python visualize_by_hist_len.py --input_dir outputs --dataset_name PTF
    python visualize_by_hist_len.py --input_dir outputs --dataset_name PTF --metric raw
    python visualize_by_hist_len.py --input_dir outputs --models mamba_base,mamba_finetuned

図のラベルは日本語フォント未導入環境でも崩れないよう英語にしてある。
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy が無くても図は出す
    wilcoxon = None


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

ZERO_COLOR = "#7f8c9a"
CTX_COLOR = "#2f6fb3"
GAIN_POS = "#2f8f4e"
GAIN_NEG = "#c0392b"


# =====================================================================
# 入出力
# =====================================================================
def metric_columns(metric):
    if metric == "raw":
        return {
            "zero": "raw_mae_zero",
            "ctx": "raw_mae_context",
            "gain": "raw_context_gain",
            "zero_mean": "raw_mae_zero_mean",
            "ctx_mean": "raw_mae_context_mean",
            "label": "Raw MAE",
        }

    return {
        "zero": "std_mae_zero",
        "ctx": "std_mae_context",
        "gain": "std_context_gain",
        "zero_mean": "std_mae_zero_mean",
        "ctx_mean": "std_mae_context_mean",
        "label": "Standardized MAE",
    }


def load_frames(args, cols):
    """details を優先し、無ければ summary から棒グラフ用の平均だけ作る。"""
    stem = f"cgtsf_{args.dataset_name.lower()}_context_model_comparison"
    detail_path = args.details or os.path.join(args.input_dir, f"{stem}_details.csv")
    summary_path = args.summary or os.path.join(args.input_dir, f"{stem}_summary.csv")

    detail_df = None
    summary_df = None

    if os.path.exists(detail_path):
        detail_df = pd.read_csv(detail_path)
        print(f"[load] details: {detail_path} ({len(detail_df)} rows)")

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print(f"[load] summary: {summary_path} ({len(summary_df)} rows)")

    if detail_df is None and summary_df is None:
        raise SystemExit(
            f"CSV が見つかりません:\n  {detail_path}\n  {summary_path}\n"
            "--input_dir / --dataset_name を確認してください。"
        )

    if detail_df is not None and summary_df is None:
        summary_df = (
            detail_df.groupby(["model", "hist_len"], as_index=False)
            .agg(**{
                cols["zero_mean"]: (cols["zero"], "mean"),
                cols["ctx_mean"]: (cols["ctx"], "mean"),
            })
        )

    return detail_df, summary_df


def filter_models(df, models):
    if df is None or models is None:
        return df

    keep = [m.strip() for m in models.split(",") if m.strip()]
    out = df[df["model"].isin(keep)]

    if len(out) == 0:
        raise SystemExit(f"--models {models} に一致する行がありません。")

    return out


def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(" -", path)


# =====================================================================
# 集計
# =====================================================================
def aggregate_by_hist_len(detail_df, cols):
    """(model, hist_len) ごとに平均・標準誤差・勝率・検定 p 値をまとめる。"""
    rows = []

    for (model, hist_len), g in detail_df.groupby(["model", "hist_len"], sort=False):
        g = g.dropna(subset=[cols["zero"], cols["ctx"]])

        if len(g) == 0:
            continue

        z = g[cols["zero"]].to_numpy(dtype=float)
        c = g[cols["ctx"]].to_numpy(dtype=float)
        gain = z - c
        n = len(gain)

        p_value = np.nan
        if wilcoxon is not None and n >= 6 and np.any(gain != 0):
            try:
                p_value = float(wilcoxon(z, c).pvalue)
            except ValueError:
                p_value = np.nan

        rows.append({
            "model": model,
            "hist_len": int(hist_len),
            "n": n,
            "zero_mean": float(z.mean()),
            "ctx_mean": float(c.mean()),
            "zero_sem": float(z.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "ctx_sem": float(c.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "gain_mean": float(gain.mean()),
            "gain_sem": float(gain.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "gain_median": float(np.median(gain)),
            "win_rate": float((c < z).mean()),
            "rel_gain_pct": float(gain.mean() / z.mean() * 100) if z.mean() != 0 else np.nan,
            "p_value": p_value,
        })

    return pd.DataFrame(rows).sort_values(["model", "hist_len"]).reset_index(drop=True)


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def model_order(df):
    return list(dict.fromkeys(df["model"]))


# =====================================================================
# 図 A: モデルごとに hist_len × (zero / context) の棒グラフ
# =====================================================================
def plot_grouped_bars(agg_df, summary_df, cols, out_dir):
    name = "hist_len_fig1_grouped_bars.png"

    use_agg = agg_df is not None and len(agg_df) > 0
    df = agg_df if use_agg else summary_df.rename(columns={
        cols["zero_mean"]: "zero_mean",
        cols["ctx_mean"]: "ctx_mean",
    })

    models = model_order(df)
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow),
                             squeeze=False, sharey=True, layout="constrained")

    for ax, m in zip(axes.ravel(), models):
        sub = df[df["model"] == m].sort_values("hist_len")
        x = np.arange(len(sub))
        w = 0.38

        zero_err = sub["zero_sem"] if use_agg else None
        ctx_err = sub["ctx_sem"] if use_agg else None

        ax.bar(x - w / 2, sub["zero_mean"], w, yerr=zero_err, capsize=3,
               label="zero-shot (no text)", color=ZERO_COLOR)
        ax.bar(x + w / 2, sub["ctx_mean"], w, yerr=ctx_err, capsize=3,
               label="with context", color=CTX_COLOR)

        if use_agg:
            top = max(sub["zero_mean"].max(), sub["ctx_mean"].max())
            for xi, (_, r) in enumerate(sub.iterrows()):
                mark = stars(r["p_value"])
                if mark:
                    ax.text(xi, top * 1.04, mark, ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["hist_len"].astype(int))
        ax.set_xlabel("History length")
        ax.set_title(m, fontsize=11)

    axes.ravel()[0].set_ylabel(cols["label"] + " (lower is better)")
    axes.ravel()[0].legend(fontsize=8)

    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)

    note = "error bars: SEM,  * p<.05  ** p<.01  *** p<.001 (Wilcoxon, paired)" if use_agg else ""
    fig.suptitle("Zero-shot vs context at each history length\n" + note, fontsize=11)

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 B: context gain と勝率を hist_len に対してプロット
# =====================================================================
def plot_gain_and_winrate(agg_df, out_dir):
    name = "hist_len_fig2_gain_and_winrate.png"
    models = model_order(agg_df)
    palette = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    ax_gain, ax_win = axes

    for i, m in enumerate(models):
        sub = agg_df[agg_df["model"] == m].sort_values("hist_len")
        color = palette(i % 10)

        ax_gain.errorbar(sub["hist_len"], sub["gain_mean"], yerr=sub["gain_sem"],
                         fmt="o-", capsize=3, color=color, label=m)
        ax_win.plot(sub["hist_len"], sub["win_rate"] * 100, "o-", color=color, label=m)

    ax_gain.axhline(0, color="black", lw=1)
    ax_gain.set_xlabel("History length")
    ax_gain.set_ylabel("Context gain = MAE(zero) - MAE(context)")
    ax_gain.set_title("Mean context gain  (>0 : text helps)")
    ax_gain.set_xticks(sorted(agg_df["hist_len"].unique()))
    ax_gain.legend(fontsize=8)

    ax_win.axhline(50, color="crimson", ls="--", lw=1, label="chance (50%)")
    ax_win.set_xlabel("History length")
    ax_win.set_ylabel("Windows where context wins (%)")
    ax_win.set_title("Win rate of context")
    ax_win.set_xticks(sorted(agg_df["hist_len"].unique()))
    ax_win.set_ylim(0, 100)
    ax_win.legend(fontsize=8)

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 C: hist_len ごとの誤差分布（対応ありの箱ひげ）
# =====================================================================
def plot_paired_box(detail_df, cols, out_dir):
    name = "hist_len_fig3_paired_distribution.png"
    models = model_order(detail_df)
    hist_lens = sorted(detail_df["hist_len"].unique())

    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.4 * nrow),
                             squeeze=False, sharey=True, layout="constrained")

    for ax, m in zip(axes.ravel(), models):
        data = []
        positions = []
        colors = []
        w = 0.34

        for i, hl in enumerate(hist_lens):
            sub = detail_df[(detail_df["model"] == m) & (detail_df["hist_len"] == hl)]

            z = sub[cols["zero"]].dropna().to_numpy(dtype=float)
            c = sub[cols["ctx"]].dropna().to_numpy(dtype=float)

            if len(z) > 0:
                data.append(z)
                positions.append(i - w / 2)
                colors.append(ZERO_COLOR)

            if len(c) > 0:
                data.append(c)
                positions.append(i + w / 2)
                colors.append(CTX_COLOR)

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, positions=positions, widths=w * 0.9,
                        showfliers=False, patch_artist=True,
                        medianprops={"color": "black"})

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        ax.set_xticks(np.arange(len(hist_lens)))
        ax.set_xticklabels(hist_lens)
        ax.set_xlim(-0.6, len(hist_lens) - 0.4)
        ax.set_xlabel("History length")
        ax.set_title(m, fontsize=11)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ZERO_COLOR, alpha=0.45, label="zero-shot"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CTX_COLOR, alpha=0.45, label="with context"),
    ]
    axes.ravel()[0].set_ylabel(cols["label"])
    axes.ravel()[0].legend(handles=handles, fontsize=8)

    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)

    fig.suptitle("Per-window error distribution at each history length", fontsize=11)

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 D: モデル × hist_len の gain ヒートマップ
# =====================================================================
def plot_gain_heatmap(agg_df, out_dir):
    name = "hist_len_fig4_gain_heatmap.png"

    pivot = agg_df.pivot(index="model", columns="hist_len", values="gain_mean")
    pivot = pivot.loc[model_order(agg_df)]

    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0

    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 3.5, 0.9 * pivot.shape[0] + 2.4))

    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("History length")
    ax.grid(False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9, color="black")

    fig.colorbar(im, ax=ax, label="Context gain (green = text helps)")
    ax.set_title("Context gain by model and history length")

    save_fig(fig, out_dir, name)


# =====================================================================
# main
# =====================================================================
def build_arg_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--input_dir", type=str, default="outputs")
    p.add_argument("--dataset_name", type=str, default="PTF")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="図の出力先 (既定: {input_dir}/figures)")

    p.add_argument("--details", type=str, default=None)
    p.add_argument("--summary", type=str, default=None)

    p.add_argument("--metric", type=str, default="std", choices=["std", "raw"])
    p.add_argument("--models", type=str, default=None,
                   help="カンマ区切りで対象モデルを絞る 例: mamba_base,mamba_finetuned")
    p.add_argument("--save_table", action="store_true",
                   help="hist_len 別の集計値を CSV でも保存する")

    return p


def main():
    args = build_arg_parser().parse_args()

    cols = metric_columns(args.metric)
    detail_df, summary_df = load_frames(args, cols)

    detail_df = filter_models(detail_df, args.models)
    summary_df = filter_models(summary_df, args.models)

    fig_dir = args.fig_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    agg_df = aggregate_by_hist_len(detail_df, cols) if detail_df is not None else None

    print("\nSaved figures:")

    plot_grouped_bars(agg_df, summary_df, cols, fig_dir)

    if agg_df is not None and len(agg_df) > 0:
        plot_gain_and_winrate(agg_df, fig_dir)
        plot_paired_box(detail_df, cols, fig_dir)

        if agg_df["hist_len"].nunique() > 1 and agg_df["model"].nunique() > 1:
            plot_gain_heatmap(agg_df, fig_dir)
    else:
        print("[info] details CSV が無いため棒グラフのみ出力しました。")

    if agg_df is not None and len(agg_df) > 0:
        show = agg_df[[
            "model", "hist_len", "n", "zero_mean", "ctx_mean",
            "gain_mean", "rel_gain_pct", "win_rate", "p_value",
        ]]
        print("\nContext effect by history length:")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        if args.save_table:
            table_path = os.path.join(fig_dir, "hist_len_context_effect.csv")
            agg_df.to_csv(table_path, index=False)
            print("\nSaved table:\n -", table_path)


if __name__ == "__main__":
    main()
