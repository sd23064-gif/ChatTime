"""CGTSF context 評価結果の可視化スクリプト。

評価スクリプトが出力する以下の CSV を読み込んで図を作る。

    outputs/cgtsf_{dataset}_context_model_comparison_details.csv
    outputs/cgtsf_{dataset}_context_model_comparison_summary.csv
    outputs/cgtsf_{dataset}_context_model_comparison_overall.csv

使い方:

    python visualize_results.py --input_dir outputs --dataset_name PTF
    python visualize_results.py --input_dir outputs --dataset_name PTF --metric raw

図のラベルは日本語フォント未導入環境でも崩れないよう英語にしてある。
"""

import argparse
import ast
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# =====================================================================
# 入出力
# =====================================================================
def resolve_paths(args):
    stem = f"cgtsf_{args.dataset_name.lower()}_context_model_comparison"

    return {
        "details": args.details or os.path.join(args.input_dir, f"{stem}_details.csv"),
        "summary": args.summary or os.path.join(args.input_dir, f"{stem}_summary.csv"),
        "overall": args.overall or os.path.join(args.input_dir, f"{stem}_overall.csv"),
    }


def load_csv(path, label):
    if not os.path.exists(path):
        print(f"[warn] {label} が見つかりません: {path}")
        return None

    df = pd.read_csv(path)

    if len(df) == 0:
        print(f"[warn] {label} が空です: {path}")
        return None

    print(f"[load] {label}: {path} ({len(df)} rows)")
    return df


def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(" -", path)
    return path


def metric_columns(metric):
    """raw / std のどちらの指標系列を使うかを決める。"""
    if metric == "raw":
        return {
            "zero": "raw_mae_zero",
            "ctx": "raw_mae_context",
            "gain": "raw_context_gain",
            "zero_mean": "raw_mae_zero_mean",
            "ctx_mean": "raw_mae_context_mean",
            "gain_mean": "raw_context_gain_mean",
            "label": "Raw MAE",
        }

    return {
        "zero": "std_mae_zero",
        "ctx": "std_mae_context",
        "gain": "std_context_gain",
        "zero_mean": "std_mae_zero_mean",
        "ctx_mean": "std_mae_context_mean",
        "gain_mean": "std_context_gain_mean",
        "label": "Standardized MAE",
    }


def require(df, cols, fig_name):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        print(f"[skip] {fig_name}: 列が足りません {missing}")
        return False

    return True


# =====================================================================
# 図 1: モデル別の zero-shot vs context (overall)
# =====================================================================
def plot_overall_bar(overall_df, cols, out_dir):
    name = "fig1_overall_zero_vs_context.png"

    if not require(overall_df, [cols["zero_mean"], cols["ctx_mean"], "model"], name):
        return

    df = overall_df.sort_values(cols["ctx_mean"])
    x = np.arange(len(df))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3, 4.2))

    ax.bar(x - w / 2, df[cols["zero_mean"]], w, label="zero-shot (no text)", color=ZERO_COLOR)
    ax.bar(x + w / 2, df[cols["ctx_mean"]], w, label="with context", color=CTX_COLOR)

    for xi, (z, c) in enumerate(zip(df[cols["zero_mean"]], df[cols["ctx_mean"]])):
        ax.text(xi - w / 2, z, f"{z:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, c, f"{c:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylabel(cols["label"] + " (lower is better)")
    ax.set_title("Overall forecast error by model")
    ax.legend()

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 2: hist_len ごとの誤差カーブ
# =====================================================================
def plot_hist_len_curve(summary_df, cols, out_dir):
    name = "fig2_hist_len_curve.png"

    if not require(summary_df, [cols["zero_mean"], cols["ctx_mean"], "hist_len", "model"], name):
        return

    models = list(dict.fromkeys(summary_df["model"]))
    palette = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    for i, m in enumerate(models):
        sub = summary_df[summary_df["model"] == m].sort_values("hist_len")
        color = palette(i % 10)

        ax.plot(sub["hist_len"], sub[cols["zero_mean"]], "o--", color=color, alpha=0.55,
                label=f"{m} (zero-shot)")
        ax.plot(sub["hist_len"], sub[cols["ctx_mean"]], "o-", color=color,
                label=f"{m} (context)")

    ax.set_xlabel("History length")
    ax.set_ylabel(cols["label"] + " (lower is better)")
    ax.set_title("Error vs history length")
    ax.set_xticks(sorted(summary_df["hist_len"].unique()))
    ax.legend(fontsize=8, ncol=1, loc="best")

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 3: context gain の分布
# =====================================================================
def plot_context_gain_box(detail_df, cols, out_dir):
    name = "fig3_context_gain_distribution.png"

    if not require(detail_df, [cols["gain"], "model"], name):
        return

    models = list(dict.fromkeys(detail_df["model"]))
    data = [detail_df.loc[detail_df["model"] == m, cols["gain"]].dropna().values for m in models]
    data = [d for d in data if len(d) > 0]
    models = [m for m, d in zip(models, data) if len(d) > 0]

    if not data:
        print(f"[skip] {name}: 有効な gain がありません")
        return

    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 3, 4.6))

    # labels / tick_labels は matplotlib のバージョンで名前が違うため後から設定する
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models)

    for patch in bp["boxes"]:
        patch.set_facecolor(CTX_COLOR)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        jitter = rng.normal(0, 0.05, size=len(d))
        ax.plot(i + jitter, d, ".", color=CTX_COLOR, alpha=0.5, markersize=5)
        ax.plot(i, np.mean(d), "D", color="crimson", markersize=6,
                label="mean" if i == 1 else None)

    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Context gain = MAE(zero) - MAE(context)")
    ax.set_title("Per-sample context gain  (>0 : text helps)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 4: サンプル単位の zero vs context 散布図
# =====================================================================
def plot_zero_vs_context_scatter(detail_df, cols, out_dir):
    name = "fig4_per_sample_scatter.png"

    if not require(detail_df, [cols["zero"], cols["ctx"], "model"], name):
        return

    models = list(dict.fromkeys(detail_df["model"]))
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False, layout="constrained")

    for ax, m in zip(axes.ravel(), models):
        sub = detail_df[detail_df["model"] == m].dropna(subset=[cols["zero"], cols["ctx"]])

        if len(sub) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(sub[cols["zero"]], sub[cols["ctx"]], s=22, alpha=0.65, color=CTX_COLOR,
                   edgecolor="white", linewidth=0.5)

        lim_max = float(max(sub[cols["zero"]].max(), sub[cols["ctx"]].max())) * 1.08
        ax.plot([0, lim_max], [0, lim_max], "--", color="black", lw=1)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)

        win = float((sub[cols["ctx"]] < sub[cols["zero"]]).mean() * 100)
        ax.set_title(f"{m}\ncontext better on {win:.0f}% of windows", fontsize=10)
        ax.set_xlabel(cols["label"] + " (zero-shot)")
        ax.set_ylabel(cols["label"] + " (context)")

    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)

    fig.suptitle("Below the diagonal = context improved the forecast", fontsize=11)

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 5: 方向精度（sign flip rate）
# =====================================================================
def plot_sign_flip(overall_df, out_dir):
    name = "fig5_sign_flip_rate.png"

    if not require(overall_df, ["sign_flip_zero_mean", "sign_flip_context_mean", "model"], name):
        return

    df = overall_df.sort_values("sign_flip_context_mean")
    x = np.arange(len(df))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3, 4.2))

    ax.bar(x - w / 2, df["sign_flip_zero_mean"], w, label="zero-shot", color=ZERO_COLOR)
    ax.bar(x + w / 2, df["sign_flip_context_mean"], w, label="with context", color=CTX_COLOR)
    ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance level")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylabel("Sign flip rate (lower is better)")
    ax.set_title("Directional accuracy")
    ax.legend()

    save_fig(fig, out_dir, name)


# =====================================================================
# 図 6: 予測系列の例（--save_predictions で保存した場合のみ）
# =====================================================================
def to_series(v):
    if isinstance(v, str):
        return np.asarray(ast.literal_eval(v), dtype=np.float64)

    if isinstance(v, (list, np.ndarray)):
        return np.asarray(v, dtype=np.float64)

    return np.asarray([], dtype=np.float64)


def plot_example_forecasts(detail_df, out_dir, n_examples=3, model=None, hist_len=None):
    name = "fig6_example_forecasts.png"
    needed = ["hist", "true", "pred_zero", "pred_context"]

    if not require(detail_df, needed, name):
        print("      (評価時に --save_predictions を付けると描画できます)")
        return

    sub = detail_df

    if model is not None:
        sub = sub[sub["model"] == model]
    elif "model" in sub.columns:
        sub = sub[sub["model"] == sub["model"].iloc[0]]

    if hist_len is not None and "hist_len" in sub.columns:
        sub = sub[sub["hist_len"] == hist_len]
    elif "hist_len" in sub.columns:
        sub = sub[sub["hist_len"] == sub["hist_len"].max()]

    sub = sub.dropna(subset=["true"]).head(n_examples)

    if len(sub) == 0:
        print(f"[skip] {name}: 該当サンプルがありません")
        return

    fig, axes = plt.subplots(len(sub), 1, figsize=(9, 3.1 * len(sub)),
                             squeeze=False, layout="constrained")

    for ax, (_, row) in zip(axes.ravel(), sub.iterrows()):
        hist = to_series(row["hist"])
        true = to_series(row["true"])
        pz = to_series(row["pred_zero"])
        pc = to_series(row["pred_context"])

        h = np.arange(-len(hist), 0)
        f = np.arange(len(true))

        ax.plot(h, hist, color="black", lw=1.2, label="history")
        ax.plot(f, true, color="black", lw=2, label="ground truth")
        ax.plot(np.arange(len(pz)), pz, "--", color=ZERO_COLOR, lw=1.8, label="zero-shot")
        ax.plot(np.arange(len(pc)), pc, "-", color=CTX_COLOR, lw=1.8, label="context")
        ax.axvline(-0.5, color="gray", ls=":", lw=1)

        title = f"sample_id={row.get('sample_id', '?')}"
        if "model" in row:
            title = f"{row['model']}  |  " + title
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time step (0 = forecast start)")

    axes.ravel()[0].legend(fontsize=8, ncol=4, loc="upper left")

    save_fig(fig, out_dir, name)


# =====================================================================
# コンソール用の要約
# =====================================================================
def print_text_summary(overall_df, cols):
    keep = ["model", cols["zero_mean"], cols["ctx_mean"], cols["gain_mean"]]
    keep = [c for c in keep if c in overall_df.columns]

    if len(keep) <= 1:
        return

    print("\nOverall (sorted by context error):")
    print(overall_df[keep].sort_values(cols["ctx_mean"]).to_string(index=False))


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
    p.add_argument("--overall", type=str, default=None)

    p.add_argument("--metric", type=str, default="std", choices=["std", "raw"],
                   help="std: 標準化 MAE, raw: 生の MAE")

    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--example_model", type=str, default=None)
    p.add_argument("--example_hist_len", type=int, default=None)

    return p


def main():
    args = build_arg_parser().parse_args()

    paths = resolve_paths(args)
    cols = metric_columns(args.metric)

    fig_dir = args.fig_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    detail_df = load_csv(paths["details"], "details")
    summary_df = load_csv(paths["summary"], "summary")
    overall_df = load_csv(paths["overall"], "overall")

    if detail_df is None and summary_df is None and overall_df is None:
        raise SystemExit("読み込める CSV がありません。--input_dir / --dataset_name を確認してください。")

    # overall / summary が無ければ details から作り直す
    if overall_df is None and detail_df is not None:
        overall_df = (
            detail_df.groupby(["model"], as_index=False)
            .agg(**{
                cols["zero_mean"]: (cols["zero"], "mean"),
                cols["ctx_mean"]: (cols["ctx"], "mean"),
                cols["gain_mean"]: (cols["gain"], "mean"),
                "sign_flip_zero_mean": ("sign_flip_zero", "mean"),
                "sign_flip_context_mean": ("sign_flip_context", "mean"),
            })
        )

    if summary_df is None and detail_df is not None:
        summary_df = (
            detail_df.groupby(["model", "hist_len"], as_index=False)
            .agg(**{
                cols["zero_mean"]: (cols["zero"], "mean"),
                cols["ctx_mean"]: (cols["ctx"], "mean"),
            })
        )

    print("\nSaved figures:")

    if overall_df is not None:
        plot_overall_bar(overall_df, cols, fig_dir)
        plot_sign_flip(overall_df, fig_dir)

    if summary_df is not None:
        plot_hist_len_curve(summary_df, cols, fig_dir)

    if detail_df is not None:
        plot_context_gain_box(detail_df, cols, fig_dir)
        plot_zero_vs_context_scatter(detail_df, cols, fig_dir)
        plot_example_forecasts(
            detail_df,
            fig_dir,
            n_examples=args.n_examples,
            model=args.example_model,
            hist_len=args.example_hist_len,
        )

    if overall_df is not None:
        print_text_summary(overall_df, cols)


if __name__ == "__main__":
    main()
