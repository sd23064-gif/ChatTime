import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(
    r"###(Nan|[-+]?\d+(?:\.\d+)?)###"
)

EXPECTED_LENGTHS = [36, 72, 144, 288, 576]


def parse_chattime_values(text):
    """
    ChatTime形式の文字列をfloat配列へ変換する。

    例:
        ###-0.4159### ###0.1234###
        ->
        np.array([-0.4159, 0.1234])
    """
    matches = TOKEN_PATTERN.findall(str(text))

    values = []

    for value in matches:
        if value == "Nan":
            values.append(np.nan)
        else:
            try:
                values.append(float(value))
            except ValueError:
                values.append(np.nan)

    return np.asarray(values, dtype=np.float32)


def detect_sequence_length(text):
    return len(TOKEN_PATTERN.findall(str(text)))


def get_boundary(seq_len):
    """
    ChatTimeの各window設定について、
    historyとpredictionの境界を返す。
    """
    configs = {
        576: (512, 64),
        288: (256, 32),
        144: (128, 16),
        72: (64, 8),
        36: (32, 4),
    }

    return configs.get(seq_len, (None, None))


def short_title(row, row_index):
    parts = [f"row={row_index}"]

    if "seq_len" in row.index:
        parts.append(f"len={row['seq_len']}")

    if "cluster_id" in row.index:
        parts.append(f"cluster={row['cluster_id']}")

    if "cluster_size" in row.index:
        parts.append(f"size={row['cluster_size']}")

    if "ecg_id" in row.index and pd.notna(row["ecg_id"]):
        parts.append(f"ecg={row['ecg_id']}")

    if "patient_id" in row.index and pd.notna(row["patient_id"]):
        parts.append(f"patient={row['patient_id']}")

    return ", ".join(parts)


def plot_waveform(
    axis,
    values,
    title,
    show_boundary=True,
):
    x = np.arange(len(values))

    axis.plot(
        x,
        values,
        color="#1f77b4",
        linewidth=1.0,
    )

    if show_boundary:
        hist_len, pred_len = get_boundary(len(values))

        if hist_len is not None:
            axis.axvline(
                hist_len - 0.5,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

            axis.axvspan(
                hist_len,
                hist_len + pred_len - 1,
                color="orange",
                alpha=0.12,
            )

    axis.set_title(title, fontsize=8)
    axis.set_xlabel("Token index", fontsize=7)
    axis.set_ylabel("Discretized value", fontsize=7)
    axis.tick_params(axis="both", labelsize=7)
    axis.grid(alpha=0.2)

    finite_values = values[np.isfinite(values)]

    if len(finite_values) > 0:
        margin = max(
            0.02,
            0.08 * (finite_values.max() - finite_values.min()),
        )

        axis.set_ylim(
            finite_values.min() - margin,
            finite_values.max() + margin,
        )


def save_grid(
    selected_df,
    output_path,
    title,
    columns=3,
    show_boundary=True,
):
    if len(selected_df) == 0:
        print(f"[WARN] No samples for: {title}")
        return

    rows = math.ceil(len(selected_df) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.2 * rows),
        squeeze=False,
    )

    axes = axes.reshape(-1)

    for plot_index, (row_index, row) in enumerate(
        selected_df.iterrows()
    ):
        values = parse_chattime_values(row["text"])

        plot_waveform(
            axis=axes[plot_index],
            values=values,
            title=short_title(row, row_index),
            show_boundary=show_boundary,
        )

    for unused_index in range(len(selected_df), len(axes)):
        axes[unused_index].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved:", output_path)


def visualize_random_samples(
    df,
    out_dir,
    samples_per_length,
    random_seed,
):
    """
    系列長ごとにランダムな代表波形を選び、別々の画像に保存する。
    """
    rng = np.random.default_rng(random_seed)

    for seq_len in sorted(df["_seq_len"].unique()):
        group = df[df["_seq_len"] == seq_len]

        if len(group) == 0:
            continue

        sample_count = min(samples_per_length, len(group))

        selected_positions = rng.choice(
            len(group),
            size=sample_count,
            replace=False,
        )

        selected = group.iloc[selected_positions]

        save_grid(
            selected_df=selected,
            output_path=out_dir / f"random_length_{seq_len}.png",
            title=(
                f"Random clustered ECG representatives "
                f"(sequence length={seq_len})"
            ),
            columns=3,
        )


def visualize_largest_clusters(
    df,
    out_dir,
    samples_per_length,
):
    """
    cluster_sizeが大きい代表波形を系列長ごとに表示する。
    metadata CSVにcluster_sizeがある場合のみ実行する。
    """
    if "cluster_size" not in df.columns:
        print(
            "[WARN] cluster_size column is absent. "
            "Largest-cluster visualization was skipped."
        )
        return

    for seq_len in sorted(df["_seq_len"].unique()):
        group = df[df["_seq_len"] == seq_len].copy()

        group["cluster_size"] = pd.to_numeric(
            group["cluster_size"],
            errors="coerce",
        )

        group = group.sort_values(
            "cluster_size",
            ascending=False,
        ).head(samples_per_length)

        save_grid(
            selected_df=group,
            output_path=out_dir / f"largest_clusters_length_{seq_len}.png",
            title=(
                f"Representatives of largest clusters "
                f"(sequence length={seq_len})"
            ),
            columns=3,
        )


def visualize_smallest_clusters(
    df,
    out_dir,
    samples_per_length,
):
    """
    cluster_sizeが小さい代表波形を表示する。
    希少clusterや外れ値候補の確認に使用する。
    """
    if "cluster_size" not in df.columns:
        return

    for seq_len in sorted(df["_seq_len"].unique()):
        group = df[df["_seq_len"] == seq_len].copy()

        group["cluster_size"] = pd.to_numeric(
            group["cluster_size"],
            errors="coerce",
        )

        group = (
            group.dropna(subset=["cluster_size"])
            .sort_values("cluster_size", ascending=True)
            .head(samples_per_length)
        )

        save_grid(
            selected_df=group,
            output_path=out_dir / f"smallest_clusters_length_{seq_len}.png",
            title=(
                f"Representatives of smallest clusters "
                f"(sequence length={seq_len})"
            ),
            columns=3,
        )


def save_length_distribution(df, out_dir):
    """
    代表データセット内の系列長分布を棒グラフとして保存する。
    """
    counts = (
        df["_seq_len"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        [str(x) for x in counts.index],
        counts.values,
        color="#4c78a8",
    )

    plt.title("Sequence-length distribution after clustering")
    plt.xlabel("Sequence length")
    plt.ylabel("Number of representative samples")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    output_path = out_dir / "sequence_length_distribution.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", output_path)


def save_cluster_size_distribution(df, out_dir):
    """
    cluster sizeのヒストグラムを系列長別に保存する。
    """
    if "cluster_size" not in df.columns:
        return

    plot_df = df.copy()
    plot_df["cluster_size"] = pd.to_numeric(
        plot_df["cluster_size"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["cluster_size"])

    if len(plot_df) == 0:
        return

    sequence_lengths = sorted(plot_df["_seq_len"].unique())

    fig, axes = plt.subplots(
        len(sequence_lengths),
        1,
        figsize=(9, 3 * len(sequence_lengths)),
        squeeze=False,
    )

    axes = axes.reshape(-1)

    for axis, seq_len in zip(axes, sequence_lengths):
        values = plot_df.loc[
            plot_df["_seq_len"] == seq_len,
            "cluster_size",
        ].to_numpy()

        axis.hist(
            values,
            bins=40,
            color="#59a14f",
            alpha=0.85,
        )

        axis.set_title(
            f"Cluster-size distribution: sequence length={seq_len}"
        )
        axis.set_xlabel("Cluster size")
        axis.set_ylabel("Number of clusters")
        axis.grid(alpha=0.2)

    fig.tight_layout()

    output_path = out_dir / "cluster_size_distribution.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved:", output_path)


def save_basic_statistics(df, out_dir):
    """
    データセットの基本統計をCSVへ保存する。
    """
    rows = []

    for seq_len, group in df.groupby("_seq_len"):
        row = {
            "seq_len": int(seq_len),
            "num_representatives": int(len(group)),
        }

        if "cluster_size" in group.columns:
            sizes = pd.to_numeric(
                group["cluster_size"],
                errors="coerce",
            )

            row.update({
                "mean_cluster_size": sizes.mean(),
                "median_cluster_size": sizes.median(),
                "min_cluster_size": sizes.min(),
                "max_cluster_size": sizes.max(),
                "represented_original_samples": sizes.sum(),
            })

        rows.append(row)

    statistics_df = pd.DataFrame(rows).sort_values("seq_len")

    output_path = out_dir / "cluster_statistics.csv"
    statistics_df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("")
    print("=== Cluster statistics ===")
    print(statistics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clustered_file",
        type=str,
        required=True,
        help=(
            "クラスタリング後のCSV。"
            "可能ならcluster_metadata.csvを指定する。"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--samples_per_length",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=3407,
    )

    args = parser.parse_args()

    clustered_file = Path(args.clustered_file)
    out_dir = Path(args.out_dir)

    if not clustered_file.exists():
        raise FileNotFoundError(clustered_file)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:", clustered_file)

    df = pd.read_csv(clustered_file)

    if "text" not in df.columns:
        raise ValueError(
            "The input CSV must contain the 'text' column."
        )

    df["_seq_len"] = df["text"].astype(str).apply(
        detect_sequence_length
    )

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("")
    print("Sequence-length distribution:")
    print(df["_seq_len"].value_counts().sort_index())

    unexpected_lengths = sorted(
        set(df["_seq_len"].unique()) - set(EXPECTED_LENGTHS)
    )

    if unexpected_lengths:
        print(
            "[WARN] Unexpected sequence lengths:",
            unexpected_lengths,
        )

    save_length_distribution(
        df=df,
        out_dir=out_dir,
    )

    save_cluster_size_distribution(
        df=df,
        out_dir=out_dir,
    )

    save_basic_statistics(
        df=df,
        out_dir=out_dir,
    )

    visualize_random_samples(
        df=df,
        out_dir=out_dir,
        samples_per_length=args.samples_per_length,
        random_seed=args.random_seed,
    )

    visualize_largest_clusters(
        df=df,
        out_dir=out_dir,
        samples_per_length=args.samples_per_length,
    )

    visualize_smallest_clusters(
        df=df,
        out_dir=out_dir,
        samples_per_length=args.samples_per_length,
    )

    print("")
    print("Visualization completed.")
    print("Output directory:", out_dir)


if __name__ == "__main__":
    main()
