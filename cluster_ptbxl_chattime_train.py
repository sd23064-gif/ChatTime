import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


TOKEN_PATTERN = re.compile(
    r"###(Nan|[-+]?\d+(?:\.\d+)?)###"
)


WINDOW_CONFIGS = {
    576: {"hist_len": 512, "pred_len": 64},
    288: {"hist_len": 256, "pred_len": 32},
    144: {"hist_len": 128, "pred_len": 16},
    72:  {"hist_len": 64,  "pred_len": 8},
    36:  {"hist_len": 32,  "pred_len": 4},
}


def parse_chattime_values(text):
    """
    ChatTime形式の文字列から離散値列を抽出する。

    例:
        ###-0.4159### ###0.1234###
        ->
        [-0.4159, 0.1234]
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


def choose_cluster_counts(length_counts, keep_ratio, min_clusters):
    """
    各window長のサンプル数に比例してcluster数を決める。
    """
    cluster_counts = {}

    for seq_len, count in length_counts.items():
        requested = int(round(count * keep_ratio))

        n_clusters = max(min_clusters, requested)
        n_clusters = min(n_clusters, count)

        cluster_counts[int(seq_len)] = int(n_clusters)

    return cluster_counts


def build_matrix(texts, expected_len):
    """
    同じ長さのtext列を2次元float行列へ変換する。
    """
    matrix = np.empty(
        (len(texts), expected_len),
        dtype=np.float32,
    )

    valid_mask = np.ones(len(texts), dtype=bool)

    for i, text in enumerate(
        tqdm(texts, desc=f"Parsing length={expected_len}")
    ):
        values = parse_chattime_values(text)

        if len(values) != expected_len:
            valid_mask[i] = False
            matrix[i] = 0.0
            continue

        # ECGデータにNanがある場合はwindow内中央値で補完
        if np.isnan(values).any():
            finite_values = values[np.isfinite(values)]

            if len(finite_values) == 0:
                valid_mask[i] = False
                matrix[i] = 0.0
                continue

            fill_value = np.median(finite_values)
            values = np.nan_to_num(values, nan=fill_value)

        matrix[i] = values

    return matrix, valid_mask


def reduce_dimensions(
    matrix,
    pca_dim,
    batch_size,
    random_seed,
):
    """
    PCAでクラスタリング用特徴量を圧縮する。

    サンプル数や系列長が小さい場合、
    または削減後の次元数が元の次元数以上になる場合はPCAを使用しない。

    Parameters
    ----------
    matrix : np.ndarray
        shape = (num_samples, sequence_length)

    pca_dim : int
        PCA後の目標次元数。

    batch_size : int
        IncrementalPCAで使用するバッチサイズ。

    random_seed : int
        現在のIncrementalPCAでは直接使用しないが、
        関数インターフェース統一のために受け取る。

    Returns
    -------
    features : np.ndarray
        クラスタリングに使用する特徴量。

    pca : IncrementalPCA または None
        PCAを使わなかった場合はNone。
    """
    del random_seed

    if matrix.ndim != 2:
        raise ValueError(
            f"matrix must be 2-dimensional, got shape={matrix.shape}"
        )

    num_samples, original_dim = matrix.shape

    if num_samples < 2:
        print(
            f"PCA skipped: num_samples={num_samples} is too small."
        )
        return matrix.astype(np.float32, copy=False), None

    max_dim = min(
        pca_dim,
        original_dim,
        num_samples - 1,
    )

    if max_dim < 2:
        print(
            f"PCA skipped: available PCA dimension={max_dim}."
        )
        return matrix.astype(np.float32, copy=False), None

    if max_dim >= original_dim:
        print(
            f"PCA skipped: original_dim={original_dim}, "
            f"requested/effective_dim={max_dim}."
        )
        return matrix.astype(np.float32, copy=False), None

    effective_batch_size = max(batch_size, max_dim)

    # batch_sizeがサンプル数より大きくても動作可能だが、
    # 必要以上に大きくしない。
    effective_batch_size = min(
        effective_batch_size,
        num_samples,
    )

    print(
        f"Running IncrementalPCA: "
        f"samples={num_samples}, "
        f"{original_dim} -> {max_dim}, "
        f"batch_size={effective_batch_size}"
    )

    pca = IncrementalPCA(
        n_components=max_dim,
        batch_size=effective_batch_size,
    )

    features = pca.fit_transform(matrix)
    features = features.astype(np.float32, copy=False)

    explained_variance = float(
        np.sum(pca.explained_variance_ratio_)
    )

    print(
        f"PCA completed: "
        f"{original_dim} -> {features.shape[1]}, "
        f"explained variance={explained_variance:.4f}"
    )

    return features, pca


def select_representatives(
    features,
    labels,
    centers,
    strategy,
    rng,
):
    """
    各clusterから代表サンプルを1件選択する。

    random:
        ChatTime論文に近く、cluster内から無作為に1件。

    nearest:
        cluster centerに最も近いサンプルを選択。
        より典型的なmedoid近似となる。
    """
    selected_indices = []

    unique_labels = np.unique(labels)

    for cluster_id in tqdm(
        unique_labels,
        desc=f"Selecting representatives ({strategy})",
    ):
        member_indices = np.flatnonzero(labels == cluster_id)

        if len(member_indices) == 0:
            continue

        if strategy == "random":
            selected_index = rng.choice(member_indices)

        elif strategy == "nearest":
            member_features = features[member_indices]
            center = centers[cluster_id]

            distances = np.sum(
                (member_features - center) ** 2,
                axis=1,
            )

            selected_index = member_indices[np.argmin(distances)]

        else:
            raise ValueError(
                f"Unknown representative strategy: {strategy}"
            )

        selected_indices.append(int(selected_index))

    return np.asarray(selected_indices, dtype=np.int64)


def cluster_one_length(
    group_df,
    seq_len,
    n_clusters,
    pca_dim,
    batch_size,
    max_iter,
    representative_strategy,
    random_seed,
):
    """
    1種類のwindow長をclusterし、代表サンプルを返す。
    """
    print("")
    print("=" * 80)
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {len(group_df)}")
    print(f"Clusters: {n_clusters}")
    print("=" * 80)

    texts = group_df["text"].astype(str).tolist()

    matrix, valid_mask = build_matrix(
        texts=texts,
        expected_len=seq_len,
    )

    valid_positions = np.flatnonzero(valid_mask)
    valid_df = group_df.iloc[valid_positions].reset_index(drop=True)
    matrix = matrix[valid_mask]

    print(
        f"Valid samples: {len(valid_df)} / {len(group_df)}"
    )

    if len(valid_df) == 0:
        return pd.DataFrame()

    n_clusters = min(n_clusters, len(valid_df))

    if n_clusters == len(valid_df):
        result = valid_df.copy()
        result["cluster_id"] = np.arange(len(result))
        result["seq_len"] = seq_len
        result["cluster_size"] = 1
        return result

    features, _ = reduce_dimensions(
        matrix=matrix,
        pca_dim=pca_dim,
        batch_size=batch_size,
        random_seed=random_seed,
    )

    print("Running MiniBatchKMeans...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(batch_size, len(features)),
        max_iter=max_iter,
        n_init=3,
        reassignment_ratio=0.01,
        random_state=random_seed,
        verbose=0,
    )

    labels = kmeans.fit_predict(features)

    rng = np.random.default_rng(random_seed + seq_len)

    chosen_positions = select_representatives(
        features=features,
        labels=labels,
        centers=kmeans.cluster_centers_,
        strategy=representative_strategy,
        rng=rng,
    )

    result = valid_df.iloc[chosen_positions].copy()
    result["seq_len"] = seq_len
    result["cluster_id"] = labels[chosen_positions]

    cluster_sizes = np.bincount(
        labels,
        minlength=n_clusters,
    )

    result["cluster_size"] = [
        int(cluster_sizes[c])
        for c in result["cluster_id"]
    ]

    result = result.reset_index(drop=True)

    print(f"Selected representatives: {len(result)}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1,
        help="各系列長で残す代表サンプルの比率。",
    )

    parser.add_argument(
        "--min_clusters_per_length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--pca_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--representative_strategy",
        choices=["random", "nearest"],
        default="random",
        help=(
            "randomはChatTime論文に近い選択。"
            "nearestはcentroidに最も近い代表例。"
        ),
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--shuffle_output",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep_ratio must be in (0, 1].")

    train_file = Path(args.train_file)
    output_file = Path(args.output_file)

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading train CSV:", train_file)
    df = pd.read_csv(train_file)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    print("Input rows:", len(df))

    print("Detecting sequence lengths...")

    df["_seq_len"] = df["text"].astype(str).apply(
        lambda text: len(TOKEN_PATTERN.findall(text))
    )

    length_counts = (
        df["_seq_len"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Sequence-length distribution:")
    print(length_counts)

    unknown_lengths = set(length_counts) - set(WINDOW_CONFIGS)

    if unknown_lengths:
        print(
            "[WARN] Unknown sequence lengths will be skipped:",
            sorted(unknown_lengths),
        )

    usable_counts = {
        seq_len: count
        for seq_len, count in length_counts.items()
        if seq_len in WINDOW_CONFIGS
    }

    cluster_counts = choose_cluster_counts(
        length_counts=usable_counts,
        keep_ratio=args.keep_ratio,
        min_clusters=args.min_clusters_per_length,
    )

    print("Cluster counts:")
    print(cluster_counts)

    representative_dfs = []

    for seq_len in sorted(usable_counts, reverse=True):
        group_df = (
            df[df["_seq_len"] == seq_len]
            .drop(columns=["_seq_len"])
            .reset_index(drop=True)
        )

        representatives = cluster_one_length(
            group_df=group_df,
            seq_len=seq_len,
            n_clusters=cluster_counts[seq_len],
            pca_dim=args.pca_dim,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            representative_strategy=args.representative_strategy,
            random_seed=args.random_seed,
        )

        if len(representatives) > 0:
            representative_dfs.append(representatives)

    if len(representative_dfs) == 0:
        raise RuntimeError("No representative samples were selected.")

    output_df = pd.concat(
        representative_dfs,
        ignore_index=True,
    )

    if args.shuffle_output:
        output_df = output_df.sample(
            frac=1.0,
            random_state=args.random_seed,
        ).reset_index(drop=True)

    # 学習用ファイルはtextだけにする
    output_df[["text"]].to_csv(
        output_file,
        index=False,
    )

    if args.metadata_file is not None:
        metadata_file = Path(args.metadata_file)
        metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_df.to_csv(
            metadata_file,
            index=False,
        )

    summary = {
        "input_file": str(train_file),
        "output_file": str(output_file),
        "input_rows": int(len(df)),
        "output_rows": int(len(output_df)),
        "keep_ratio_requested": float(args.keep_ratio),
        "keep_ratio_actual": float(len(output_df) / len(df)),
        "representative_strategy": args.representative_strategy,
        "pca_dim": int(args.pca_dim),
        "length_counts": {
            str(k): int(v)
            for k, v in length_counts.items()
        },
        "cluster_counts": {
            str(k): int(v)
            for k, v in cluster_counts.items()
        },
    }

    summary_path = output_file.with_suffix(
        ".summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("")
    print("=== Clustering completed ===")
    print("Input rows :", len(df))
    print("Output rows:", len(output_df))
    print("Actual keep ratio:", len(output_df) / len(df))
    print("Training CSV:", output_file)

    if args.metadata_file is not None:
        print("Metadata CSV:", args.metadata_file)

    print("Summary JSON:", summary_path)


if __name__ == "__main__":
    main()