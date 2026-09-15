#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


REGRESSION_TARGETS = [
    "value",
    "abs_value",
    "current_delta",
    "next_value",
    "next_delta",
    "second_diff",
    "local_slope",
    "local_volatility",
]

CLASSIFICATION_TARGETS = [
    "sign",
]


def parse_int_list(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_float_list(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def discover_feature_arrays(npz_file):
    """
    対応例:
      L7_token_state
      L14_pre_token_state
      layer_7_token_state
      layer7_token_state
    """
    patterns = [
        re.compile(
            r"^L(?P<layer>\d+)_(?P<state>token_state|pre_token_state)$"
        ),
        re.compile(
            r"^layer_?(?P<layer>\d+)_(?P<state>token_state|pre_token_state)$"
        ),
    ]

    discovered = {}

    for key in npz_file.files:
        match = None

        for pattern in patterns:
            match = pattern.fullmatch(key)

            if match:
                break

        if match is None:
            continue

        layer = int(match.group("layer"))
        state = match.group("state")
        discovered[(layer, state)] = key

    if not discovered:
        raise ValueError(
            "No feature arrays were discovered. "
            f"Available NPZ keys: {npz_file.files}"
        )

    return discovered


def validate_metadata(mamba, llama):
    keys = ["series_id", "kind", "position"]

    if len(mamba) != len(llama):
        raise ValueError(
            f"Metadata row mismatch: Mamba={len(mamba)}, Llama={len(llama)}"
        )

    if not mamba[keys].equals(llama[keys]):
        raise ValueError(
            "Mamba and Llama metadata rows are not aligned."
        )

    targets = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

    for target in targets:
        left = mamba[target].to_numpy(dtype=np.float64)
        right = llama[target].to_numpy(dtype=np.float64)

        if not np.isclose(
            left,
            right,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ).all():
            raise ValueError(
                f"Target mismatch between models: {target}"
            )


def split_series_ids(series_ids, seed, train_ratio, val_ratio):
    unique_ids = np.asarray(sorted(set(series_ids)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1

    train_ids = set(shuffled[:n_train].tolist())
    val_ids = set(shuffled[n_train:n_train + n_val].tolist())
    test_ids = set(shuffled[n_train + n_val:].tolist())

    if not test_ids:
        raise ValueError(
            f"Not enough series for train/validation/test split: {n_total}"
        )

    return train_ids, val_ids, test_ids


def build_masks(metadata, kind, train_ids, val_ids, test_ids):
    kind_mask = metadata["kind"].astype(str).eq(str(kind))

    train_mask = (
        kind_mask
        & metadata["series_id"].isin(train_ids)
    ).to_numpy()

    val_mask = (
        kind_mask
        & metadata["series_id"].isin(val_ids)
    ).to_numpy()

    test_mask = (
        kind_mask
        & metadata["series_id"].isin(test_ids)
    ).to_numpy()

    return train_mask, val_mask, test_mask


def finite_masks(y, train_mask, val_mask, test_mask):
    finite = np.isfinite(y)
    return (
        train_mask & finite,
        val_mask & finite,
        test_mask & finite,
    )


def fit_transform_features(x_train, x_val, x_test, pca_dim, seed):
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)

    if pca_dim is None or pca_dim <= 0:
        return x_train, x_val, x_test, None

    effective_pca_dim = min(
        pca_dim,
        x_train.shape[0] - 1,
        x_train.shape[1],
    )

    if effective_pca_dim >= x_train.shape[1]:
        return x_train, x_val, x_test, None

    pca = PCA(
        n_components=effective_pca_dim,
        svd_solver="randomized",
        random_state=seed,
    )
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)

    return x_train, x_val, x_test, pca

def evaluate_regression(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    alphas,
):
    best_alpha = None
    best_val_score = -np.inf

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)

        val_prediction = model.predict(x_val)
        val_score = r2_score(y_val, val_prediction)

        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha

    final_x = np.concatenate([x_train, x_val], axis=0)
    final_y = np.concatenate([y_train, y_val], axis=0)

    model = Ridge(alpha=best_alpha)
    model.fit(final_x, final_y)

    prediction = model.predict(x_test)

    return {
        "metric": "r2",
        "score": float(r2_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "best_alpha": float(best_alpha),
        "validation_score": float(best_val_score),
    }


def evaluate_classification(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    c_values,
):
    best_c = None
    best_val_score = -np.inf

    for c_value in c_values:
        model = LogisticRegression(
            C=c_value,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(x_train, y_train)

        val_prediction = model.predict(x_val)
        val_score = balanced_accuracy_score(
            y_val,
            val_prediction,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_c = c_value

    final_x = np.concatenate([x_train, x_val], axis=0)
    final_y = np.concatenate([y_train, y_val], axis=0)

    model = LogisticRegression(
        C=best_c,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(final_x, final_y)

    prediction = model.predict(x_test)

    return {
        "metric": "balanced_accuracy",
        "score": float(
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        ),
        "mae": np.nan,
        "best_alpha": float(best_c),
        "validation_score": float(best_val_score),
    }


def analyze_model(
    model_name,
    npz_path,
    metadata,
    layer_state_keys,
    kinds,
    states,
    targets,
    seeds,
    pca_dim,
    train_ratio,
    val_ratio,
    alphas,
    c_values,
):
    rows = []

    with np.load(npz_path) as npz_file:
        available = discover_feature_arrays(npz_file)

        for layer, state in layer_state_keys:
            if state not in states:
                continue

            if (layer, state) not in available:
                print(
                    f"[SKIP] {model_name}: "
                    f"layer={layer}, state={state} not found"
                )
                continue

            array_key = available[(layer, state)]

            print(
                f"Loading {model_name}: "
                f"layer={layer}, state={state}, key={array_key}"
            )

            features = np.asarray(
                npz_file[array_key],
                dtype=np.float32,
            )

            if len(features) != len(metadata):
                raise ValueError(
                    f"Feature/metadata row mismatch for {model_name}, "
                    f"{array_key}: {len(features)} != {len(metadata)}"
                )

            for kind in kinds:
                kind_ids = (
                    metadata.loc[
                        metadata["kind"].astype(str).eq(str(kind)),
                        "series_id",
                    ]
                    .unique()
                    .tolist()
                )

                for seed in seeds:
                    train_ids, val_ids, test_ids = split_series_ids(
                        kind_ids,
                        seed,
                        train_ratio,
                        val_ratio,
                    )

                    base_train, base_val, base_test = build_masks(
                        metadata,
                        kind,
                        train_ids,
                        val_ids,
                        test_ids,
                    )

                    for target in targets:
                        y = metadata[target].to_numpy(dtype=np.float64)

                        train_mask, val_mask, test_mask = finite_masks(
                            y,
                            base_train,
                            base_val,
                            base_test,
                        )

                        if (
                            train_mask.sum() == 0
                            or val_mask.sum() == 0
                            or test_mask.sum() == 0
                        ):
                            continue

                        x_train, x_val, x_test, pca = fit_transform_features(
                            features[train_mask],
                            features[val_mask],
                            features[test_mask],
                            pca_dim,
                            seed,
                        )

                        y_train = y[train_mask]
                        y_val = y[val_mask]
                        y_test = y[test_mask]

                        if target in CLASSIFICATION_TARGETS:
                            y_train = y_train.astype(np.int64)
                            y_val = y_val.astype(np.int64)
                            y_test = y_test.astype(np.int64)

                            # すべての分割に複数クラスが必要
                            if (
                                len(np.unique(y_train)) < 2
                                or len(np.unique(y_val)) < 2
                                or len(np.unique(y_test)) < 2
                            ):
                                continue

                            result = evaluate_classification(
                                x_train,
                                y_train,
                                x_val,
                                y_val,
                                x_test,
                                y_test,
                                c_values,
                            )
                        else:
                            result = evaluate_regression(
                                x_train,
                                y_train,
                                x_val,
                                y_val,
                                x_test,
                                y_test,
                                alphas,
                            )

                        rows.append({
                            "model": model_name,
                            "layer": int(layer),
                            "state": state,
                            "kind": kind,
                            "target": target,
                            "seed": int(seed),
                            "metric": result["metric"],
                            "score": result["score"],
                            "mae": result["mae"],
                            "best_alpha": result["best_alpha"],
                            "validation_score": result[
                                "validation_score"
                            ],
                            "pca_dim_requested": int(pca_dim),
                            "pca_dim_effective": (
                                int(pca.n_components_)
                                if pca is not None
                                else int(x_train.shape[1])
                            ),
                            "n_train_rows": int(train_mask.sum()),
                            "n_val_rows": int(val_mask.sum()),
                            "n_test_rows": int(test_mask.sum()),
                            "n_train_series": int(len(train_ids)),
                            "n_val_series": int(len(val_ids)),
                            "n_test_series": int(len(test_ids)),
                        })

    return rows


def summarize_results(all_scores):
    return (
        all_scores
        .groupby(
            [
                "model",
                "layer",
                "state",
                "kind",
                "target",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            mae_mean=("mae", "mean"),
            validation_score_mean=(
                "validation_score",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
    )


def build_corresponding_comparison(summary, layer_pairs):
    rows = []

    for depth_name, llama_layer, mamba_layer, relative_depth in layer_pairs:
        llama = summary[
            (summary["model"] == "llama")
            & (summary["layer"] == llama_layer)
        ].copy()

        mamba = summary[
            (summary["model"] == "mamba")
            & (summary["layer"] == mamba_layer)
        ].copy()

        merge_columns = [
            "state",
            "kind",
            "target",
            "metric",
        ]

        paired = llama.merge(
            mamba,
            on=merge_columns,
            suffixes=("_llama", "_mamba"),
            how="inner",
        )

        for _, row in paired.iterrows():
            difference = (
                row["score_mean_mamba"]
                - row["score_mean_llama"]
            )

            rows.append({
                "depth_name": depth_name,
                "relative_depth": relative_depth,
                "llama_layer": llama_layer,
                "mamba_layer": mamba_layer,
                "state": row["state"],
                "kind": row["kind"],
                "target": row["target"],
                "metric": row["metric"],
                "llama_score_mean": row["score_mean_llama"],
                "llama_score_std": row["score_std_llama"],
                "mamba_score_mean": row["score_mean_mamba"],
                "mamba_score_std": row["score_std_mamba"],
                "mamba_minus_llama": difference,
                "winner": (
                    "mamba"
                    if difference > 0
                    else "llama"
                    if difference < 0
                    else "equal"
                ),
            })

    return pd.DataFrame(rows)


def parse_layer_pairs(text):
    pairs = []

    for item in text.split(","):
        name, llama_layer, mamba_layer, relative_depth = item.split(":")

        pairs.append((
            name,
            int(llama_layer),
            int(mamba_layer),
            float(relative_depth),
        ))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run probe analysis by synthetic-series type "
            "and compare corresponding Llama/Mamba layers."
        )
    )
    parser.add_argument("--mamba_npz", required=True)
    parser.add_argument("--llama_npz", required=True)
    parser.add_argument("--mamba_metadata", default=None)
    parser.add_argument("--llama_metadata", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--layer_pairs",
        default=(
            "depth_25:7:16:0.25,"
            "depth_50:14:32:0.5,"
            "depth_100:28:64:1.0"
        ),
    )
    parser.add_argument(
        "--states",
        default="token_state,pre_token_state",
    )
    parser.add_argument(
        "--targets",
        default=",".join(
            REGRESSION_TARGETS + CLASSIFICATION_TARGETS
        ),
    )
    parser.add_argument(
        "--kinds",
        default=None,
        help="Comma-separated kinds. Default: all kinds.",
    )
    parser.add_argument(
        "--seeds",
        default="3407,3408,3409,3410,3411",
    )
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1.0,10.0,100.0",
    )
    parser.add_argument(
        "--c_values",
        default="0.01,0.1,1.0,10.0",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mamba_metadata_path = (
        args.mamba_metadata
        or f"{args.mamba_npz}.metadata.csv"
    )
    llama_metadata_path = (
        args.llama_metadata
        or f"{args.llama_npz}.metadata.csv"
    )

    mamba_metadata = pd.read_csv(mamba_metadata_path)
    llama_metadata = pd.read_csv(llama_metadata_path)

    validate_metadata(
        mamba_metadata,
        llama_metadata,
    )

    states = [
        value.strip()
        for value in args.states.split(",")
        if value.strip()
    ]

    targets = [
        value.strip()
        for value in args.targets.split(",")
        if value.strip()
    ]

    unknown_targets = sorted(
        set(targets)
        - set(REGRESSION_TARGETS + CLASSIFICATION_TARGETS)
    )

    if unknown_targets:
        raise ValueError(
            f"Unknown targets: {unknown_targets}"
        )

    if args.kinds is None:
        kinds = sorted(
            mamba_metadata["kind"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        kinds = [
            value.strip()
            for value in args.kinds.split(",")
            if value.strip()
        ]

    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    c_values = parse_float_list(args.c_values)
    layer_pairs = parse_layer_pairs(args.layer_pairs)

    llama_layer_states = [
        (llama_layer, state)
        for _, llama_layer, _, _ in layer_pairs
        for state in states
    ]

    mamba_layer_states = [
        (mamba_layer, state)
        for _, _, mamba_layer, _ in layer_pairs
        for state in states
    ]

    all_rows = []

    all_rows.extend(
        analyze_model(
            model_name="llama",
            npz_path=args.llama_npz,
            metadata=llama_metadata,
            layer_state_keys=llama_layer_states,
            kinds=kinds,
            states=states,
            targets=targets,
            seeds=seeds,
            pca_dim=args.pca_dim,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            alphas=alphas,
            c_values=c_values,
        )
    )

    all_rows.extend(
        analyze_model(
            model_name="mamba",
            npz_path=args.mamba_npz,
            metadata=mamba_metadata,
            layer_state_keys=mamba_layer_states,
            kinds=kinds,
            states=states,
            targets=targets,
            seeds=seeds,
            pca_dim=args.pca_dim,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            alphas=alphas,
            c_values=c_values,
        )
    )

    all_scores = pd.DataFrame(all_rows)

    if all_scores.empty:
        raise RuntimeError(
            "No probe scores were produced."
        )

    summary = summarize_results(all_scores)
    comparison = build_corresponding_comparison(
        summary,
        layer_pairs,
    )

    all_scores_path = output_dir / "kind_probe_scores_all_seeds.csv"
    summary_path = output_dir / "kind_probe_scores_summary.csv"
    comparison_path = (
        output_dir / "kind_corresponding_layer_comparison.csv"
    )

    all_scores.to_csv(all_scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    print("\nSeries kinds:")
    print(kinds)

    print("\nCorresponding-layer comparison:")
    print(
        comparison[
            [
                "depth_name",
                "state",
                "kind",
                "target",
                "llama_score_mean",
                "mamba_score_mean",
                "mamba_minus_llama",
                "winner",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(" -", all_scores_path)
    print(" -", summary_path)
    print(" -", comparison_path)


if __name__ == "__main__":
    main()
