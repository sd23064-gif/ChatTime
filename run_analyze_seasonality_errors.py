import os
import pandas as pd


def main():
    detail_path = "outputs/chattime_tsqa_hf_details.csv"
    output_dir = "outputs/seasonality_analysis"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(detail_path)

    print("Loaded:", detail_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # =========================
    # 1. Seasonalityのみ抽出
    # =========================
    season_df = df[df["task"] == "Seasonality"].copy()

    print("\nSeasonality shape:", season_df.shape)

    if len(season_df) == 0:
        raise ValueError("Seasonality task is not found in detail CSV.")

    # =========================
    # 2. 全体正解率
    # =========================
    acc = season_df["correct"].mean()
    n = len(season_df)
    n_correct = season_df["correct"].sum()

    print("\nSeasonality Accuracy")
    print("accuracy:", acc)
    print("n_samples:", n)
    print("n_correct:", n_correct)

    # =========================
    # 3. labelごとの正解率
    # =========================
    label_summary = (
        season_df
        .groupby("label", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum")
        )
        .sort_values("accuracy")
    )

    print("\nAccuracy by label:")
    print(label_summary)

    label_summary.to_csv(
        os.path.join(output_dir, "seasonality_accuracy_by_label.csv"),
        index=False
    )

    # =========================
    # 4. sizeごとの正解率
    # =========================
    size_summary = (
        season_df
        .groupby("size", as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            n_correct=("correct", "sum")
        )
        .sort_values("size")
    )

    print("\nAccuracy by size:")
    print(size_summary)

    size_summary.to_csv(
        os.path.join(output_dir, "seasonality_accuracy_by_size.csv"),
        index=False
    )

    # =========================
    # 5. gold_choice → pred_choice の混同行列
    # =========================
    confusion_choice = pd.crosstab(
        season_df["gold_choice"],
        season_df["pred_choice"],
        rownames=["gold_choice"],
        colnames=["pred_choice"],
        dropna=False
    )

    print("\nConfusion matrix by choice:")
    print(confusion_choice)

    confusion_choice.to_csv(
        os.path.join(output_dir, "seasonality_confusion_by_choice.csv")
    )

    # =========================
    # 6. label → pred_choice の表
    # =========================
    confusion_label_pred = pd.crosstab(
        season_df["label"],
        season_df["pred_choice"],
        rownames=["gold_label"],
        colnames=["pred_choice"],
        dropna=False
    )

    print("\nConfusion matrix: label vs pred_choice")
    print(confusion_label_pred)

    confusion_label_pred.to_csv(
        os.path.join(output_dir, "seasonality_label_vs_pred_choice.csv")
    )

    # =========================
    # 7. 間違えたサンプルだけ抽出
    # =========================
    wrong_df = season_df[season_df["correct"] == 0].copy()

    print("\nWrong samples:", len(wrong_df))

    wrong_df.to_csv(
        os.path.join(output_dir, "seasonality_wrong_samples.csv"),
        index=False
    )

    # =========================
    # 8. 間違いパターンの集計
    # =========================
    wrong_pattern = (
        wrong_df
        .groupby(["label", "gold_choice", "pred_choice"], dropna=False, as_index=False)
        .agg(
            n_errors=("correct", "count")
        )
        .sort_values("n_errors", ascending=False)
    )

    print("\nWrong patterns:")
    print(wrong_pattern)

    wrong_pattern.to_csv(
        os.path.join(output_dir, "seasonality_wrong_patterns.csv"),
        index=False
    )

    # =========================
    # 9. モデル出力が抽出不能だったケース
    # =========================
    none_pred_df = season_df[season_df["pred_choice"].isna()].copy()

    print("\nPrediction extraction failures:", len(none_pred_df))

    none_pred_df.to_csv(
        os.path.join(output_dir, "seasonality_pred_choice_none.csv"),
        index=False
    )

    # =========================
    # 10. 間違い例を数件表示
    # =========================
    print("\nExample wrong samples:")
    show_cols = [
        "size",
        "label",
        "answer",
        "gold_choice",
        "model_output",
        "pred_choice",
        "question"
    ]

    existing_cols = [c for c in show_cols if c in wrong_df.columns]

    for i, row in wrong_df.head(5).iterrows():
        print("\n" + "=" * 80)
        for col in existing_cols:
            value = row[col]
            if col == "question":
                value = str(value)[:800]
            print(f"{col}: {value}")

    print("\nSaved analysis files to:", output_dir)


if __name__ == "__main__":
    main()