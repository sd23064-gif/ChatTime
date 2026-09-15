"""
ecg_prepare.py

PTB-XL の波形を「ビンインデックス配列 (uint16)」+ メタデータ jsonl に変換する。

重要な設計判断:

  * 振幅は **レコードごとに正規化しない**。低電位, LVH 電位基準, ST 偏位量など
    絶対振幅そのものが診断情報なので、固定スケール mv_scale で割るだけにする。
    既定 5.0 mV -> 1 ビン = 2/10000 * 5.0 = 0.001 mV = 1 uV 相当。
  * ビン境界は Discretizer の centers から中点で作るので、
    既存 (train_base_mamba.py) の語彙と完全に一致する。
    => 学習済みチェックポイントをそのまま継続学習に使える。

出力:
  out_path/signals.npy    (N, L, T) uint16   ビンインデックス
  out_path/records.jsonl  1 行 1 レコードのメタデータ
  out_path/config.json    再現用の設定
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

LEADS_ALL = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_SETS = {
    "all12": LEADS_ALL,
    # III, aVR, aVL, aVF は I, II の線形結合なので落としても情報は減らない
    "indep8": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    "limb6": LEADS_ALL[:6],
    "ii": ["II"],
}

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------------------
# 語彙 (train_base_mamba.py と同一)
# ---------------------------------------------------------------------------

def build_centers(code_path: str, low: float, high: float, n_tokens: int) -> np.ndarray:
    sys.path.append(code_path)
    from utils.tools import Discretizer
    d = Discretizer(low_limit=low, high_limit=high, n_tokens=n_tokens)
    centers = np.asarray(d.centers[1:-1], dtype=np.float64)
    assert np.all(np.diff(centers) > 0), "centers は昇順であること"
    return centers


class Quantizer:
    """値 -> 最近傍 center のインデックス。"""

    def __init__(self, centers: np.ndarray):
        self.centers = centers
        self.edges = (centers[1:] + centers[:-1]) / 2.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self.centers[0], self.centers[-1])
        return np.searchsorted(self.edges, x).astype(np.uint16)


# ---------------------------------------------------------------------------
# メタデータ
# ---------------------------------------------------------------------------

def load_database(root: str, min_likelihood: float) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    diag = agg[agg["diagnostic"] == 1]

    def _map(codes, column):
        out = set()
        for k, v in codes.items():
            if v < min_likelihood:
                continue
            if k in diag.index:
                c = diag.loc[k, column]
                if isinstance(c, str) and c:
                    out.add(c)
        return sorted(out)

    df["superclass"] = df["scp_codes"].apply(lambda c: _map(c, "diagnostic_class"))
    df["subclass"] = df["scp_codes"].apply(lambda c: _map(c, "diagnostic_subclass"))
    return df


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_path", required=True, help="ChatTime 側 (utils.tools) のパス")
    p.add_argument("--ptbxl_path", required=True, help="ptbxl_database.csv のあるディレクトリ")
    p.add_argument("--out_path", required=True)

    p.add_argument("--fs", type=int, default=100, choices=[100, 500])
    p.add_argument("--leads", default="indep8", choices=sorted(LEAD_SETS))
    p.add_argument("--mv_scale", type=float, default=5.0,
                   help="この値で割って [-1,1] に入れる。レコード毎正規化はしない")
    p.add_argument("--highpass", type=float, default=0.5,
                   help="基線動揺除去のハイパスカットオフ Hz。0 で無効")
    p.add_argument("--min_likelihood", type=float, default=0.0)

    # 語彙 (元コードと同じ既定値)
    p.add_argument("--low_limit", type=float, default=-1)
    p.add_argument("--high_limit", type=float, default=1)
    p.add_argument("--n_tokens", type=int, default=10002)

    p.add_argument("--limit", type=int, default=-1, help="デバッグ用に先頭 N 件だけ")
    args = p.parse_args()

    import wfdb
    from scipy.signal import butter, sosfiltfilt

    os.makedirs(args.out_path, exist_ok=True)

    centers = build_centers(args.code_path, args.low_limit, args.high_limit, args.n_tokens)
    quant = Quantizer(centers)
    n_num = len(centers)

    df = load_database(args.ptbxl_path, args.min_likelihood)
    if args.limit > 0:
        df = df.iloc[: args.limit]

    leads = LEAD_SETS[args.leads]
    T = args.fs * 10                      # PTB-XL は全レコード 10 秒
    N, L = len(df), len(leads)
    print(f"records={N} leads={L} T={T} -> {N*L*T*2/1e9:.2f} GB (uint16)")

    sig_path = os.path.join(args.out_path, "signals.npy")
    arr = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.uint16,
                                    shape=(N, L, T))

    sos = None
    if args.highpass > 0:
        sos = butter(3, args.highpass / (args.fs / 2), btype="highpass", output="sos")

    clipped = 0
    total = 0
    records = []
    col = "filename_hr" if args.fs == 500 else "filename_lr"

    for i, (ecg_id, row) in enumerate(df.iterrows()):
        sig, meta = wfdb.rdsamp(os.path.join(args.ptbxl_path, row[col]))
        sig = np.asarray(sig, dtype=np.float64)          # (T, 12) [mV]
        names = list(meta["sig_name"])
        idx = [names.index(l) for l in leads]
        sig = sig[:T, idx]                               # (T, L)

        if sig.shape[0] < T:                             # 念のためゼロ詰め
            sig = np.pad(sig, ((0, T - sig.shape[0]), (0, 0)))

        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        if sos is not None:
            sig = sosfiltfilt(sos, sig, axis=0)

        x = sig / args.mv_scale
        clipped += int(np.sum(np.abs(x) > 1.0))
        total += x.size
        arr[i] = quant(x).T                              # (L, T)

        age = _clean(row.get("age"))
        if age is not None and age > 89:                 # PTB-XL は 89 歳超を匿名化
            age = ">89"
        elif age is not None:
            age = int(age)

        records.append({
            "ecg_id": int(ecg_id),
            "patient_id": _clean(row.get("patient_id")),
            "age": age,
            "sex": _clean(row.get("sex")),               # 0=male, 1=female
            "height": _clean(row.get("height")),
            "weight": _clean(row.get("weight")),
            "device": _clean(row.get("device")),
            "strat_fold": int(row["strat_fold"]),
            "scp_codes": {k: float(v) for k, v in row["scp_codes"].items()},
            "superclass": row["superclass"],
            "subclass": row["subclass"],
            "report": _clean(row.get("report")),
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N}")

    arr.flush()

    with open(os.path.join(args.out_path, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = dict(fs=args.fs, leads=leads, lead_set=args.leads, T=T,
               mv_scale=args.mv_scale, highpass=args.highpass,
               low_limit=args.low_limit, high_limit=args.high_limit,
               n_tokens=args.n_tokens, n_num=n_num,
               min_likelihood=args.min_likelihood, n_records=N)
    with open(os.path.join(args.out_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nclip 率 = {clipped/max(total,1)*100:.4f}%  "
          f"(0.1% を大きく超えるなら --mv_scale を上げる)")
    print(f"1 ビン = {(args.high_limit-args.low_limit)/n_num*args.mv_scale*1000:.2f} uV")
    print(f"saved to {args.out_path}")


if __name__ == "__main__":
    main()
