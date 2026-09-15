"""
ecg_prepare.py

PTB-XL の波形を「数値トークンのインデックス (uint16)」+ メタデータに変換する。

量子化の基準は stage-0 のトークナイザから直接読む (ecg_vocab.numeric_vocab)。
Discretizer を再構築しないので、prec などの設定ズレで語彙が食い違う事故がない。

重要な設計判断:
  * 振幅は **レコードごとに正規化しない**。低電位, Sokolow-Lyon 電位基準,
    ST 偏位量など絶対振幅そのものが診断情報なので、固定スケールで割るだけにする。
    既定 5.0 mV -> 1 ビン = 2/10000 * 5.0 mV = 1 uV 相当。

出力:
  out_path/signals.npy    (N, L, T) uint16
  out_path/records.jsonl
  out_path/config.json
"""

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

from ecg_vocab import Quantizer, describe, numeric_vocab

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", required=True,
                   help="stage-0 の出力ディレクトリ (数値トークン追加済み)")
    p.add_argument("--ptbxl_path", required=True)
    p.add_argument("--out_path", required=True)

    p.add_argument("--fs", type=int, default=100, choices=[100, 500])
    p.add_argument("--leads", default="indep8", choices=sorted(LEAD_SETS))
    p.add_argument("--mv_scale", type=float, default=5.0,
                   help="この値で割って値域に収める。レコード毎正規化はしない")
    p.add_argument("--highpass", type=float, default=0.5,
                   help="基線動揺除去のカットオフ Hz。0 で無効")
    p.add_argument("--min_likelihood", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=-1)
    args = p.parse_args()

    import wfdb
    from scipy.signal import butter, sosfiltfilt
    from transformers import AutoTokenizer

    os.makedirs(args.out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                              trust_remote_code=True)
    num_start, values, nan_id = numeric_vocab(tokenizer)
    describe(num_start, values, nan_id, len(tokenizer))
    quant = Quantizer(values)

    df = load_database(args.ptbxl_path, args.min_likelihood)
    if args.limit > 0:
        df = df.iloc[: args.limit]

    leads = LEAD_SETS[args.leads]
    T = args.fs * 10                      # PTB-XL は全レコード 10 秒
    N, L = len(df), len(leads)
    print(f"\nrecords={N} leads={L} T={T} -> {N*L*T*2/1e9:.2f} GB (uint16)")

    arr = np.lib.format.open_memmap(os.path.join(args.out_path, "signals.npy"),
                                    mode="w+", dtype=np.uint16, shape=(N, L, T))

    sos = None
    if args.highpass > 0:
        sos = butter(3, args.highpass / (args.fs / 2), btype="highpass", output="sos")

    clipped = total = 0
    records = []
    col = "filename_hr" if args.fs == 500 else "filename_lr"

    for i, (ecg_id, row) in enumerate(df.iterrows()):
        sig, meta = wfdb.rdsamp(os.path.join(args.ptbxl_path, row[col]))
        sig = np.asarray(sig, dtype=np.float64)          # (T, 12) [mV]
        names = list(meta["sig_name"])
        sig = sig[:T, [names.index(l) for l in leads]]
        if sig.shape[0] < T:
            sig = np.pad(sig, ((0, T - sig.shape[0]), (0, 0)))

        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        if sos is not None:
            sig = sosfiltfilt(sos, sig, axis=0)

        x = sig / args.mv_scale
        clipped += int(np.sum((x < values[0]) | (x > values[-1])))
        total += x.size
        arr[i] = quant(x).T

        age = _clean(row.get("age"))
        if age is not None and age > 89:                 # 89 歳超は匿名化済み
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
               n_num=len(values), value_min=float(values[0]),
               value_max=float(values[-1]),
               tokenizer_path=os.path.abspath(args.tokenizer_path),
               min_likelihood=args.min_likelihood, n_records=N)
    with open(os.path.join(args.out_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    bw = float(np.diff(values).mean()) * args.mv_scale * 1000
    print(f"\nclip 率 = {clipped/max(total,1)*100:.4f}%  "
          f"(0.1% を大きく超えるなら --mv_scale を上げる)")
    print(f"1 ビン = {bw:.2f} uV")
    print(f"saved to {args.out_path}")


if __name__ == "__main__":
    main()
