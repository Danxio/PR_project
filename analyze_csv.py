#!/usr/bin/env python3
"""
analyze_results.py

Wczytuje CSV (średnik jako separator, przecinek jako separator dziesiętny)
i generuje zwięzłe tabele/statystyki wymagane do sprawozdania.

Usage:
    python3 analyze_results.py batch_results.csv --out results_dir

Wyniki:
 - results_dir/summary_by_BS_{BS}.csv   (zbiorcza tabela dla danego BS)
 - results_dir/summary_all.csv          (zbiorcze agregaty)
 - results_dir/nwys_table.csv           (wyliczone N_wys per BS,R,mode,k)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from math import ceil

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Wejściowy CSV (średnik separator, przecinek dziesiętny)")
    p.add_argument("--out", "-o", default="results_tables", help="Katalog wyjściowy")
    p.add_argument("--nwys-threshold", type=float, default=0.02,
                   help="Próg względnej zmiany GFLOPS do wykrywania N_wys (domyślnie 0.02 -> 2%%)")
    args = p.parse_args()
    return args

def to_float_comma(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int,float)):
        return float(x)
    s = str(x).strip()
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def load_csv(path):
    # read using pandas with semicolon separator, don't coerce decimals automatically
    df = pd.read_csv(path, sep=';', dtype=str, keep_default_na=False)
    # normalize numeric columns that may use comma
    for col in ['avgKernelMs','gflops','cpuMs','cpuGflops']:
        if col in df.columns:
            df[col] = df[col].apply(to_float_comma)
    # parse ints
    for col in ['package','N','R','BS','k','nIter','sharedBytes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    # mode columns as str
    return df

def compute_ops(N, R):
    outW = N - 2*R
    outH = outW
    w = (2*R + 1)
    ops = outW * outH * (w * w)  # number of adds (approx FLOPs)
    return int(ops), int(outW), int(outH)

def estimate_bytes_mode(mode, N, R, BS, k):
    """
    Przybliżenie globalnych bajtów odczytu dla jednego wywołania kernela (na całą macierz).
    - mode 'a'/'b': każdy output czyta window elementów -> reads = outPixels * window
    - mode 'c'/'d': zakładamy, że tile jest ładowane raz na blok:
         blocks_x = ceil(outWidth / (BS*k))
         blocks_y = ceil(outHeight / BS)
         tileWidth = BS*k + 2*R
         tileHeight = BS + 2*R
         total_loads = blocks_x * blocks_y * tileWidth * tileHeight
      oraz zapisów = outPixels
    Wynik w bajtach (float32).
    Uwaga: przybliżenie — nie uwzględnia L2 cache/overlap.
    """
    outPixels = (N - 2*R) * (N - 2*R)
    window = (2*R + 1) * (2*R + 1)
    bytes_per_float = 4.0
    if mode.lower() in ['a','b']:
        reads = outPixels * window
        writes = outPixels
        total_bytes = (reads + writes) * bytes_per_float
        return total_bytes
    else:
        tileWidth = BS * k + 2 * R
        tileHeight = BS + 2 * R
        blocks_x = ceil((N - 2*R) / (BS * k))
        blocks_y = ceil((N - 2*R) / BS)
        total_loads = blocks_x * blocks_y * tileWidth * tileHeight
        writes = outPixels
        total_bytes = (total_loads + writes) * bytes_per_float
        return total_bytes

def main():
    args = parse_args()
    in_csv = args.csv
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    df = load_csv(in_csv)

    # Ensure required columns present
    required = ['N','R','BS','k','mode','avgKernelMs','gflops','cpuMs','valid','sharedBytes']
    for c in required:
        if c not in df.columns:
            print("Brak kolumny w CSV:", c, file=sys.stderr)
    # convert mode to lowercase for grouping consistency
    df['mode'] = df['mode'].astype(str)
    df['actualMode'] = df.get('actualMode', df['mode']).astype(str)

    # convert numeric columns if not already
    df['N'] = pd.to_numeric(df['N'], errors='coerce').astype('Int64')
    df['R'] = pd.to_numeric(df['R'], errors='coerce').astype('Int64')
    df['BS'] = pd.to_numeric(df['BS'], errors='coerce').astype('Int64')
    df['k'] = pd.to_numeric(df['k'], errors='coerce').astype('Int64')
    df['sharedBytes'] = pd.to_numeric(df['sharedBytes'], errors='coerce').astype('Int64')
    df['valid_bool'] = df['valid'].astype(str).str.upper() == 'PASS'

    # Group by BS,N,R,k,mode
    group_cols = ['BS','N','R','k','mode','actualMode']
    agg = df.groupby(group_cols).agg(
        n_runs = ('avgKernelMs','count'),
        avgKernelMs_med = ('avgKernelMs','median'),
        avgKernelMs_mean = ('avgKernelMs','mean'),
        avgKernelMs_std = ('avgKernelMs','std'),
        gflops_med = ('gflops','median'),
        gflops_mean = ('gflops','mean'),
        cpuMs_med = ('cpuMs','median'),
        sharedBytes_med = ('sharedBytes','median'),
        pass_rate = ('valid_bool','mean')
    ).reset_index()

    # compute ops/out dims and estimated bandwidth/AI
    est_bw_list = []
    ai_list = []
    ops_list = []
    outW_list = []
    outH_list = []
    for idx, row in agg.iterrows():
        N = int(row['N'])
        R = int(row['R'])
        BS = int(row['BS'])
        k = int(row['k'])
        mode = row['mode']
        ops, outW, outH = compute_ops(N,R)
        ops_list.append(ops)
        outW_list.append(outW)
        outH_list.append(outH)
        avgKernelMs = row['avgKernelMs_med']
        if np.isnan(avgKernelMs) or avgKernelMs <= 0:
            est_bw_list.append(np.nan)
            ai_list.append(np.nan)
            continue
        bytes_est = estimate_bytes_mode(mode, N, R, BS, k)  # bytes per full kernel
        seconds = float(avgKernelMs) / 1000.0
        est_bw = bytes_est / seconds / 1e9  # GB/s
        est_bw_list.append(est_bw)
        # AI = FLOPs / bytes
        ai = ops / bytes_est if bytes_est > 0 else np.nan
        ai_list.append(ai)

    agg['ops'] = ops_list
    agg['outWidth'] = outW_list
    agg['outHeight'] = outH_list
    agg['estBandwidth_GBps'] = est_bw_list
    agg['arithIntensity_FlopPerByte'] = ai_list

    # write per-BS tables
    bs_values = sorted(agg['BS'].dropna().unique())
    for bs in bs_values:
        sub = agg[agg['BS'] == bs].copy()
        outfile = os.path.join(out_dir, f"summary_by_BS_{bs}.csv")
        sub.to_csv(outfile, index=False, sep=';')
        print("Zapisano:", outfile)

    # overall summary
    outfile_all = os.path.join(out_dir, "summary_all.csv")
    agg.to_csv(outfile_all, index=False, sep=';')
    print("Zapisano:", outfile_all)

    # detect N_wys per (BS,R,mode,k)
    nws_rows = []
    group_for_nwys = agg.groupby(['BS','R','mode','k'])
    for name, grp in group_for_nwys:
        bs, r, mode, k = name
        sub = grp[['N','gflops_med']].dropna().sort_values('N')
        if len(sub) < 2:
            nws = np.nan
        else:
            med_vals = sub['gflops_med'].values
            Ns = sub['N'].values
            nws = np.nan
            for i in range(len(Ns)-1):
                prev = med_vals[i]
                nxt = med_vals[i+1]
                if prev == 0:
                    continue
                rel = (nxt - prev) / prev
                if rel <= args.nwys_threshold:
                    nws = int(Ns[i+1])
                    break
        nws_rows.append({'BS':bs, 'R':r, 'mode':mode, 'k':int(k), 'N_wys':nws})

    df_nwys = pd.DataFrame(nws_rows)
    nws_out = os.path.join(out_dir, "nwys_table.csv")
    df_nwys.to_csv(nws_out, index=False, sep=';')
    print("Zapisano:", nws_out)

    print("Analiza zakonczona. Pliki w:", out_dir)
    print("Uwaga: estymowane pasmo i AI to przybliżenia. Dla dokładnej analizy profiluj ncu/nvprof.")

if __name__ == "__main__":
    main()
