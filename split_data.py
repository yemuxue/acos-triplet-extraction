#!/usr/bin/env python3
"""数据划分: 训练集 / 验证集 / 开发评估集 (按行索引切分, 评估集不参与训练)"""
import os
import ast
import pandas as pd
from collections import Counter

import sys
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/converted.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "data"

# 划分策略 (示例: 3000/500/500 切分)
TRAIN_START, TRAIN_END = 0, 3000       # 3000条
VAL_START,   VAL_END   = 3000, 3500     # 500条
DEV_START,   DEV_END   = 3500, 4000     # 500条

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"总数据: {len(df)} 条")

train = df.iloc[TRAIN_START:TRAIN_END].copy()
val = df.iloc[VAL_START:VAL_END].copy()
dev = df.iloc[DEV_START:DEV_END].copy()

train.to_csv(f"{OUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUT_DIR}/val.csv", index=False)
dev.to_csv(f"{OUT_DIR}/dev.csv", index=False)

print(f"训练集:   {len(train)} 条 → data/train.csv  (索引 {TRAIN_START}-{TRAIN_END-1})")
print(f"验证集:   {len(val)} 条 → data/val.csv    (索引 {VAL_START}-{VAL_END-1})")
print(f"开发评估: {len(dev)} 条 → data/dev.csv    (索引 {DEV_START}-{DEV_END-1})")
print(f"预留:     {len(df) - DEV_END} 条 (索引 {DEV_END}-{len(df)-1})")
print()

# 统计训练集类别分布
cat_counter = Counter()
for gold_str in train['gold_tuples']:
    try:
        for t in ast.literal_eval(gold_str):
            if len(t) >= 2:
                cat = str(t[1]).strip()
                if ',' in cat:
                    for c in cat.split(','):
                        cat_counter[c.strip()] += 1
                else:
                    cat_counter[cat] += 1
    except Exception:
        continue

print("训练集类别分布 (Top 20):")
for cat, cnt in cat_counter.most_common(20):
    print(f"  {cat}: {cnt}")

rare = [(c, n) for c, n in cat_counter.items() if n < 5]
if rare:
    print(f"\n⚠️  低频类别 (样本<5): {len(rare)} 个")
    for c, n in sorted(rare, key=lambda x: x[1]):
        print(f"  {c}: {n}")
