#!/usr/bin/env python3
"""将标注 Excel (列: 产品评价 / 加工结果) 转换为标准训练 CSV。

输入列: 产品评价, 加工结果
输出列: sentence, gold_tuples_count, predicted_tuples_count, correct_matches, gold_tuples, predicted_tuples

gold_tuples = [(方面词, 方面, 情感词, 情感倾向), ...]  (与旧版 csv 一致, 丢弃"是否有隐性情感")

标注规则修订:
  R1 包装规则: 方面 ⊆ {包装颜色,包装图案,包装质感} 且情感词为笼统评价
     (好看/大气/上档次等, 未提及具体颜色/图案/质感) → 方面改为"外观包装"
  R2 品牌规则: 方面 ⊆ {品牌定位,品牌知名度,品牌美誉度} 时, 按情感词信号重判类:
     适合谁/场景/角色→品牌定位; 出名程度→品牌知名度; 口碑形象→品牌美誉度
"""
import re
import ast
import pandas as pd
from collections import Counter

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/source.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/converted.csv"
CHANGE_REPORT = sys.argv[3] if len(sys.argv) > 3 else "data/rule_changes.csv"

# ═══ 解析: 逐项扫描, 字段名作锚点 (情感词内可含分号/换行/逗号) ═══
ITEM_RE = re.compile(
    r"(?:\d+\s*[.．、])?\s*方面词：(.*?)，方面：(.*?)，情感词：(.*?)，情感倾向：(.*?)，是否有隐性情感：(.*?)(?=；|$)",
    re.DOTALL,
)

# ═══ R1 包装规则关键词 ═══
PKG_FINE = {"包装颜色", "包装图案", "包装质感"}
PKG_SPECIFIC = (  # 提到具体包装属性 → 保持细粒度
    "颜色", "色调", "配色", "色彩", "色系", "主色调", "底色", "基调",
    "图案", "花纹", "纹路", "字样", "印刷", "画", "字", "主题",
    "质感", "材质", "手感", "触感", "磨砂", "烫金", "压纹", "工艺",
    "镭射", "凹印", "立体感", "触觉",
    "红", "金", "黄", "蓝", "绿", "白", "黑", "紫", "银", "棕", "灰", "粉", "橙",
)
PKG_VAGUE = (  # 笼统评价 → 收敛到 外观包装
    "好看", "漂亮", "大气", "高档", "高端", "上档次", "奢华", "精美", "精致",
    "高级", "简约", "新颖", "美观", "气派", "档次", "颜值", "品味", "辨识度",
    "用心", "不错", "喜欢", "高大上", "大方", "华丽", "时髦", "潮流", "夺目",
    "抢眼", "亮眼", "养眼", "耐看", "讲究", "有面", "贵气", "时尚", "雅致",
    "眼前一亮", "耳目一新", "用心", "精致感", "高级感", "设计感", "很好看",
)

# ═══ R2 品牌规则关键词 (依据 品牌形象易混淆点改动.md) ═══
BRAND_FINE = {"品牌定位", "品牌知名度", "品牌美誉度"}
BRAND_SIGNALS = {
    "品牌定位": (  # 适合谁 / 适合什么场景 / 扮演什么角色
        "适合", "送礼", "应酬", "面子", "撑场面", "新手", "老用户", "年轻人",
        "尝鲜", "长期抽", "拿得出手", "待客", "送人", "有面子", "招待",
        "社交", "身份", "档次", "高端", "上档次", "礼品装", "礼盒",
        "商务", "聚会", "请客", "排面", "场景",
    ),
    "品牌知名度": (  # 有多出名 / 认不认得 / 名气大不大
        "名气", "知名", "老牌子", "老品牌", "有名", "大品牌", "著名",
        "全国", "都知道", "认识", "听说过", "名头", "牌子响", "出圈",
        "名声", "家喻户晓", "国民", "耳熟能详", "驰名", "出不出名",
    ),
    "品牌美誉度": (  # 口碑好不好 / 形象好不好 / 大家评价高不高
        "口碑", "认可", "信赖", "信任", "风评", "都说好", "评价", "形象",
        "印象", "喜欢", "好评", "推荐", "信得过", "值得", "满意", "佳",
        "棒", "赞", "不错", "好", "良心", "放心", "靠谱", "品质",
        "底蕴", "向往", "好评",
    ),
}


def parse_annotation(text: str):
    """解析一行加工结果 → [(方面词, 方面, 情感词, 情感倾向), ...]"""
    s = str(text).strip().strip("｛{}｝ ")
    if not s:
        return []
    triples = []
    for m in ITEM_RE.finditer(s):
        aspect, cat, op, pol, _ = m.groups()
        triples.append((aspect.strip(), cat.strip(), op.strip(), pol.strip()))
    return triples


def apply_rules(triples):
    """R1 包装 + R2 品牌 标注规则修订; 返回 (新triples, 改动列表)"""
    new_triples = []
    changes = []
    for t in triples:
        aspect, cat, op, pol = t
        cats = set(c.strip() for c in cat.split(","))
        new_cat = cat
        # R1: 包装笼统评价 → 外观包装
        if cats and cats <= PKG_FINE:
            has_specific = any(k in op for k in PKG_SPECIFIC)
            has_vague = any(k in op for k in PKG_VAGUE)
            if not has_specific and has_vague:
                new_cat = "外观包装"
        # R2: 品牌类信号重判
        elif cats and cats <= BRAND_FINE:
            hits = {g for g, kws in BRAND_SIGNALS.items() if any(k in op for k in kws)}
            if len(hits) == 1:
                target = next(iter(hits))
                if target not in cats:
                    new_cat = target
        if new_cat != cat:
            changes.append((aspect, cat, new_cat, op, pol))
        new_triples.append((aspect, new_cat, op, pol))
    return new_triples, changes


def main():
    df = pd.read_excel(SRC)
    print(f"读取 {SRC}: {len(df)} 行")

    rows, all_changes, parse_fail = [], [], 0
    for idx, row in df.iterrows():
        sentence = str(row["产品评价"]).strip()
        triples = parse_annotation(row["加工结果"])
        if not triples:
            parse_fail += 1
            if parse_fail <= 10:
                print(f"  [行 {idx}] 解析为空: {str(row['加工结果'])[:80]}")
            continue
        triples, changes = apply_rules(triples)
        all_changes.extend((idx, aspect, old, new, op, pol)
                           for aspect, old, new, op, pol in changes)
        rows.append({
            "sentence": sentence,
            "gold_tuples_count": len(triples),
            "predicted_tuples_count": 0,
            "correct_matches": 0,
            "gold_tuples": str(triples),
            "predicted_tuples": "[]",
        })

    out_df = pd.DataFrame(rows, columns=[
        "sentence", "gold_tuples_count", "predicted_tuples_count",
        "correct_matches", "gold_tuples", "predicted_tuples",
    ])
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")

    # 规则修订报告
    chg_df = pd.DataFrame(all_changes, columns=["行号", "方面词", "原方面", "新方面", "情感词", "情感倾向"])
    chg_df.to_csv(CHANGE_REPORT, index=False, encoding="utf-8-sig")

    # 统计
    print(f"\n写出 {OUT}: {len(out_df)} 行")
    print(f"解析失败/空行: {parse_fail}")
    print(f"三元组总数: {out_df['gold_tuples_count'].sum()}")
    pol = Counter()
    for g in out_df["gold_tuples"]:
        for t in ast.literal_eval(g):
            pol[t[3]] += 1
    print("情感倾向分布:", dict(pol))
    n_r1 = sum(1 for c in all_changes if c[2] in PKG_FINE and c[3] == "外观包装")
    n_r2 = sum(1 for c in all_changes if c[2] in BRAND_FINE)
    print(f"R1 包装规则改动: {n_r1} 条 (方面→外观包装)")
    print(f"R2 品牌规则改动: {n_r2} 条 (品牌类重判)")
    print(f"修订报告: {CHANGE_REPORT}")

    # 示例 (两种规则各10条)
    r1_ex = [c for c in all_changes if c[2] in PKG_FINE and c[3] == "外观包装"][:10]
    r2_ex = [c for c in all_changes if c[2] in BRAND_FINE][:15]
    print("\n=== R1 示例 (包装笼统→外观包装) ===")
    for idx, aspect, old, new, op, pol in r1_ex:
        print(f"  [行{idx}] {old} → {new} | 情感词: {op[:40]}")
    print("\n=== R2 示例 (品牌重判) ===")
    for idx, aspect, old, new, op, pol in r2_ex:
        print(f"  [行{idx}] {old} → {new} | 情感词: {op[:40]}")


if __name__ == "__main__":
    main()
