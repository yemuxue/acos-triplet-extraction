# 方面类别三元组抽取

基于 **Qwen3-32B + LoRA 微调 + RAG 动态 Few-Shot + 规则后处理** 的方面类别三元组抽取（ACOS）项目。

输入一条用户评论，输出结构化三元组列表：

```
句子：入口顺滑，带劲，回甘明显，包装漂亮
输出：[('顺畅性', '顺滑', '正向'), ('劲头', '带劲', '正向'),
       ('回甜', '回甘明显', '正向'), ('包装图案', '漂亮', '正向')]
```

每个三元组 = **(方面类别, 观点词, 情感极性)**，其中类别采用**三级层级体系**：

| 层级 | 数量 | 示例 |
|------|------|------|
| 一级类 | 9 | 内在口味、外观包装、烟支设计、工艺品质、品牌形象、产品设计创意、价格、焦油、负外部性 |
| 二级类 | 12 | 内在口味 → 香气 / 烟气 / 口感 / 品质（口味）延续性 |
| 三级类 | 38 | 香气 → 香韵特征 / 香气质 / 香气量 / 透发性 / 嗅香 / 丰富性 / 杂气 / 清晰度 |

完整层级与词典见 `constants.py` 中 `ASPECT_CATEGORY_HIERARCHY`。

## 方法架构

```
输入句子
    │
    ├─→ FewShotRetriever (BGE-small-zh 编码训练池, 动态检索 Top-5 相似标注样本)
    │       └─ 相似样本拼入 prompt 作为示例 (RAG 动态 Few-Shot)
    │
    ├─→ Qwen3-32B + LoRA adapter (rank=16, 4-bit 量化推理)
    │       └─ Beam Search (num_beams=2) 确定性解码
    │       └─ 输出: [('类别', '观点词', '情感'), ...]
    │
    ├─→ 规则后处理流水线 (6 步, ~600 行)
    │   ├─ _correct_categories   类别模糊匹配纠正 (非法类别/极性兜底)
    │   ├─ _refine_category_label 类别下钻 (一级类 → 三级类)
    │   ├─ _refine_opinion_term  观点词清洗 (方面名词当观点词时回填)
    │   ├─ _refine_polarity      极性纠偏 (特殊表达, 如"微醺"=负向)
    │   ├─ _should_drop_...      误报过滤 (无触发词支撑即丢弃)
    │   └─ _deduplicate_...      父子层级去重 (删父留子)
    │
    └─→ 最终三元组
```

设计原则：**LLM 负责语义理解，规则负责精确控制**。LoRA 微调后模型语义判别能力显著提升，
后处理规则按错误分析持续"做减法"——删除欠预测召回、短句兜底等补偿规则并屏蔽 L3-L5 语义
匹配后，F1 不降反升（详见"版本演进"）。

## 版本演进与效果

| 版本 | 核心改动 | 一级 F1 | 二级 F1 | 三级 F1 |
|------|---------|:------:|:------:|:------:|
| v001 | Qwen3-14B + Greedy 解码 | 43.7% | 42.2% | 43.9% |
| v004 | Qwen3-32B + Beam Search + 静态 Few-Shot | 66.0% | 58.8% | 47.9% |
| v005 | + RAG 动态 Few-Shot (BGE embedding) | 69.9% | 65.3% | 56.0% |
| v007 | + LoRA 微调 (1000 条, rank=16) | 87.0% | 85.0% | 66.0% |
| v009 | + 规则清理 (删除欠预测召回/短句兜底) | 86.6% | 84.8% | 67.6% |
| v010 | + 简化匹配 (屏蔽 L3-L5 语义层) | 89.0% | 87.1% | 69.4% |
| **v011** | **训练集扩至 3000 条 + 全新独立评估集** | **81.7%** | **81.6%** | **73.5%** |

- v001–v010 在同一 500 条开发集上纵向对比；v011 更换为**全新独立标注的开发集**（与训练
  数据分源），绝对水平与历史不可直接比较，但为全项目最高。
- v011 完整三元组口径 P/R/F1 = 79.1 / 72.5 / 75.7；句子级全对率 44.8%。
- 观点词匹配中 **L1 精确相等占比 94.6%~96.2%**（LoRA 后模型输出与人工标注措辞高度对齐）。

## 目录结构

```
├── constants.py                类别层级、领域词典、匹配策略常量 (评估/后处理共用)
├── predict_triples_fuzzy.py    推理 + 规则后处理 + 严格评估 (含模型常驻服务模式)
├── train_lora.py               Qwen3-32B LoRA 微调 (4-bit, 支持断点续训)
├── convert_data.py             标注 Excel → 标准 CSV (含标注规则修订 R1/R2)
├── split_data.py               按行索引切分 train/val/dev (评估集零重叠)
├── check_gpu.sh                GPU 健康检查与自动选卡
├── run_predict_triples_fuzzy.sh 一键评估入口 (自动选卡 + 日志编号)
└── data/                       数据目录 (业务数据, 未随仓库公开, 见下)
```

## 数据说明

每行一条文本, CSV 列: `sentence, gold_tuples_count, gold_tuples, ...`
其中 `gold_tuples` 为四元组 Python 字面量:

```python
[('方面词', '方面类别', '观点词', '情感倾向'), ...]   # 情感倾向: 0=负, 1=中性, 2=正
```

模型侧将其投影为三元组 `(类别, 观点词, 极性)` 后训练/推理。历史数据中还包含公开基准
Restaurant-ACOS / Laptop-ACOS 的复用版本，用于早期方法验证。

> ⚠️ **本仓库不包含任何业务数据与模型权重**（合作业务数据、Qwen3-32B 权重、LoRA 适配器
> 均不随仓库发布）。如需复现流程，请自行准备任意 CSV（格式同上，500+ 条即可完成一次
> LoRA 微调与评估）。

## 快速开始

### 1. 环境与依赖

```bash
# 基础依赖 (建议 miniconda/venv)
pip install -r requirements.txt   # 见文件底部注释中的版本组合亦可
# 需要: transformers, peft, bitsandbytes, datasets, modelscope,
#       sentence-transformers, pandas, scikit-learn
```

### 2. 配置

所有路径可用环境变量覆盖（默认相对仓库根目录）：

| 环境变量 | 用途 | 默认 |
|---------|------|------|
| `QWEN_MODEL_PATH` | Qwen3-32B 本地模型目录 | **必填**（推理/训练入口会校验） |
| `TRAIN_CSV` / `VAL_CSV` / `EVAL_CSV` | 训练/验证/开发集 | `data/train.csv` 等 |
| `LORA_OUT_DIR` | LoRA 适配器输出/加载目录 | `lora_output` |
| `ACOS_BASE_DIR` / `ACOS_PYTHON` | shell 脚本基准目录 / Python 解释器 | 脚本所在目录 / `python3` |

### 3. 运行流程

```bash
# ① 数据准备: 标注 Excel → 标准 CSV; 按行索引切分
python convert_data.py data/source.xlsx data/converted.csv
python split_data.py data/converted.csv data        # 默认切 3000/500/500

# ② LoRA 微调 (单卡, 4-bit 量化训练, 断点续训自动开启)
QWEN_MODEL_PATH=/path/to/Qwen3-32B python train_lora.py

# ③ 评估 (500 条开发集, 三层指标 + 观点词匹配统计)
bash run_predict_triples_fuzzy.sh
# 或直接:
QWEN_MODEL_PATH=/path/to/Qwen3-32B python predict_triples_fuzzy.py \
    --sample-size 500 --log-path run.log
```

### 4. 常驻服务模式（多次评估免重复加载模型）

```bash
python predict_triples_fuzzy.py --resident-server          # 启动常驻进程 (模型只加载一次)
python predict_triples_fuzzy.py --resident-run --log-path r.log   # 复用常驻模型跑评估
python predict_triples_fuzzy.py --resident-stop            # 停止
```

## 评估口径（严格三元组匹配）

- **类别**：gold/pred 按其中更浅的层级投影后比较（三级评估仅取真实三级类）。
- **观点词**：L1 原始字符串精确相等 + L2 去标点双向包含；**否定冲突拦截**
  （"刺激" 与 "不刺激" 判为不匹配）。
- **极性**：精确匹配 + 同义词表兜底。
- **整句计数**：匈牙利算法求二分图最大匹配，避免重复消费同一预测。

指标经过独立复算验证：与主报告逐位一致、匹配宽松度可控（无系统性高估）。

## 项目亮点 / 工程要点

- **4-bit 量化 + LoRA 直接推理**：不 merge 权重（merge 会反量化至 fp16，显存 ~80GB 必
  OOM），量化+LoRA 等价推理仅需 ~20GB，单卡可跑。
- **训练可靠性**：支持断点续训；定位并绕过 4-bit 加载污染 CUDA 上下文导致的 bf16 误报
  （TrainingArguments 提前于模型加载构造）。
- **评估工程化**：500 条独立开发集零重叠、日志自动编号、GPU 自动调度、结果可复现。

---

*演示代码。若本项目源自企业合作，公开前请先确认不违反保密约定。*
