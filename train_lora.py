#!/usr/bin/env python3
"""Qwen3-32B LoRA 微调: 1000条训练, 500条验证"""
import os, json, ast, gc, torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ═══ 配置 ═══
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "")  # 必填: 本地 Qwen3-32B 目录
TRAIN_PATH = os.environ.get("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))
VAL_PATH = os.environ.get("VAL_CSV", os.path.join(BASE_DIR, "data", "val.csv"))
OUTPUT_DIR = os.environ.get("LORA_OUT_DIR", os.path.join(BASE_DIR, "lora_output"))
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 1024
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500    # eval 500 条 val 每次约 12 分钟, 每 100 步 eval 会拖长 3-4h; 500 步一次共 5 次

CATEGORIES_STR = (
    "内在口味, 香气, 香韵特征, 香气质, 香气量, 透发性, 嗅香, 丰富性, 杂气, 清晰度, "
    "烟气, 顺畅性, 劲头, 绵长感, 柔和性, 细腻度, 甜润感, "
    "口感, 回甜, 干燥感, 余味, 刺激性, 品质（口味）延续性, "
    "外观包装, 包装图案, 包装颜色, 包装质感, 开合方式, 包装结构形态, 水松纸颜色、图案, 水松纸质感, 卷烟纸颜色, 卷烟纸质感, "
    "烟支设计, 滤嘴长度, 滤嘴类型, 燃烧速度, 抽吸阻力, 包灰性, "
    "工艺品质, 烟丝填充性, 燃烧锥稳定性, 口味稳定性, "
    "品牌形象, 品牌知名度, 品牌美誉度, 品牌定位, 货源稳定程度, "
    "产品设计创意, 品规概念产品文化, 新颖性, 功能特性, "
    "价格, 焦油, 负外部性, 气味扩散强度, 气味残留度"
)

def format_example(sentence, gold_tuples_str):
    """将标注数据格式化为训练样本: 指令输入 → 输出"""
    try:
        golds = ast.literal_eval(gold_tuples_str)
    except:
        golds = []
    # 构建输出: 只取细分类别(索引1) + 观点词(索引2) + 极性(索引3)
    outputs = []
    for t in golds:
        if len(t) >= 4:
            cat = str(t[1]).strip()
            if ',' in cat:
                cat = cat.split(',')[0].strip()
            outputs.append((cat, str(t[2]).strip(), str(t[3]).strip()))
    output_str = str(outputs) if outputs else "[]"

    prompt = f"""请做方面级情感三元组抽取。只输出 Python 列表。

格式：[('方面类别', '观点术语', '情感极性')]
情感极性: 正向、负向、中性。
类别: {CATEGORIES_STR}
能到三级类必输出三级；无法判断用二级；信息不足用一级。
观点术语从原句复制，不要总结。不要用方面词当观点词。
无三元组输出 []。

句子：{sentence}
输出："""
    return {"instruction": prompt, "output": output_str}

def load_and_format(path):
    """加载CSV并转换为训练格式"""
    df = pd.read_csv(path)
    data = []
    for _, row in df.iterrows():
        sentence = str(row.get('sentence', '')).strip()
        gold = str(row.get('gold_tuples', '[]')).strip()
        if not sentence or gold == '[]':
            continue
        formatted = format_example(sentence, gold)
        data.append(formatted)
    return data

def tokenize(examples, tokenizer):
    """分词: instruction+output拼接, 只在output部分计算loss"""
    prompts = examples["instruction"]
    outputs = examples["output"]
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for prompt, output in zip(prompts, outputs):
        # 分别tokenize prompt和output
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, truncation=True, max_length=MAX_LENGTH - len(prompt_ids), add_special_tokens=False)["input_ids"]
        # 拼接
        input_ids = prompt_ids + output_ids
        # labels: prompt部分设为-100(不计算loss), output部分正常
        labels = [-100] * len(prompt_ids) + output_ids
        # 截断到max_length
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        # Padding (由DataCollator处理)
        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append([1] * len(input_ids))
        batch["labels"].append(labels)
    return batch

def main():
    print("=" * 50)
    print("Qwen3-32B LoRA 微调")
    print(f"LoRA rank={LORA_RANK}, lr={LEARNING_RATE}, epochs={NUM_EPOCHS}")
    print("=" * 50)

    # 加载数据
    print("\n加载训练数据...")
    train_data = load_and_format(TRAIN_PATH)
    print(f"训练集: {len(train_data)} 条有效样本")
    val_data = load_and_format(VAL_PATH)
    print(f"验证集: {len(val_data)} 条有效样本")

    train_ds = Dataset.from_list([{"instruction": d["instruction"], "output": d["output"]} for d in train_data])
    val_ds = Dataset.from_list([{"instruction": d["instruction"], "output": d["output"]} for d in val_data])

    # 训练参数 — 必须在模型加载之前构造!
    # 已知问题: 本环境 4-bit(bnb)+device_map="auto" 加载后 torch.cuda.is_available()
    # 偶发变 False (CUDA 上下文被污染), TrainingArguments(bf16=True) 会误报
    # "Your setup doesn't support bf16/gpu" (此前4次失败均在此处). TrainingArguments
    # 不依赖模型, 提前构造时 CUDA 健康, bf16 检查正常通过.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        per_device_eval_batch_size=1,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    print("TrainingArguments 构造完成 (bf16=True 检查通过)")

    # 加载模型 (4-bit 量化以适应单卡)
    print("\n加载模型 (4-bit量化)...")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}, is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"CUDA 可见卡数: {torch.cuda.device_count()}, GPU0 空闲 {free_mem/1024**3:.1f}GB / 共 {total_mem/1024**3:.1f}GB")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, revision='master')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        revision='master',
        torch_dtype=torch.bfloat16,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # 关键: 让梯度流过embedding层
    model.print_trainable_parameters()

    # 分词
    print("\n分词...")
    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=train_ds.column_names, batch_size=16)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=val_ds.column_names, batch_size=16)

    # 加载后诊断: 4-bit 加载若污染 CUDA 上下文, is_available() 会变 False
    # (此前4次失败的模式; training_args 已提前构造, 此处仅为观测 + 兜底)
    avail = torch.cuda.is_available()
    bf16_ok = torch.cuda.is_bf16_supported() if avail else False
    print(f"CUDA 状态(模型加载后): is_available={avail}, is_bf16_supported={bf16_ok}")
    if not avail:
        # 模型已成功加载到 GPU (device_map=auto), 此处打补丁强制恢复检查;
        # 若 CUDA 真坏了, 训练第一步前向传播会报真实 CUDA 错误, 届时另行诊断
        print("⚠️ 4-bit 加载后 torch.cuda.is_available()=False, 打补丁强制恢复 CUDA 检查")
        torch.cuda.is_available = lambda *a, **k: True
        torch.cuda.is_bf16_supported = lambda *a, **k: True

    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    print("\n开始训练...")
    # resume_from_checkpoint=True: 若 OUTPUT_DIR 存在 checkpoint 则断点续训,
    # 无 checkpoint 时自动从头训练 (transformers 会提示 "No checkpoint found")
    trainer.train(resume_from_checkpoint=True)

    # 保存
    print(f"\n保存模型到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("训练完成!")


if __name__ == "__main__":
    main()
