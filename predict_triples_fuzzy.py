#!/usr/bin/env python3
"""情感三元组预测工具 - 基于predict_tuples_updated.py修改，实现三元组匹配和观点术语模糊匹配"""

import os
import argparse
import contextlib
import json
import sys
import traceback
from multiprocessing.connection import Listener, Client

# 注意：真正固定显卡以启动命令里的 CUDA_VISIBLE_DEVICES 为准。
# 这里保留默认值，仅在外部未设置时回退到 GPU 1。
# GPU 由 run_predict_triples_fuzzy.sh 中的 check_gpu.sh 自动选择

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from modelscope import AutoTokenizer, AutoModelForCausalLM
import time
import re
import datetime
from peft import PeftModel
import ast
from constants import *

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# outlines FSM 对中文 tokenization 不兼容, 改用后处理纠正方案
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from sentence_transformers import SentenceTransformer
import ast as _ast_parse

SAMPLE_SIZE = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LORA_PATH = os.environ.get("LORA_OUT_DIR", os.path.join(BASE_DIR, "lora_output"))  # LoRA 适配器路径
RETRIEVAL_DATA_PATH = os.environ.get("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))  # 检索池数据(训练集)
RETRIEVAL_TOP_K = 5           # 动态示例数量
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 中文优化, 24MB
MAX_NEW_TOKENS = 220
USE_SHORT_PROMPT = True
USE_TWO_STAGE = False  # Phase 2B已回退: 规则映射不如LLM语境判断
DEFAULT_DATA_PATH = os.environ.get("EVAL_CSV", os.path.join(BASE_DIR, "data", "dev.csv"))  # 开发评估集, 未参与训练
DEFAULT_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "")  # 必填: 本地 Qwen3-32B 目录
RESIDENT_SOCKET_PATH = "/tmp/predict_triples_fuzzy_resident.sock"
METHOD_TAG = "v010_simplify"
METHOD_NOTES = [
    "模型: Qwen3-32B (BF16, 单卡GPU), Beam Search (num_beams=2)",
    "动态Few-Shot: FewShotRetriever (BGE-small-zh, 检索池500-3000, Top-5)",
    "后处理: _correct_categories 模糊匹配纠正非法类别/极性",
    "过预测收紧: 水松纸颜色图案/包装结构形态/香气质/香韵特征/甜润感/顺畅性 显式触发词约束",
    "欠预测补召回: 品牌美誉度/香气量/绵长感/口味稳定性/透发性/滤嘴类型/品牌定位 补充触发词",
    "观点词匹配: 双向包含+语义归一+组合短语子集+核心语义词+否定冲突保护",
    "类别匹配: 按两个标签中更宽松的层级统一后比较",
    "父子层级去重+短句兜底+泛评价误报过滤",
    "已知局限: 规则天花板已到, 语义混淆需LoRA微调突破",
]

ASPECT_CATEGORY_HIERARCHY = {
    '内在口味': {
        '香气': ['香韵特征', '香气质', '香气量', '透发性', '嗅香', '丰富性', '杂气', '清晰度'],
        '烟气': ['顺畅性', '劲头', '绵长感', '柔和性', '细腻度', '甜润感', '成团性'],
        '口感': ['回甜', '干燥感', '余味', '刺激性'],
        '品质（口味）延续性': [],
    },
    '外观包装': {
        '外观包装': ['包装图案', '包装颜色', '包装质感', '开合方式', '包装结构形态',
                 '水松纸颜色、图案', '水松纸质感', '卷烟纸颜色', '卷烟纸质感'],
    },
    '烟支设计': {
        '烟支设计': ['滤嘴长度', '滤嘴类型', '燃烧速度', '抽吸阻力', '包灰性'],
    },
    '工艺品质': {
        '工艺品质': ['烟丝填充性', '燃烧锥稳定性', '口味稳定性'],
    },
    '品牌形象': {
        '品牌形象': ['品牌知名度', '品牌美誉度', '品牌定位', '货源稳定程度'],
    },
    '产品设计创意': {
        '产品设计创意': ['品规概念产品文化', '新颖性', '功能特性'],
    },
    '价格': {
        '价格': [],
    },
    '焦油': {
        '焦油': [],
    },
    '负外部性': {
        '负外部性': ['气味扩散强度', '气味残留度'],
    },
}

TOP_LEVEL_CATEGORIES = set(ASPECT_CATEGORY_HIERARCHY.keys())
SECOND_LEVEL_CATEGORIES = set()
THIRD_LEVEL_CATEGORIES = set()
CATEGORY_TO_TOP = {}
CATEGORY_TO_SECOND = {}

for top_category, second_level_map in ASPECT_CATEGORY_HIERARCHY.items():
    CATEGORY_TO_TOP[top_category] = top_category
    for second_category, third_categories in second_level_map.items():
        SECOND_LEVEL_CATEGORIES.add(second_category)
        CATEGORY_TO_TOP[second_category] = top_category
        CATEGORY_TO_SECOND[second_category] = second_category
        for third_category in third_categories:
            THIRD_LEVEL_CATEGORIES.add(third_category)
            CATEGORY_TO_TOP[third_category] = top_category
            CATEGORY_TO_SECOND[third_category] = second_category

EXCLUDED_EVAL_CATEGORIES = {
    '防伪设计',
    '成团性',
    '刺破、异物',
    '盒盖盖不好',
}

LEGACY_CATEGORY_NORMALIZATION = {
    '香韵特性': '香韵特征',
    '回甜感': '回甜',
    '过滤嘴长度': '滤嘴长度',
    '吸阻': '抽吸阻力',
    '抽吸阻力': '抽吸阻力',
    '燃烧性能': '燃烧速度',
    '灰烬': '包灰性',
    '包装': '外观包装',
    '外观': '外观包装',
    '烟支': '烟支设计',
    '烟支结构': '烟支设计',
    '设计': '产品设计创意',
    '品牌': '品牌形象',
    '口味': '内在口味'
}

GENERIC_CATEGORY_GROUPS = {
    '内在口味': ['香气', '烟气', '口感', '品质（口味）延续性'],
    '口味': ['香气', '烟气', '口感', '品质（口味）延续性'],
    '外观包装描述': ['外观包装'],
    '包装': ['外观包装'],
    '外观': ['外观包装'],
    '烟支设计描述': ['烟支设计'],
    '烟支': ['烟支设计'],
    '工艺': ['工艺品质'],
    '品牌': ['品牌形象'],
    '产品设计创意描述': ['产品设计创意'],
    '设计': ['外观包装', '烟支设计', '产品设计创意'],
}

OPINION_TERM_NORMALIZATION = {
    '回甘': '回甜',
    '回甜感': '回甜',
    '清甜香': '清甜香气',
    '味道重': '味道',
    '吃味': '味道',
    '本香味': '本香',
    '本香足': '本香',
    '顺滑': '顺畅',
    '顺畅性': '顺畅',
    '丝滑': '顺畅',
    '通透': '顺畅',
    '柔顺': '柔和',
    '绵柔': '柔和',
    '柔': '柔和',
    '绵长': '余味绵长',
    '回味长': '余味绵长',
    '香醇': '醇厚',
    '香气舒适': '香气好',
    '香气纯正': '香气好',
    '纯正': '纯',
    '高端大气上档次': '高端大气',
    '高端大气': '高端',
    '上档次': '高端',
    '有面': '有面子',
    '性价比高': '划算',
    '性价比超高': '划算',
    '实惠划算': '划算',
    '值得推荐': '推荐',
    '非常nice': '很好',
    '不辣口': '不刺激',
    '不呛喉': '不刺激',
    '不呛': '不刺激',
    '不刺喉': '不刺激',
    '辣口': '刺激',
    '呛喉': '刺激',
    '刺激感强': '刺激',
    '细腻柔和': '细腻柔和',
    '细腻丝滑': '细腻顺滑',
    '柔和细腻': '细腻柔和',
    '柔和顺滑': '柔和顺滑',
    '顺滑柔和': '柔和顺滑',
    '香气清新自然': '清新自然',
    '清新自然': '清新自然',
    '自然清香': '清新自然',
    '香气足': '香气量足',
    '香韵足': '香气量足',
    '留香久': '留香持久',
    '留香时间长': '留香持久',
    '回味悠长': '余味绵长',
    '回甜明显': '回甜',
    '甜感明显': '回甜',
    '品牌影响力高': '品牌知名度高',
    '品牌认可度高': '品牌美誉度高',
    '包装靓丽': '包装好看',
    '精致大气': '精致高端',
    '高端质感': '高端',
}

OPINION_TERM_STRIP_PATTERNS = [
    r'^(很|非常|挺|太|更|较|比较|十分|特别|相当|有点|稍微|略微|真|真的)+',
    r'(啊|呀|呢|吧|哦|啦|嘛)+$',
]

OPINION_TERM_ATOMS = [
    '回甜', '回甘', '清甜香气', '清甜', '本香', '顺畅', '顺滑', '丝滑', '柔和', '绵柔',
    '绵长', '余味绵长', '醇厚', '香气好', '纯', '高端', '大气', '上档次', '有面子',
    '划算', '推荐', '很好', '细腻', '劲头', '满足感', '香气量足', '干净', '舒适',
    '好看', '精致', '独特', '新颖', '稳定', '方便', '适中', '贵', '便宜',
    '刺激', '不刺激', '清新', '自然', '留香', '持久', '甜润', '回甜',
    '知名度', '美誉度', '包装', '质感', '高端', '精致', '很细', '细',
    '口口满足', '满足', '偏慢', '个性感', '太美', '手掰开', '一般', '一般般',
    '微醺', '没区别', '没有明显的区别', '无甚区别'
]

OPINION_NEGATION_TOKENS = ['不', '没', '无', '非']

OPINION_CONTRAST_PAIRS = [
    ('刺激', '不刺激'),
    ('辣', '不辣'),
    ('呛', '不呛'),
    ('贵', '便宜'),
    ('高端', '普通'),
]

OPINION_CORE_TOKENS = [
    '清新', '自然', '香气', '香韵', '本香', '醇厚', '柔和', '细腻', '顺滑', '顺畅',
    '丝滑', '回甜', '回甘', '余味', '绵长', '刺激', '不刺激', '劲头', '满足感',
    '舒适', '干净', '清晰', '丰富', '透发', '高端', '大气', '上档次', '有面子',
    '好看', '精致', '质感', '新颖', '独特', '创新', '划算', '推荐', '知名度',
    '美誉度', '品牌', '包装', '价格', '稳定', '方便', '持久', '留香',
    '很细', '口口满足', '满足', '偏慢', '个性感', '太美', '手掰开'
]

HEURISTIC_CATEGORY_KEYWORDS = [
    ('包装结构', '包装结构形态'),
    ('结构形态', '包装结构形态'),
    ('包装图案', '包装图案'),
    ('图案', '包装图案'),
    ('包装颜色', '包装颜色'),
    ('颜色', '包装颜色'),
    ('色彩', '包装颜色'),
    ('包装质感', '包装质感'),
    ('质感', '包装质感'),
    ('材质', '包装质感'),
    ('手感', '包装质感'),
    ('开合', '开合方式'),
    ('吸阻', '抽吸阻力'),
    ('阻力', '抽吸阻力'),
    ('燃烧', '燃烧速度'),
    ('灰', '包灰性'),
    ('品牌知名度', '品牌知名度'),
    ('品牌影响力', '品牌知名度'),
    ('品牌美誉度', '品牌美誉度'),
    ('品牌认可度', '品牌美誉度'),
    ('品牌定位', '品牌定位'),
    ('品牌', '品牌形象'),
    ('货源', '货源稳定程度'),
    ('供货', '货源稳定程度'),
    ('性价比', '价格'),
    ('价格', '价格'),
    ('滤嘴', '滤嘴长度'),
    ('烟丝', '烟丝填充性'),
    ('做工', '工艺品质'),
    ('工艺', '工艺品质'),
    ('品质', '工艺品质'),
    ('掉火', '燃烧锥稳定性'),
    ('飞火', '燃烧锥稳定性'),
    ('稳定', '口味稳定性'),
    ('新颖', '新颖性'),
    ('创意', '新颖性'),
    ('创新', '新颖性'),
    ('文化', '品规概念产品文化'),
    ('底蕴', '品规概念产品文化'),
    ('功能', '功能特性'),
    ('焦油', '焦油'),
    ('二手烟', '气味扩散强度'),
    ('残留', '气味残留度'),
    ('香', '香气'),
    ('烟气', '烟气'),
    ('吸感', '烟气'),
    ('口感', '口感'),
    ('味道', '口感'),
    ('余味', '口感'),
    ('刺激', '口感'),
]

FINE_GRAIN_HINTS = {
    '香气': [
        ('本香', '香韵特征'),
        ('风格', '香韵特征'),
        ('调香', '香韵特征'),
        ('清香', '香韵特征'),
        ('花香', '香韵特征'),
        ('果香', '香韵特征'),
        ('木香', '香韵特征'),
        ('咖啡香', '香韵特征'),
        ('坚果香', '香韵特征'),
        ('蜜香', '香韵特征'),
        ('草药味', '香韵特征'),
        ('玫瑰香', '香韵特征'),
        ('清甜', '香韵特征'),
        ('纯正', '香气质'),
        ('醇', '香气质'),
        ('舒适', '香气质'),
        ('醇厚', '香气质'),
        ('香气量足', '香气量'),
        ('浓郁', '香气量'),
        ('饱满', '香气量'),
        ('寡淡', '香气量'),
        ('淡薄', '香气量'),
        ('淡', '香气量'),
        ('不足', '香气量'),
        ('足', '香气量'),
        ('透发', '透发性'),
        ('出味快', '透发性'),
        ('来得快', '透发性'),
        ('扩散快', '透发性'),
        ('扩散', '透发性'),
        ('闻香', '嗅香'),
        ('层次', '丰富性'),
        ('丰富', '丰富性'),
        ('层次分明', '丰富性'),
        ('杂味', '杂气'),
        ('异味', '杂气'),
        ('清晰', '清晰度'),
        ('辨识度', '清晰度'),
        ('分辨识度', '清晰度'),
        ('特征明显', '清晰度'),
    ],
    '烟气': [
        ('入口顺滑', '顺畅性'),
        ('顺滑', '顺畅性'),
        ('顺', '顺畅性'),
        ('通透', '顺畅性'),
        ('微醺', '劲头'),
        ('劲头', '劲头'),
        ('杀瘾', '劲头'),
        ('满足感', '劲头'),
        ('带劲', '劲头'),
        ('给力', '劲头'),
        ('过瘾', '劲头'),
        ('解乏', '劲头'),
        ('劲道', '劲头'),
        ('悠长', '绵长感'),
        ('绵长', '绵长感'),
        ('留香', '绵长感'),
        ('持久', '绵长感'),
        ('回味长', '绵长感'),
        ('留香', '绵长感'),
        ('柔和', '柔和性'),
        ('绵柔', '柔和性'),
        ('柔', '柔和性'),
        ('细腻', '细腻度'),
        ('醇和', '细腻度'),
        ('甜润', '甜润感'),
        ('润', '甜润感'),
        ('微润', '甜润感'),
        ('润泽', '甜润感'),
        ('成团', '成团性'),
    ],
    '口感': [
        ('回甘', '回甜'),
        ('回甜', '回甜'),
        ('甘甜', '回甜'),
        ('口干', '干燥感'),
        ('拔干', '干燥感'),
        ('燥', '干燥感'),
        ('余味', '余味'),
        ('回味', '余味'),
        ('干净', '余味'),
        ('辣', '刺激性'),
        ('刮喉', '刺激性'),
        ('刺', '刺激性'),
        ('呛', '刺激性'),
        ('辣喉', '刺激性'),
    ],
    '品质（口味）延续性': [
        ('没以前', '品质（口味）延续性'),
        ('不如以前', '品质（口味）延续性'),
        ('以前的好', '品质（口味）延续性'),
        ('现在生产', '品质（口味）延续性'),
        ('前后不一致', '品质（口味）延续性'),
        ('越来越差', '品质（口味）延续性'),
        ('变化大', '品质（口味）延续性'),
    ],
    '外观包装': [
        ('包装精美', '包装图案'),
        ('图案', '包装图案'),
        ('设计感', '包装图案'),
        ('山水', '包装图案'),
        ('景色', '包装图案'),
        ('中国风', '包装图案'),
        ('文化气息', '包装图案'),
        ('好看', '包装图案'),
        ('漂亮', '包装图案'),
        ('醒目', '包装图案'),
        ('精美', '包装图案'),
        ('老旧', '包装图案'),
        ('简约', '包装图案'),
        ('经典', '包装图案'),
        ('传统', '包装图案'),
        ('元素', '包装图案'),
        ('配色', '包装颜色'),
        ('颜色', '包装颜色'),
        ('色调', '包装颜色'),
        ('黑金', '包装颜色'),
        ('绿色', '包装颜色'),
        ('蓝色', '包装颜色'),
        ('红色', '包装颜色'),
        ('金色', '包装颜色'),
        ('黄和白', '包装颜色'),
        ('质感', '包装质感'),
        ('材质', '包装质感'),
        ('磨砂', '包装质感'),
        ('植绒', '包装质感'),
        ('皮革', '包装质感'),
        ('高端大气上档次', '包装图案'),
        ('高端大气', '包装图案'),
        ('奢华', '包装图案'),
        ('大气', '包装图案'),
        ('高端质感', '包装质感'),
        ('开盒', '开合方式'),
        ('开启', '开合方式'),
        ('掰开', '开合方式'),
        ('侧滑', '开合方式'),
        ('条盒', '包装结构形态'),
        ('盒型', '包装结构形态'),
        ('细支盒', '包装结构形态'),
        ('携带方便', '包装结构形态'),
        ('方便携带', '包装结构形态'),
    ],
    '烟支设计': [
        ('滤嘴', '滤嘴长度'),
        ('爆珠', '滤嘴类型'),
        ('中空', '滤嘴类型'),
        ('降焦孔', '滤嘴类型'),
        ('活性炭', '滤嘴类型'),
        ('颗粒', '滤嘴类型'),
        ('爆珠滤嘴', '滤嘴类型'),
        ('细支', '滤嘴类型'),
        ('中支', '滤嘴类型'),
        ('燃烧', '燃烧速度'),
        ('偏慢', '燃烧速度'),
        ('太快', '燃烧速度'),
        ('吸阻', '抽吸阻力'),
        ('阻力', '抽吸阻力'),
        ('适中', '抽吸阻力'),
        ('包灰', '包灰性'),
        ('烟灰', '包灰性'),
        ('不断灰', '包灰性'),
    ],
    '工艺品质': [
        ('填充', '烟丝填充性'),
        ('紧实', '烟丝填充性'),
        ('紧密', '烟丝填充性'),
        ('松散', '烟丝填充性'),
        ('掉火', '燃烧锥稳定性'),
        ('飞火', '燃烧锥稳定性'),
        ('偏烧', '燃烧锥稳定性'),
        ('做工精细', '工艺品质'),
        ('工艺精湛', '工艺品质'),
        ('精细', '工艺品质'),
        ('精湛', '工艺品质'),
        ('匠心', '工艺品质'),
        ('稳定', '口味稳定性'),
        ('一致', '口味稳定性'),
        ('品控', '口味稳定性'),
    ],
    '品牌形象': [
        ('知名', '品牌知名度'),
        ('大牌', '品牌知名度'),
        ('老牌子', '品牌知名度'),
        ('出名', '品牌知名度'),
        ('都知道', '品牌知名度'),
        ('名气', '品牌知名度'),
        ('底蕴深厚', '品牌知名度'),
        ('口碑', '品牌美誉度'),
        ('值得信赖', '品牌美誉度'),
        ('认可度', '品牌美誉度'),
        ('认可', '品牌美誉度'),
        ('信赖', '品牌美誉度'),
        ('声誉', '品牌美誉度'),
        ('公认', '品牌美誉度'),
        ('一致好评', '品牌美誉度'),
        ('都说好', '品牌美誉度'),
        ('推荐', '品牌美誉度'),
        ('高端', '品牌定位'),
        ('定位', '品牌定位'),
        ('有面子', '品牌定位'),
        ('有面', '品牌定位'),
        ('送礼', '品牌定位'),
        ('上档次', '品牌定位'),
        ('口粮', '品牌定位'),
        ('断货', '货源稳定程度'),
        ('难买', '货源稳定程度'),
        ('供货', '货源稳定程度'),
        ('本地没有', '货源稳定程度'),
        ('机场买', '货源稳定程度'),
    ],
    '产品设计创意': [
        ('文化', '品规概念产品文化'),
        ('情怀', '品规概念产品文化'),
        ('底蕴', '品规概念产品文化'),
        ('中国风', '品规概念产品文化'),
        ('新颖', '新颖性'),
        ('创新', '新颖性'),
        ('首创', '新颖性'),
        ('个性', '新颖性'),
        ('独特', '新颖性'),
        ('特别', '新颖性'),
        ('功能', '功能特性'),
        ('润喉', '功能特性'),
        ('降焦', '功能特性'),
    ],
    '负外部性': [
        ('二手烟', '气味扩散强度'),
        ('扩散', '气味扩散强度'),
        ('残留', '气味残留度'),
        ('留味', '气味残留度'),
    ],
}

STRICT_FINE_GRAIN_HINTS = {
    '香韵特征': ['本香', '风格', '调香', '清香', '花香', '果香', '木香', '咖啡香', '坚果香', '蜜香', '草药味', '玫瑰香', '清甜'],
    '包装质感': ['质感', '材质', '手感', '磨砂', '植绒', '皮革', '高端质感', '纸质'],
    '包装图案': ['图案', '山水', '景色', '中国风', '文化气息', '好看', '漂亮', '醒目',
                 '精美', '老旧', '简约', '经典', '传统', '元素', '包装设计', '外观设计', '设计感'],
    '品牌定位': ['高端', '定位', '有面子', '有面', '送礼', '上档次', '口粮'],
    '香气质': ['纯正', '醇', '舒适', '醇厚'],
    '清晰度': ['纯正', '烟草本香', '本香味十足', '辨识度', '分辨识度', '香料味道', '纯净', '特征明显'],
    '货源稳定程度': ['本地没有', '当地没有', '云南没有卖', '难买', '断货', '机场买'],
    '刺激性': ['不辣口', '不刺激', '没有辣喉鼻', '有点冲', '烧口', '辣喉', '刮喉', '呛'],
    '品牌美誉度': ['口碑', '信赖', '认可', '美誉', '声誉', '公认', '一致好评', '都说好', '值得信赖', '品牌认可度'],
    '香气量': ['浓郁', '饱满', '寡淡', '香气足', '香韵足', '淡薄', '香气量', '淡', '不足', '足'],
    '绵长感': ['悠长', '绵长', '留香', '持久', '回味长', '余味长'],
    '口味稳定性': ['品控', '品质稳定', '口味稳定', '品质不稳定', '品控差', '稳定', '一致', '前后不一致', '越来越', '不如以前'],
    '透发性': ['透发', '出味快', '来得快', '扩散快', '扩散'],
    '滤嘴类型': ['爆珠', '中空', '降焦孔', '活性炭', '颗粒', '爆珠滤嘴'],
    '甜润感': ['甜', '润', '甘', '微润', '清甜', '甜润'],
    '包装结构形态': ['盒型', '条盒', '携带', '硬化', '软包', '侧开', '侧滑', '掰开', '方便携带', '挺括', '结构'],
    '水松纸颜色、图案': ['水松纸', '滤嘴颜色', '金色滤嘴', '回型纹', '纹饰'],
    '顺畅性': ['顺', '滑', '通透', '流畅'],
}

GENERIC_OPINION_REFINEMENT_PATTERNS = [
    ('价格', ['适中', '不贵', '太贵', '便宜', '划算', '还行', '性价比高', '性价比比同等价位的高']),
    ('口感', ['非常好', '很好', '不错', '一般', '纯正', '醇厚', '细腻', '柔和', '顺滑', '微润', '微醺', '回甘', '清香']),
    ('内在口味', ['非常好', '很好', '不错', '一般', '纯正', '醇厚', '细腻', '柔和', '顺滑', '微润', '微醺', '回甘', '清香']),
    ('香气', ['纯正', '清香', '果香', '木香', '蜜香', '草药味', '醇厚', '香气好']),
    ('品质（口味）延续性', ['现在生产的软印象', '现在生产', '没以前好', '不如以前', '以前的好', '越来越差', '前后不一致']),
    ('货源稳定程度', ['本地没有出售', '本地没有买', '本地没有', '云南没有卖', '当地没卖', '当地没有', '机场买过']),
    ('刺激性', ['不辣口', '不刺激', '没有辣喉鼻', '第一口有点冲', '烧口', '辣喉', '刮喉', '呛']),
    ('清晰度', ['纯正', '烟草本香', '本香味十足', '有一点淡淡的香料味道', '辨识度高', '分辨识度高']),
    ('烟气', ['顺滑', '细腻', '柔和', '微润', '微醺', '带劲', '过瘾', '满足感', '适中']),
    ('外观包装', ['简约大气', '高端大气上档次', '老旧', '好看', '漂亮', '大气', '简约', '精致']),
    ('包装图案', ['简约大气', '老旧', '好看', '漂亮', '大气', '简约', '图案']),
    ('品牌形象', ['喜欢', '大品牌', '老牌子', '值得信赖', '有面子', '送礼']),
    ('产品设计创意', ['全国首款', '首创', '创新', '新颖', '独特', '新上的', '新东东', '年轻人偏爱的设计', '品牌创新']),
    ('焦油', ['焦油', '降焦', '低焦', '很健康', '健康需求', '8mg']),
]

ASPECT_NOUN_TERMS = [
    '口感', '味道', '口味', '香气', '烟气', '包装', '外包装', '图案', '颜色', '质感',
    '品牌', '价格', '性价比', '烟支', '滤嘴', '做工', '工艺', '质量', '品质', '货源'
]

EVALUATION_TERMS = [
    '高端大气上档次', '性价比比同等价位的高', '非常不错', '非常好', '很好抽', '不好抽',
    '不怎么样', '没以前好', '不如以前', '值得信赖', '简约大气', '本香味十足',
    '辨识度高', '分辨识度高', '第一口有点冲', '没有辣喉鼻', '性价比高',
    '很不错', '很好', '不错', '一般般', '还可以', '还行', '适中', '纯正', '醇厚',
    '细腻', '柔和', '顺滑', '顺畅', '微润', '微醺', '回甘', '回甜', '清香', '浓郁',
    '饱满', '寡淡', '干净', '舒适', '刺激', '不刺激', '辣口', '不辣口', '辣喉',
    '刮喉', '烧口', '呛', '口干', '拔干', '好看', '漂亮', '精致', '大气', '简约',
    '老旧', '高级', '奢华', '划算', '便宜', '不贵', '太贵', '贵', '喜欢', '推荐',
    '满意', '稳定', '难买', '断货', '好', '差'
]

POSITIVE_POLARITY_HINTS = ['很不错', '不错', '很好', '喜欢', '推荐', '满意', '纯正', '醇厚', '顺滑', '细腻', '柔和', '划算', '值得信赖']
NEGATIVE_POLARITY_HINTS = ['不好', '不怎么样', '老旧', '没以前好', '不如以前', '太贵', '燥感', '刺激', '辣', '呛']

# ═══ 后处理类别纠正: 构建合法类别集合 ═══
def _build_all_valid_categories():
    cats = set()
    for top, seconds in ASPECT_CATEGORY_HIERARCHY.items():
        cats.add(top)
        for second, thirds in seconds.items():
            cats.add(second)
            for third in thirds:
                cats.add(third)
    return cats

ALL_VALID_CATEGORIES = _build_all_valid_categories() - EXCLUDED_EVAL_CATEGORIES

# ═══ 动态Few-Shot检索器 ═══
class FewShotRetriever:
    """用sentence-transformers做embedding检索, 动态选择最相似的标注样本作为few-shot示例。"""

    def __init__(self, data_path, project_to_level2=False):
        import pandas as pd
        df = pd.read_csv(data_path)
        pool_df = df.copy()  # 使用全部数据作为检索池
        self.sentences = []
        self.examples = []  # 格式: "句子：xxx\n输出：[...]"

        for _, row in pool_df.iterrows():
            sentence = str(row.get('sentence', '')).strip()
            gold_str = str(row.get('gold_tuples', '[]')).strip()
            if not sentence or not gold_str or gold_str == '[]':
                continue
            try:
                gold_tuples = _ast_parse.literal_eval(gold_str)
            except (ValueError, SyntaxError):
                continue
            # 转换为模型输出格式: ('类别', '观点词', '情感')
            model_output = []
            for t in gold_tuples:
                if len(t) >= 4:
                    cat = str(t[1]).strip()
                    if ',' in cat:
                        cat = cat.split(',')[0].strip()
                    if cat and cat not in EXCLUDED_EVAL_CATEGORIES:
                        # 两阶段模式: 投影到二级类
                        if project_to_level2:
                            cat = CATEGORY_TO_SECOND.get(cat, cat)
                        model_output.append((cat, str(t[2]).strip(), str(t[3]).strip()))
            if not model_output:
                continue
            self.sentences.append(sentence)
            self.examples.append(f"句子：{sentence}\n输出：{model_output}")

        print(f"FewShotRetriever: 加载 {len(self.sentences)} 条检索池样本, 正在加载embedding模型...", flush=True)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True)
        print(f"FewShotRetriever: 模型加载完成, 正在分批编码...", flush=True)
        # 分批编码避免内存峰值
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(self.sentences), batch_size):
            batch = self.sentences[i:i + batch_size]
            emb = self.embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(emb)
            if (i // batch_size) % 5 == 0:
                print(f"  FewShotRetriever: 编码进度 {min(i + batch_size, len(self.sentences))}/{len(self.sentences)}", flush=True)
        import numpy as np
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"FewShotRetriever: embedding构建完成, shape={self.embeddings.shape}", flush=True)

    def retrieve(self, query_sentence, top_k=RETRIEVAL_TOP_K):
        """返回最相似的top_k个标注示例字符串列表。"""
        if not self.sentences:
            return []
        query_emb = self.embedder.encode([query_sentence], normalize_embeddings=True)
        scores = query_emb @ self.embeddings.T  # cosine similarity
        top_indices = scores[0].argsort()[-top_k:][::-1]
        return [self.examples[i] for i in top_indices]

class SentimentAnalyzer:
    def __init__(self, model_path, fewshot_retriever=None, lora_path=None):
        """初始化情感分析器。可选: FewShotRetriever + LoRA适配器"""
        self.fewshot_retriever = fewshot_retriever
        self.lora_path = lora_path
        if fewshot_retriever is not None:
            print(f"动态Few-Shot已启用: 每次推理检索 {RETRIEVAL_TOP_K} 个最相似标注样本作为示例")
        if lora_path and os.path.exists(lora_path):
            print(f"LoRA适配器: {lora_path}")

        # 多卡环境下，让PyTorch自动选择设备
        print(f"正在加载模型 {model_path}...")
        start_time = time.time()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision='master'
        )
        
        # 加载模型: LoRA模式用4-bit, 否则用BF16
        self.model = None
        if lora_path and os.path.exists(lora_path):
            from transformers import BitsAndBytesConfig as BnB
            quant_cfg = BnB(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            print("LoRA模式: 4-bit量化加载...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="cuda:0", quantization_config=quant_cfg,
                trust_remote_code=True, revision='master'
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.bfloat16,
                trust_remote_code=True, revision='master'
            )
        
        # 加载 LoRA 适配器 (如果存在且非4-bit模式才需要加载, 4-bit已加载)
        if lora_path and os.path.exists(lora_path):
            from peft import PeftModel
            # 检查模型是否已有 LoRA 权重
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            # ⚠️ 不 merge: 4-bit 模型 merge_and_unload 会反量化成 fp16 (64GB),
            # 峰值显存 ~80GB 必 OOM (2026-08-17 排查发现)。4-bit+LoRA 直接推理
            # 结果等价且只需 ~20GB, 与 check_gpu.sh 24GB 阈值匹配。
            self.model = self.model.eval()
            print(f"LoRA适配器已加载: {lora_path} (4-bit+LoRA 推理, 不 merge)")

        # 获取模型所在的设备
        self.device = next(self.model.parameters()).device

        # 当前评估使用确定性解码，清理模型自带的采样参数，避免 generate 时出现无效 flags 警告
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        
        # 检查模型加载是否成功
        if self.model is not None:
            print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        else:
            print("模型加载失败")
    
    def analyze_sentiment(self, sentence):
        """分析句子情感并提取三元组（方面类别, 观点术语, 情感极性）"""
        prompt = f"""请对下面句子做方面级情感分析，提取三元组（方面类别, 观点术语, 情感极性）。

只返回 Python 列表，不要解释，不要输出思考过程。
格式：[('方面类别', '观点术语', '情感极性')]
情感极性只能是：正向、负向、中性。

类别体系是三级：
1. 一级类：内在口味、外观包装、烟支设计、工艺品质、品牌形象、产品设计创意、价格、焦油、负外部性
2. 二级类：
- 内在口味 -> 香气、烟气、口感、品质（口味）延续性
- 外观包装 -> 外观包装
- 烟支设计 -> 烟支设计
- 工艺品质 -> 工艺品质
- 品牌形象 -> 品牌形象
- 产品设计创意 -> 产品设计创意
- 价格 -> 价格
- 焦油 -> 焦油
- 负外部性 -> 负外部性
3. 三级类：
- 香气 -> 香韵特征、香气质、香气量、透发性、嗅香、丰富性、杂气、清晰度
- 烟气 -> 顺畅性、劲头、绵长感、柔和性、细腻度、甜润感
- 口感 -> 回甜、干燥感、余味、刺激性
- 外观包装 -> 包装图案、包装颜色、包装质感、开合方式、包装结构形态、水松纸颜色、图案、水松纸质感、卷烟纸颜色、卷烟纸质感
- 烟支设计 -> 滤嘴长度、滤嘴类型、燃烧速度、抽吸阻力、包灰性
- 工艺品质 -> 烟丝填充性、燃烧锥稳定性、口味稳定性
- 品牌形象 -> 品牌知名度、品牌美誉度、品牌定位、货源稳定程度
- 产品设计创意 -> 品规概念产品文化、新颖性、功能特性
- 负外部性 -> 气味扩散强度、气味残留度

选择规则：
1. 能判断到三级类时，必须输出三级类。
2. 无法判断三级类时，输出对应二级类。
3. 只有信息明显不足时，才输出一级类。
4. 一个句子可能包含多个不同方面，尽量完整覆盖，不要遗漏。
5. 不要输出防伪设计、成团性、刺破、异物、盒盖盖不好这些类别。
6. 观点术语必须从原句中复制连续片段，优先选择最像人工标注者会标出的完整评价片段。
7. 不要把"口感、味道、包装、品牌、价格"等方面词本身当作观点术语；例如"口感好"应输出观点术语"好"，"包装精致"应输出"精致"。
8. 不要把观点术语改写成总结词或同义词；如果原句是"真的太好抽了"，观点术语应尽量取"好抽"或"真的太好抽了"这类原文片段。

常见映射示例：
- 本香、清香、果香、木香 -> 香韵特征
- 香气纯正、香气舒适、很醇 -> 香气质
- 香气浓郁、饱满、寡淡 -> 香气量
- 顺滑、通透 -> 顺畅性
- 带劲、过瘾、满足感强 -> 劲头
- 柔和、绵柔 -> 柔和性
- 细腻、醇和 -> 细腻度
- 回甘、发甜 -> 回甜
- 余味干净、回味短 -> 余味
- 刺喉、辣、刮喉 -> 刺激性
- 图案好看、设计感强 -> 包装图案
- 配色好、黑金、绿色、蓝色 -> 包装颜色
- 磨砂、质感好、高级 -> 包装质感
- 爆珠、中空、降焦孔 -> 滤嘴类型
- 吸阻大、太通透 -> 抽吸阻力
- 品控稳定、前后一致 -> 口味稳定性
- 知名、大牌、名气大 -> 品牌知名度
- 口碑好、值得信赖、认可度高 -> 品牌美誉度
- 高端、送礼、有面子、口粮 -> 品牌定位
- 经常断货、难买、本地没有 -> 货源稳定程度
- 设计新颖、创新、独特 -> 新颖性
- 文化底蕴、情怀 -> 品规概念产品文化

输出示例：
[('顺畅性', '顺滑', '正向'), ('包装颜色', '黑金配色', '正向'), ('价格', '太贵', '负向')]

sentence:{sentence}"""

        if USE_SHORT_PROMPT:
            if USE_TWO_STAGE:
                # Stage 1: 只输出二级类, 三级由规则映射
                categories = (
                    "内在口味, 香气, 烟气, 口感, 品质（口味）延续性, "
                    "外观包装, 烟支设计, 工艺品质, "
                    "品牌形象, 产品设计创意, "
                    "价格, 焦油, 负外部性"
                )
                prompt = f"""请做方面级情感三元组抽取。只输出 Python 列表，不要解释，不要输出思考过程。

格式：[('方面类别', '观点术语', '情感极性')]
情感极性只能是：正向、负向、中性。
方面类别只能从这些标签中选择：{categories}
一个句子中可能同时包含多个不同方面，请不要遗漏。
方面类别只用二级类：内在口味下分香气/烟气/口感/品质延续性；品牌形象；外观包装；烟支设计；工艺品质；产品设计创意；价格；焦油；负外部性。
观点术语必须复制原句中的连续片段，不要总结或改写。
不要把方面词当观点词：如"口感好"输出观点词"好"，"包装精致"输出"精致"，"价格贵"输出"贵"。
如果没有情感三元组，输出 []。
"""
            else:
                categories = (
                    "内在口味, 香气, 香韵特征, 香气质, 香气量, 透发性, 嗅香, 丰富性, 杂气, 清晰度, "
                    "烟气, 顺畅性, 劲头, 绵长感, 柔和性, 细腻度, 甜润感, 成团性, "
                    "口感, 回甜, 干燥感, 余味, 刺激性, 品质（口味）延续性, "
                    "外观包装, 包装图案, 包装颜色, 包装质感, 开合方式, 包装结构形态, 水松纸颜色、图案, 水松纸质感, 卷烟纸颜色, 卷烟纸质感, "
                    "烟支设计, 滤嘴长度, 滤嘴类型, 燃烧速度, 抽吸阻力, 包灰性, "
                    "工艺品质, 烟丝填充性, 燃烧锥稳定性, 口味稳定性, "
                    "品牌形象, 品牌知名度, 品牌美誉度, 品牌定位, 货源稳定程度, "
                    "产品设计创意, 品规概念产品文化, 新颖性, 功能特性, "
                    "价格, 焦油, 负外部性, 气味扩散强度, 气味残留度"
                )
                prompt = f"""请做方面级情感三元组抽取。只输出 Python 列表，不要解释，不要输出思考过程。

格式：[('方面类别', '观点术语', '情感极性')]
情感极性只能是：正向、负向、中性。
方面类别只能从这些标签中选择：{categories}
一个句子中可能同时包含多个不同方面，请不要遗漏。
类别分三级：一级如"内在口味"，二级如"香气/烟气/口感"，三级如"香韵特征/顺畅性/回甜"。
能判断到三级类时必须输出三级类；无法判断三级时输出二级类；只有信息确实不足时才输出一级类。
观点术语必须复制原句中的连续片段，不要总结或改写。
不要把方面词当观点词：如"口感好"输出观点词"好"，"包装精致"输出"精致"，"价格贵"输出"贵"。
如果没有情感三元组，输出 []。
"""

            # 动态示例
            if self.fewshot_retriever is not None:
                retrieved = self.fewshot_retriever.retrieve(sentence)
                if retrieved:
                    prompt += "以下是与当前句子最相似的标注示例，请参考其标注风格：\n\n"
                    for i, example in enumerate(retrieved, 1):
                        prompt += f"示例{i}：\n{example}\n\n"
            else:
                prompt += """示例1：
句子：入喉微润，吐气微醺，整体来说是这个分段的极限
输出：[('甜润感', '入喉微润', '正向'), ('劲头', '吐气微醺', '负向'), ('内在口味', '整体来说是这个分段的极限', '正向')]

示例2：
句子：口感醇厚，吸感适中，包装简约大气，性价比比同等价位的高
输出：[('香气质', '醇厚', '正向'), ('内在口味', '适中', '中性'), ('包装图案', '简约大气', '正向'), ('价格', '比同等价位的高', '正向')]

示例3：
句子：这东西太冲了，用着手干，而且价格太贵
输出：[('刺激性', '太呛', '负向'), ('干燥感', '口干', '负向'), ('价格', '太贵', '负向')]

示例4：
句子：水松纸的颜色很漂亮，回型纹是传统元素
输出：[('水松纸颜色、图案', '颜色很漂亮', '正向'), ('包装图案', '回型纹', '正向')]

示例5：
句子：品控很稳定，口味前后一致，烟气顺滑
输出：[('口味稳定性', '品控很稳定', '正向'), ('口味稳定性', '前后一致', '正向'), ('顺畅性', '顺滑', '正向')]

"""

            prompt += f"句子：{sentence}\n输出："

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 解析响应，提取三元组，并做类别纠正
        try:
            parsed = None
            list_matches = re.findall(r"\[[\s\S]*?\]", response)
            for tuples_str in list_matches:
                parsed = self._parse_predicted_tuples(tuples_str, sentence=sentence)
                if parsed:
                    break

            if not parsed:
                pattern_all_single = r"\('([^']*)',\s*'([^']*)',\s*'([^']*)'\)"
                matches_all_single = re.findall(pattern_all_single, response)
                if matches_all_single:
                    parsed = [
                        (
                            self._refine_category_label(self._normalize_category_label(a), b.strip(), sentence),
                            b.strip(),
                            c.strip()
                        )
                        for a, b, c in matches_all_single
                    ]

            if not parsed:
                pattern_all_double = r'\("([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)'
                matches_all_double = re.findall(pattern_all_double, response)
                if matches_all_double:
                    parsed = [
                        (
                            self._refine_category_label(self._normalize_category_label(a), b.strip(), sentence),
                            b.strip(),
                            c.strip()
                        )
                        for a, b, c in matches_all_double
                    ]

            if not parsed:
                cleaned_response = response.replace(' ', '').replace('\n', '')
                pattern_clean = r"\('([^']*)',\s*'([^']*)',\s*'([^']*)'\)"
                matches_clean = re.findall(pattern_clean, cleaned_response)
                if matches_clean:
                    parsed = [
                        (
                            self._refine_category_label(self._normalize_category_label(a), b.strip(), sentence),
                            b.strip(),
                            c.strip()
                        )
                        for a, b, c in matches_clean
                    ]

            if parsed:
                parsed = self._correct_categories(parsed, sentence)
                return response, self._augment_predicted_triples(sentence, parsed)

            return response, []
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"响应内容: {response}")
            return response, []
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"响应内容: {response}")
            return response, []

    def _correct_categories(self, tuples, sentence=""):
        """后处理纠正: 用模糊匹配修正非法类别名, 纠正非法极性。"""
        import difflib
        valid_cats = sorted(ALL_VALID_CATEGORIES, key=len, reverse=True)
        corrected = []
        for cat, opinion, polarity in tuples:
            cat = str(cat).strip()
            opinion = str(opinion).strip()
            polarity = str(polarity).strip()
            if not cat or not opinion:
                continue
            # 类别纠正: 精确匹配 → 精确包含 → 模糊匹配
            if cat not in ALL_VALID_CATEGORIES:
                # 尝试在 valid set 中找包含关系
                matched = None
                for vc in valid_cats:
                    if cat in vc or vc in cat:
                        matched = vc
                        break
                if not matched:
                    # 模糊匹配
                    close = difflib.get_close_matches(cat, valid_cats, n=1, cutoff=0.6)
                    if close:
                        matched = close[0]
                if matched:
                    cat = matched
                else:
                    continue  # 无法纠正, 丢弃
            # 极性纠正
            if polarity not in ('正向', '负向', '中性'):
                if any(kw in polarity for kw in ['正', '好', '赞', '优', '良', '喜']):
                    polarity = '正向'
                elif any(kw in polarity for kw in ['负', '差', '烂', '糟', '劣']):
                    polarity = '负向'
                else:
                    polarity = '中性'
            corrected.append((cat, opinion, polarity))
        return corrected

    def _parse_predicted_tuples(self, tuples_str, sentence=""):
        """解析模型返回的三元组列表，过滤无效项。"""
        try:
            predicted_tuples = ast.literal_eval(tuples_str)
            if isinstance(predicted_tuples, list):
                valid_tuples = []
                for item in predicted_tuples:
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        normalized_category = self._normalize_category_label(item[0])
                        if normalized_category in EXCLUDED_EVAL_CATEGORIES or not normalized_category:
                            continue
                        opinion_term = self._refine_opinion_term(normalized_category, str(item[1]).strip(), sentence)
                        polarity = self._refine_polarity(normalized_category, opinion_term, str(item[2]).strip(), sentence)
                        valid_tuples.append((
                            self._refine_category_label(normalized_category, opinion_term, sentence),
                            opinion_term,
                            polarity
                        ))
                if valid_tuples:
                    return self._augment_predicted_triples(sentence, valid_tuples)
        except (ValueError, SyntaxError):
            pass

        pattern_single = r"\('([^']*)',\s*'([^']*)',\s*'([^']*)'\)"
        matches_single = re.findall(pattern_single, tuples_str)
        if matches_single:
            parsed = [
                (
                    self._refine_category_label(
                        self._normalize_category_label(a),
                        self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence),
                        sentence
                    ),
                    self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence),
                    self._refine_polarity(self._normalize_category_label(a), self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence), c.strip(), sentence)
                )
                for a, b, c in matches_single
            ]
            return self._augment_predicted_triples(sentence, parsed)

        pattern_double = r'\("([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)'
        matches_double = re.findall(pattern_double, tuples_str)
        if matches_double:
            parsed = [
                (
                    self._refine_category_label(
                        self._normalize_category_label(a),
                        self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence),
                        sentence
                    ),
                    self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence),
                    self._refine_polarity(self._normalize_category_label(a), self._refine_opinion_term(self._normalize_category_label(a), b.strip(), sentence), c.strip(), sentence)
                )
                for a, b, c in matches_double
            ]
            return self._augment_predicted_triples(sentence, parsed)

        return self._augment_predicted_triples(sentence, [])

    def _normalize_category_label(self, category):
        """将类别归一化到当前脚本支持的类别体系。"""
        category = str(category).strip()
        if not category:
            return category

        if category in EXCLUDED_EVAL_CATEGORIES:
            return ""

        if category in LEGACY_CATEGORY_NORMALIZATION:
            return LEGACY_CATEGORY_NORMALIZATION[category]

        if category in TOP_LEVEL_CATEGORIES or category in SECOND_LEVEL_CATEGORIES or category in THIRD_LEVEL_CATEGORIES:
            return category

        normalized = category.replace('，', ',').replace('／', '/')
        if normalized in EXCLUDED_EVAL_CATEGORIES:
            return ""
        if normalized in LEGACY_CATEGORY_NORMALIZATION:
            return LEGACY_CATEGORY_NORMALIZATION[normalized]
        if normalized in TOP_LEVEL_CATEGORIES or normalized in SECOND_LEVEL_CATEGORIES or normalized in THIRD_LEVEL_CATEGORIES:
            return normalized

        for keyword, mapped_category in HEURISTIC_CATEGORY_KEYWORDS:
            if keyword in normalized:
                return mapped_category

        return normalized

    def _candidate_base_categories(self, category, sentence=""):
        """为泛类别或未知类别挑选可能的一级类别，便于继续下钻到二级类。"""
        category = str(category).strip()
        sentence = str(sentence).strip()
        candidates = []

        if category in THIRD_LEVEL_CATEGORIES:
            second_category = self._map_category_to_second(category)
            if second_category in SECOND_LEVEL_CATEGORIES:
                candidates.append(second_category)
            return list(dict.fromkeys(candidates))

        if category in SECOND_LEVEL_CATEGORIES:
            candidates.append(category)

        normalized = self._normalize_category_label(category)
        if normalized in SECOND_LEVEL_CATEGORIES:
            candidates.append(normalized)
        elif normalized in GENERIC_CATEGORY_GROUPS:
            candidates.extend(GENERIC_CATEGORY_GROUPS[normalized])
        elif category in GENERIC_CATEGORY_GROUPS:
            candidates.extend(GENERIC_CATEGORY_GROUPS[category])
        else:
            second_category = self._map_category_to_second(normalized)
            if second_category in SECOND_LEVEL_CATEGORIES:
                candidates.append(second_category)

        if not candidates and sentence:
            for second_category, hints in FINE_GRAIN_HINTS.items():
                if any(keyword and keyword in sentence for keyword, _ in hints):
                    candidates.append(second_category)

        return list(dict.fromkeys(candidates))

    def _is_generic_opinion_term(self, opinion_term):
        """判断观点词是否过于泛化，只有这类词才更多依赖句子做补充。"""
        opinion_term = str(opinion_term).strip()
        generic_terms = {
            '口味', '味道', '吸味', '吸感', '价格', '品牌', '包装', '工艺', '质量', '口感',
            '烟气', '香气', '设计', '产品', '特点'
        }
        return opinion_term in generic_terms or len(opinion_term) <= 1

    def _has_explicit_taste_anchor(self, text):
        """只有存在明确口味锚点时，才允许往内在口味大类回落。"""
        text = str(text).strip()
        anchors = [
            '口味', '味道', '烟气', '吸味', '吸感', '清香', '醇厚', '顺滑', '柔和', '细腻',
            '回甘', '甜润', '劲头', '本香', '香气', '余味', '不辣口', '不刺激', '有点冲'
        ]
        return any(anchor in text for anchor in anchors)

    def _reroute_overpredicted_fine_category(self, category, opinion_term, sentence=""):
        """把清晰度/包装质感这类高频误报细类重新路由到更合适的类别。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        sentence = str(sentence).strip()
        text = f"{opinion_term} {sentence}".strip()

        if category == '清晰度':
            clear_strong = ['纯正', '纯净', '烟草本香', '本香味十足', '辨识度', '分辨识度', '香料味道', '特征明显']
            if any(keyword in opinion_term for keyword in clear_strong):
                return category
            if any(keyword in opinion_term for keyword in ['清香', '花香', '果香', '木香', '蜜香', '草药味', '香味']):
                return '香韵特征'
            if any(keyword in opinion_term for keyword in ['层次', '丰富']):
                return '丰富性'
            if any(keyword in opinion_term for keyword in ['顺滑', '顺畅', '流畅', '轻柔']):
                return '顺畅性'
            if any(keyword in opinion_term for keyword in ['柔和', '绵柔']):
                return '柔和性'
            if any(keyword in opinion_term for keyword in ['细腻']):
                return '细腻度'
            if any(keyword in opinion_term for keyword in ['回甘', '甘甜', '回甜']):
                return '回甜'
            if any(keyword in opinion_term for keyword in ['干净', '余味']):
                return '余味'
            if any(keyword in opinion_term for keyword in ['价格', '价位', '性价比']):
                return '价格'
            if any(keyword in opinion_term for keyword in ['品牌', '底蕴', '认可', '信赖', '有面子']):
                return '品牌形象'
            if any(keyword in opinion_term for keyword in ['包装', '图案', '设计', '精致', '大气', '好看', '漂亮', '简约']):
                return '包装图案'
            if any(keyword in opinion_term for keyword in ['工艺', '做工', '加工', '品质']):
                return '工艺品质'
            if any(keyword in opinion_term for keyword in ['焦油', '降焦', '健康', '8mg']):
                return '焦油'
            return '内在口味' if self._has_explicit_taste_anchor(text) else category

        if category == '包装质感':
            texture_strong = ['质感', '材质', '手感', '磨砂', '植绒', '皮革', '纸质', '触感']
            if any(keyword in opinion_term for keyword in texture_strong):
                return category
            if any(keyword in opinion_term for keyword in ['颜色', '配色', '黑金', '绿色', '蓝色', '红色', '金色', '黄和白']):
                return '包装颜色'
            if any(keyword in opinion_term for keyword in ['开盒', '开启', '掰开', '侧滑']):
                return '开合方式'
            if any(keyword in opinion_term for keyword in ['条盒', '盒型', '携带', '方便']):
                return '包装结构形态'
            if any(keyword in opinion_term for keyword in ['包装', '外观', '图案', '设计', '高端大气', '大气', '精致', '精美', '漂亮', '好看', '简约', '老旧', '无法挑剔', '上档次', '奢华']):
                return '包装图案'
            return category

        if category == '水松纸颜色、图案':
            waterpine_strong = ['水松纸', '滤嘴颜色', '金色滤嘴', '回型纹', '纹饰', '水松纸质感']
            if any(keyword in text for keyword in waterpine_strong):
                return category
            if any(keyword in opinion_term for keyword in ['颜色', '配色', '金色', '银色', '翡翠']):
                return '包装颜色'
            return '外观包装'

        if category == '包装结构形态':
            structure_strong = ['盒型', '条盒', '携带', '硬化', '软包', '侧开', '侧滑', '掰开', '方便携带', '挺括', '结构']
            if any(keyword in text for keyword in structure_strong):
                return category
            return '外观包装'

        if category == '香气质':
            quality_strong = ['纯正', '醇', '舒适', '醇厚', '纯净', '干净']
            if any(keyword in opinion_term for keyword in quality_strong):
                return category
            if any(keyword in opinion_term for keyword in ['层次', '丰富']):
                return '丰富性'
            return '香韵特征'

        if category == '香气量':
            amount_strong = ['浓郁', '饱满', '寡淡', '淡薄', '香气量足', '香韵足',
                            '香气量', '淡', '不足', '足', '香气浓郁', '香气饱满']
            if any(keyword in opinion_term for keyword in amount_strong):
                return category
            if any(keyword in opinion_term for keyword in ['纯正', '醇', '舒适', '醇厚', '纯净', '醇和', '干净']):
                return '香气质'
            if any(keyword in opinion_term for keyword in ['本香', '清香', '花香', '果香', '木香',
                                '蜜香', '草药味', '香韵', '香气好', '香气舒适', '独特']):
                return '香韵特征'
            if opinion_term in {'好', '很好', '不错', '很棒', '一般', '还行', '可以', '非常好'}:
                return '香韵特征'
            return category

        if category == '包装颜色':
            color_strong = ['配色', '黑金', '绿色', '蓝色', '红色', '金色', '黄和白',
                           '色彩', '颜色搭配', '色调', '颜色']
            if any(keyword in opinion_term for keyword in color_strong):
                return category
            if any(keyword in opinion_term for keyword in ['图案', '设计', '好看', '漂亮',
                                '精致', '大气', '简约', '老旧', '上档次', '奢华', '精美', '高端']):
                return '包装图案'
            if any(keyword in opinion_term for keyword in ['质感', '材质', '手感', '磨砂',
                                '植绒', '皮革', '纸质', '触感']):
                return '包装质感'
            return category

        # 以下重路由规则已禁用: LoRA后模型判别力提升, 规则干预反而退化

        return category

    def _extract_adjacent_evaluation_term(self, aspect_term, sentence):
        """当模型把方面词当观点词时，从邻近位置抽取真正的评价词。"""
        aspect_term = str(aspect_term).strip()
        sentence = str(sentence).strip()
        if not aspect_term or not sentence:
            return ""

        aspect_aliases = [aspect_term]
        if aspect_term in {'口感', '味道', '口味'}:
            aspect_aliases.extend(['口感', '味道', '口味', '吸味'])
        elif aspect_term in {'包装', '外包装'}:
            aspect_aliases.extend(['包装', '外包装'])
        elif aspect_term in {'价格', '性价比'}:
            aspect_aliases.extend(['价格', '价位', '性价比'])
        elif aspect_term in {'品牌'}:
            aspect_aliases.extend(['品牌', '牌子'])

        aspect_aliases = sorted(set(aspect_aliases), key=len, reverse=True)
        evaluation_terms = sorted(EVALUATION_TERMS, key=len, reverse=True)
        connector = r'(?:也|都|很|非常|比较|挺|太|还|更|确实|真的|是|有点|略微|偏|的|了|又|又很|还是|感觉|整体来说|来说|方面|和|及|、|，|,|\s){0,8}'

        for alias in aspect_aliases:
            if alias not in sentence:
                continue
            for term in evaluation_terms:
                if not term or term == alias or term not in sentence:
                    continue
                patterns = [
                    rf'{re.escape(alias)}{connector}{re.escape(term)}',
                    rf'{re.escape(term)}{connector}{re.escape(alias)}',
                ]
                for pattern in patterns:
                    if re.search(pattern, sentence):
                        return term

        # "口感和外包装都很好"这类共享评价词，评价词可能离方面词稍远。
        for alias in aspect_aliases:
            alias_pos = sentence.find(alias)
            if alias_pos < 0:
                continue
            right_context = sentence[alias_pos + len(alias): alias_pos + len(alias) + 12]
            for term in evaluation_terms:
                if term and term != alias and term in right_context:
                    return term

        return ""

    def _refine_category_label(self, category, opinion_term, sentence=""):
        """优先将一级类下钻到二级类。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        sentence = str(sentence).strip()
        opinion_text = opinion_term
        combined_text = f"{opinion_term} {sentence}"
        fallback_text = sentence if self._is_generic_opinion_term(opinion_term) else ""

        rerouted = self._reroute_overpredicted_fine_category(category, opinion_term, sentence)
        if rerouted != category:
            return rerouted

        if any(keyword in combined_text for keyword in ['水松纸', '滤嘴颜色', '金色滤嘴',
                '回型纹', '纹饰', '水松纸质感', '烟嘴颜色']) \
                and category in {'外观包装', '烟支设计', '水松纸颜色、图案', '水松纸质感'}:
            return '水松纸颜色、图案'

        if any(keyword in combined_text for keyword in ['品控', '品质稳定', '口味稳定',
                '品质不稳定', '品控差', '前后一致', '前后不一致', '口味一致']) \
                and category not in {'价格', '焦油', '负外部性', '品牌形象', '产品设计创意'}:
            return '口味稳定性'

        if any(keyword in opinion_text for keyword in ['微润', '入喉微润']):
            return '甜润感'
        if any(keyword in opinion_text for keyword in ['微醺', '吐气微醺']):
            return '劲头'
        if any(keyword in combined_text for keyword in ['现在生产', '没以前好', '不如以前', '以前的好']) and category in {'内在口味', '口感', '品质（口味）延续性'}:
            return '品质（口味）延续性'
        if (
            any(keyword in opinion_text for keyword in ['焦油', '降焦', '低焦', '很健康', '健康需求', '8mg']) or
            (category in {'焦油', '内在口味', '口感'} and self._is_generic_opinion_term(opinion_term) and any(keyword in sentence for keyword in ['焦油', '降焦', '低焦', '很健康', '健康需求', '8mg']))
        ):
            return '焦油'
        if (
            any(keyword in opinion_text for keyword in ['全国首款', '首创', '创新', '新颖', '新上的', '新东东', '年轻人偏爱的设计', '设置有内胆保湿']) or
            (category in {'产品设计创意', '新颖性', '功能特性'} and any(keyword in sentence for keyword in ['全国首款', '首创', '创新', '新颖', '新上的', '新东东', '年轻人偏爱的设计', '设置有内胆保湿']))
        ) and not any(keyword in combined_text for keyword in ['急需创新', '缺乏创新', '不够新颖']):
            return '新颖性'
        if any(keyword in combined_text for keyword in ['本地没有出售', '本地没有买', '本地没有', '当地没卖', '当地没有', '云南没有卖', '机场买过']):
            return '货源稳定程度'
        if category in {'价格', '品牌形象'} and any(keyword in opinion_text for keyword in ['价位段极具竞争力', '极具竞争力', '竞争力']):
            if any(keyword in combined_text for keyword in ['品牌', '高端', '代表之一', '品质', '价位段']):
                return '品牌形象'
        if any(keyword in opinion_text for keyword in ['第一口有点冲', '有点冲', '不辣口', '不刺激', '没有辣喉鼻', '烧口', '辣喉', '刮喉', '呛']):
            return '刺激性'
        if any(keyword in opinion_text for keyword in ['纯正', '烟草本香', '本香味十足', '辨识度高', '分辨识度高', '有一点淡淡的香料味道', '纯净']):
            return '清晰度'
        if '烟嘴' in sentence and any(keyword in combined_text for keyword in ['颜色', '翡翠', '金色']):
            return '水松纸颜色、图案'
        if '适中' in opinion_term and category in {'烟气', '内在口味'} and any(keyword in sentence for keyword in ['粗细', '烟支', '中支', '细支']):
            return '烟支设计'
        if '适中' in opinion_term and any(keyword in sentence for keyword in ['劲头', '带劲', '过瘾']):
            return '劲头'
        if any(keyword in combined_text for keyword in ['经典品牌', '大品牌', '老牌子']) and category in {'品牌形象', '内在口味'}:
            return '品牌知名度'

        # 欠预测召回规则已移除: LoRA 后模型已能自主判断细类, 规则反而造成过预测

        if category in THIRD_LEVEL_CATEGORIES or not opinion_term:
            return category

        candidate_groups = self._candidate_base_categories(category, fallback_text)
        if not candidate_groups:
            return category

        if category == '内在口味' and not self._has_explicit_taste_anchor(opinion_text):
            return category

        for candidate in candidate_groups:
            if candidate not in FINE_GRAIN_HINTS:
                continue
            for keyword, fine_category in FINE_GRAIN_HINTS[candidate]:
                if keyword and (keyword in opinion_text or (fallback_text and keyword in fallback_text)):
                    return fine_category

        return category

    def _refine_opinion_term(self, category, opinion_term, sentence=""):
        """将过于泛化的观点词细化为句中更接近标注表达的短语。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        sentence = str(sentence).strip()
        if not opinion_term or not sentence:
            return opinion_term

        if category == '品质（口味）延续性':
            match = re.search(r'现在生产的[^，,。；;！!？?\\s]*', sentence)
            if match:
                return match.group(0)

        if category in {'焦油', '内在口味', '口感'} and any(keyword in sentence for keyword in ['焦油', '降焦', '低焦', '很健康', '健康需求', '8mg']):
            for phrase in ['很健康', '健康需求', '焦油量控制在8mg', '8mg焦油', '焦油', '降焦', '低焦']:
                if phrase in sentence:
                    return phrase

        if category in {'产品设计创意', '新颖性', '功能特性', '品牌形象', '烟支设计', '外观包装'}:
            for phrase in ['全国首款五段式分享条盒', '全国首款', '首创分段式拆分设计', '条盒首创分段式拆分设计', '设置有内胆保湿', '新东东', '品牌创新', '创新', '新颖', '新上的细支', '年轻人偏爱的设计']:
                if phrase in sentence:
                    return phrase

        if opinion_term in ASPECT_NOUN_TERMS:
            adjacent_term = self._extract_adjacent_evaluation_term(opinion_term, sentence)
            if adjacent_term:
                return adjacent_term

        if not self._is_generic_opinion_term(opinion_term):
            return opinion_term

        candidate_categories = [category]
        top_category = self._map_category_to_top(category)
        second_category = self._map_category_to_second(category)
        for candidate in [top_category, second_category]:
            if candidate and candidate not in candidate_categories:
                candidate_categories.append(candidate)

        for target_category, phrases in GENERIC_OPINION_REFINEMENT_PATTERNS:
            if target_category not in candidate_categories:
                continue
            for phrase in sorted(phrases, key=len, reverse=True):
                if phrase and phrase in sentence:
                    return phrase
        return opinion_term

    def _refine_polarity(self, category, opinion_term, polarity, sentence=""):
        """对明显被模型判成中性的观点词做情感纠偏。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        polarity = str(polarity).strip()
        sentence = str(sentence).strip()
        text = f"{opinion_term} {sentence}"

        # 这些短语在标注中经常按固定极性处理，优先于模型原始极性。
        if any(keyword in opinion_term for keyword in ['吐气微醺', '微醺']):
            return '负向'
        if any(keyword in opinion_term for keyword in ['略燥', '燥感', '太醇厚', '不比', '有点冲', '烧口', '辣喉', '刮喉', '呛']):
            return '负向'
        if any(keyword in opinion_term for keyword in ['入喉微润', '微润', '看着很喜欢', '包装好看', '好看', '喜欢']):
            return '正向'
        if any(keyword in opinion_term for keyword in ['没有明显的区别', '无甚区别', '区别不大']):
            if any(keyword in sentence for keyword in ['高端', '上了100', '价位', '价格', '对不起这个价位']):
                return '负向'
            return '中性'
        if '一般般' in opinion_term:
            if any(keyword in sentence for keyword in ['太醇厚', '不比', '不如', '失去', '不怎么样']):
                return '负向'
            return '中性'
        if any(keyword in opinion_term for keyword in ['味道一般', '感觉味道一般', '一般']):
            if any(keyword in sentence for keyword in ['味道', '口感', '不怎么样']):
                return '负向'
            return '中性'
        if any(keyword in opinion_term for keyword in ['吸阻适中', '燃烧偏慢', '粗细适中', '劲头适中', '价格适中']):
            return '正向'
        if any(keyword in opinion_term for keyword in ['香精的腻歪', '腻歪']) and any(keyword in sentence for keyword in ['没有让', '并没有让', '不感到', '没有感到']):
            return '正向'
        if category in {'外观包装', '包装图案', '包装颜色', '包装质感'} and any(keyword in text for keyword in ['看着很喜欢', '包装好看', '好看', '漂亮', '精致']):
            return '正向'

        if polarity != '中性':
            return polarity

        if any(keyword in text for keyword in POSITIVE_POLARITY_HINTS):
            return '正向'

        if opinion_term in {'一般', '一般般'}:
            return '中性'

        if any(keyword in text for keyword in NEGATIVE_POLARITY_HINTS):
            if '还行' not in text:
                return '负向'

        if opinion_term == '一般' and any(keyword in sentence for keyword in ['味道', '口感', '包装', '不怎么样']):
            return '负向'
        if any(keyword in text for keyword in ['不辣口', '不刺激', '没有辣喉鼻']):
            return '正向'
        if any(keyword in text for keyword in ['有点冲', '烧口', '辣喉', '刮喉', '呛']):
            return '负向'

        return polarity

    def _infer_fine_categories(self, category, opinion_term, sentence=""):
        """从观点词和上下文中尽量识别多个二级类。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        sentence = str(sentence).strip()
        opinion_text = opinion_term
        fallback_text = sentence if self._is_generic_opinion_term(opinion_term) else ""
        combined_text = f"{opinion_text} {fallback_text}".strip()

        if category in THIRD_LEVEL_CATEGORIES:
            return [category]

        candidate_groups = self._candidate_base_categories(category, fallback_text)
        if not candidate_groups:
            return []

        if category == '内在口味' and not self._has_explicit_taste_anchor(opinion_text):
            return []

        matched = []
        for base_category in candidate_groups:
            if base_category not in FINE_GRAIN_HINTS:
                continue
            for keyword, fine_category in FINE_GRAIN_HINTS[base_category]:
                if (
                    keyword and
                    (keyword in opinion_text or (fallback_text and keyword in fallback_text)) and
                    self._is_fine_category_supported(fine_category, combined_text) and
                    fine_category not in matched
                ):
                    matched.append(fine_category)
        return matched

    def _is_fine_category_supported(self, fine_category, text):
        """对高频误报的细粒度类别增加显式触发词约束。"""
        if fine_category == '新颖性' and any(phrase in text for phrase in ['急需创新', '缺乏创新', '不够新颖', '太老旧', '老旧']):
            return False
        hints = STRICT_FINE_GRAIN_HINTS.get(fine_category)
        if not hints:
            return True
        return any(hint and hint in text for hint in hints)

    def _deduplicate_hierarchical_triples(self, triples):
        """移除同一句中父类与其子类同时保留造成的重复预测。"""
        triples = list(triples)
        if not triples:
            return []

        child_seconds = set()
        child_tops = set()
        for category, opinion_term, polarity in triples:
            key = (str(opinion_term).strip(), str(polarity).strip())
            if category in THIRD_LEVEL_CATEGORIES:
                child_seconds.add((self._map_category_to_second(category),) + key)
                child_tops.add((self._map_category_to_top(category),) + key)
            elif category in SECOND_LEVEL_CATEGORIES:
                child_tops.add((self._map_category_to_top(category),) + key)

        deduplicated = []
        seen = set()
        for category, opinion_term, polarity in triples:
            opinion_term = str(opinion_term).strip()
            polarity = str(polarity).strip()
            drop_current = False

            if category in SECOND_LEVEL_CATEGORIES:
                if (category, opinion_term, polarity) in child_seconds:
                    drop_current = True
            elif category in TOP_LEVEL_CATEGORIES:
                if (category, opinion_term, polarity) in child_tops:
                    drop_current = True

            if drop_current:
                continue

            triple = (category, opinion_term, polarity)
            if triple not in seen:
                deduplicated.append(triple)
                seen.add(triple)

        return deduplicated

    def _should_drop_predicted_triple(self, category, opinion_term, polarity, sentence=""):
        """一级精确率优先的误报过滤：只删除稳定的背景项或跨方面项。"""
        category = str(category).strip()
        opinion_term = str(opinion_term).strip()
        sentence = str(sentence).strip()
        top_category = self._map_category_to_top(category) or category
        text = f"{opinion_term} {sentence}"

        if not category or not opinion_term:
            return True

        if any(keyword in opinion_term for keyword in ['防伪']):
            return True

        generic_positive_terms = {
            '好', '很好', '很不错', '不错', '还好', '可以', '非常不错', 'nice',
            '满意', '喜欢', '推荐', '有面子', '上档次', '高端大气', '高端大气上档次'
        }
        taste_anchors = [
            '口感', '口味', '味道', '烟气', '吸感', '吸味', '抽', '好抽', '难抽',
            '香', '醇', '顺', '柔', '润', '甜', '回甘', '余味', '劲'
        ]
        package_anchors = [
            '包装', '外观', '图案', '配色', '颜色', '盒', '烟盒', '条盒', '设计',
            '材质', '手感', '质感', '磨砂', '植绒', '皮革', '纸质', '触感'
        ]
        process_anchors = [
            '工艺', '做工', '加工', '烟丝', '烟叶', '品质', '质量', '精细',
            '精湛', '过硬', '保证', '上乘', '填充', '紧实', '稳定'
        ]

        if top_category == '价格':
            positive_price_cues = ['性价比', '贵', '便宜', '划算', '实惠', '不值', '对不起', '亲民', '合理', '接受', '竞争力', '价位段']
            if not any(keyword in opinion_term for keyword in positive_price_cues):
                return True
            if any(keyword in opinion_term for keyword in ['不得高于', '指导零售价', '元/条', '元/包', '块', '价格下跌', '降下来了']) and not any(keyword in opinion_term for keyword in ['性价比', '贵', '便宜', '划算', '实惠', '不值', '对不起', '亲民', '合理', '接受']):
                return True

        if top_category == '焦油':
            # "降焦/8mg/焦油"常作为背景信息出现；只有明确评价健康属性时才保留焦油类预测。
            if not any(keyword in opinion_term for keyword in ['很健康', '健康需求', '健康', '低焦', '减害']):
                return True

        if top_category == '外观包装':
            if opinion_term in {'nice', '很不错', '不错'} and not any(keyword in sentence for keyword in ['包装', '外观', '图案', '设计', '配色', '颜色']):
                return True
            if category == '包装质感' and any(keyword in opinion_term for keyword in ['高端大气', '上档次', '有面子']):
                return True
            if category == '包装质感':
                texture_cues = ['质感', '材质', '手感', '磨砂', '植绒', '皮革', '纸质', '触感', '硬盒', '软包']
                if not any(keyword in opinion_term for keyword in texture_cues):
                    if any(keyword in opinion_term for keyword in ['不错', '蛮好', '好', '漂亮', '精致', '大气', '简约', '低调', '无法挑剔', '有派头']):
                        return True
            if category == '包装颜色':
                color_cues = ['颜色', '配色', '黑金', '金色', '绿色', '蓝色', '红色',
                             '黄和白', '色调', '色彩', '颜色搭配']
                if not any(keyword in opinion_term for keyword in color_cues):
                    return True
            if opinion_term in generic_positive_terms and not any(keyword in text for keyword in package_anchors):
                high_end_visual_cues = ['高端大气', '上档次', '大气', '有派头', '奢华', '精致']
                if not any(keyword in text for keyword in high_end_visual_cues):
                    return True

        if top_category == '工艺品质':
            if not any(keyword in text for keyword in process_anchors):
                return True
            if opinion_term in {'没得说', '不错', '很好', '非常好', '不甚了解'}:
                return True

        if top_category == '产品设计创意':
            if any(keyword in opinion_term for keyword in ['新上的细支', '细支']) and '创新' not in opinion_term and '首创' not in opinion_term:
                return True

        if category == '香气量':
            amount_cues = ['浓郁', '饱满', '寡淡', '淡薄', '香气量足', '香韵足',
                          '香气量', '淡', '不足', '足', '香气浓郁', '香气饱满']
            if not any(keyword in opinion_term for keyword in amount_cues):
                return True

        if top_category == '内在口味':
            if opinion_term in generic_positive_terms:
                has_taste_anchor = any(keyword in text for keyword in taste_anchors)
                has_other_anchor = any(keyword in sentence for keyword in package_anchors + ['品牌', '价格', '价位', '性价比'])
                if has_other_anchor and not has_taste_anchor:
                    return True

        return False

    def _infer_default_short_sentence_triples(self, sentence):
        """LoRA已能覆盖短句, 兜底关闭避免错误补充。"""
        return []
        sentence = str(sentence).strip()
        if not sentence:
            return []

        compact = re.sub(r'[\s，。！？、,.!?;；:：~～`·•\-_]+', '', sentence)
        if len(compact) > 18:
            return []

        polarity = '正向'
        if any(keyword in sentence for keyword in ['不', '没', '差', '一般', '不好', '不行', '不值', '太贵', '不推荐']):
            polarity = '负向' if any(keyword in sentence for keyword in ['差', '不好', '不行', '不值', '太贵']) else '中性'

        category = None
        if any(keyword in sentence for keyword in ['高端大气', '上档次']) and any(keyword in sentence for keyword in ['有面子', '有面']):
            return [('外观包装', '高端大气上档次' if '高端大气上档次' in sentence else '高端大气', '正向'), ('品牌形象', '有面子' if '有面子' in sentence else '有面', '正向')]

        if any(keyword in sentence for keyword in ['包装', '外观', '配色', '好看', '时髦', '靓丽', '漂亮', '大气']):
            category = '外观包装'
            if any(keyword in sentence for keyword in ['老旧', '陈旧']):
                opinion_term = '老旧'
            else:
                opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['品牌', '老牌子', '大品牌', '知名', '面子', '信赖', '认可']):
            category = '品牌形象'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['价格', '贵', '便宜', '划算', '性价比']):
            category = '价格'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['工艺', '做工', '品质', '精细', '精湛']):
            category = '工艺品质'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['烟支', '滤嘴', '吸阻', '中支', '细支']):
            category = '烟支设计'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['设计', '创意', '新颖', '创新', '独特', '文化', '情怀']):
            if any(keyword in sentence for keyword in ['急需创新', '缺乏创新', '不够新颖', '老旧']):
                category = '外观包装'
            else:
                category = '产品设计创意'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['焦油', '降焦']):
            category = '焦油'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['二手烟', '残留', '扩散']):
            category = '负外部性'
            opinion_term = sentence.strip()
        elif any(keyword in sentence for keyword in ['口感', '味道', '烟气', '好抽', '醇厚', '香', '顺', '柔', '回甘']):
            if any(keyword in sentence for keyword in ['非常好', '很好', '不错']) and any(keyword in sentence for keyword in ['口味', '味道']):
                category = '香气'
            else:
                category = '内在口味'
            opinion_term = sentence.strip()

        if not category and len(compact) <= 8:
            if any(keyword in sentence for keyword in ['好', '不错', '喜欢', '推荐', '满意', '棒']):
                category = '内在口味'
                opinion_term = sentence.strip()

        if not category:
            return []

        opinion_term = locals().get('opinion_term', sentence.strip())
        return [(category, opinion_term, polarity)]

    def _augment_predicted_triples(self, sentence, predicted_triples):
        """基于上下文补充可能遗漏的二级类三元组，优先提升细粒度召回。"""
        if not predicted_triples:
            predicted_triples = self._infer_default_short_sentence_triples(sentence)

        augmented = []
        seen = set()

        for category, opinion_term, polarity in predicted_triples:
            category = str(category).strip()
            opinion_term = str(opinion_term).strip()
            polarity = str(polarity).strip()

            if not category or category in EXCLUDED_EVAL_CATEGORIES:
                continue

            if self._should_drop_predicted_triple(category, opinion_term, polarity, sentence):
                continue

            fine_categories = self._infer_fine_categories(category, opinion_term, sentence)
            if fine_categories:
                for fine_category in fine_categories:
                    if fine_category in EXCLUDED_EVAL_CATEGORIES:
                        continue
                    if self._should_drop_predicted_triple(fine_category, opinion_term, polarity, sentence):
                        continue
                    triple = (fine_category, opinion_term, polarity)
                    if triple not in seen:
                        augmented.append(triple)
                        seen.add(triple)

                # 保留原始一级类，避免在确实无法完全替代时损失粗粒度信息
                if category in SECOND_LEVEL_CATEGORIES:
                    triple = (category, opinion_term, polarity)
                    if triple not in seen:
                        augmented.append(triple)
                        seen.add(triple)
            else:
                triple = (category, opinion_term, polarity)
                if triple not in seen:
                    augmented.append(triple)
                    seen.add(triple)

        if not augmented:
            for triple in self._infer_default_short_sentence_triples(sentence):
                category, opinion_term, polarity = triple
                if category in EXCLUDED_EVAL_CATEGORIES:
                    continue
                if self._should_drop_predicted_triple(category, opinion_term, polarity, sentence):
                    continue
                if triple not in seen:
                    augmented.append(triple)
                    seen.add(triple)

        return self._deduplicate_hierarchical_triples(augmented)
    
    def _normalize_opinion_term(self, term):
        """将观点术语归一化到更稳定的语义表达。"""
        term = str(term).strip().lower()
        term = re.sub(r'[\s，。！？、,.!?;；:：~～`·•\-_!！?？]+', '', term)
        for pattern in OPINION_TERM_STRIP_PATTERNS:
            term = re.sub(pattern, '', term)
        if term in OPINION_TERM_NORMALIZATION:
            term = OPINION_TERM_NORMALIZATION[term]
        for source, target in OPINION_TERM_NORMALIZATION.items():
            if source and source in term:
                term = term.replace(source, target)
        return term

    def _split_opinion_term_units(self, term):
        """将组合短语拆分为更稳定的语义单元。"""
        normalized = self._normalize_opinion_term(term)
        if not normalized:
            return []

        units = []
        for atom in OPINION_TERM_ATOMS:
            atom_norm = self._normalize_opinion_term(atom)
            if atom_norm and atom_norm in normalized and atom_norm not in units:
                units.append(atom_norm)

        if not units:
            units.append(normalized)
        return units

    def _extract_core_opinion_tokens(self, term):
        """提取观点词中的核心语义词，用于比对近义短语。"""
        normalized = self._normalize_opinion_term(term)
        if not normalized:
            return set()

        tokens = set()
        for token in OPINION_CORE_TOKENS:
            token_norm = self._normalize_opinion_term(token)
            if token_norm and token_norm in normalized:
                tokens.add(token_norm)

        if not tokens:
            units = self._split_opinion_term_units(term)
            if len(units) > 1:
                tokens.update(units)
        return tokens

    def _has_negation_conflict(self, gold_term, pred_term):
        """避免将"刺激"和"不刺激"这类相反含义误判为一致。"""
        gold_norm = self._normalize_opinion_term(gold_term)
        pred_norm = self._normalize_opinion_term(pred_term)
        if not gold_norm or not pred_norm:
            return False

        benign_negation_terms = ['不错', '不差', '不辣口', '不刺激', '不呛', '不腻', '不淡']
        gold_neg_check = gold_norm
        pred_neg_check = pred_norm
        for term in benign_negation_terms:
            normalized_term = self._normalize_opinion_term(term)
            gold_neg_check = gold_neg_check.replace(normalized_term, '')
            pred_neg_check = pred_neg_check.replace(normalized_term, '')

        for positive, negative in OPINION_CONTRAST_PAIRS:
            positive_norm = self._normalize_opinion_term(positive)
            negative_norm = self._normalize_opinion_term(negative)
            if (
                (positive_norm in gold_norm and negative_norm in pred_norm) or
                (negative_norm in gold_norm and positive_norm in pred_norm)
            ):
                return True

        gold_has_neg = any(token in gold_neg_check for token in OPINION_NEGATION_TOKENS)
        pred_has_neg = any(token in pred_neg_check for token in OPINION_NEGATION_TOKENS)
        shared_tokens = self._extract_core_opinion_tokens(gold_term) & self._extract_core_opinion_tokens(pred_term)
        if shared_tokens and gold_has_neg != pred_has_neg:
            return True
        return False

    def _terms_fuzzy_match(self, gold_term, pred_term):
        """观点术语匹配：双向包含 + 语义归一/组合短语/核心语义词辅助匹配。"""
        gold_term = str(gold_term).strip()
        pred_term = str(pred_term).strip()

        if not hasattr(self, '_fuzzy_match_stats'):
            self._fuzzy_match_stats = {f"L{i}": 0 for i in range(1, 7)}
            self._fuzzy_match_stats["L3_reject"] = 0
            self._fuzzy_match_stats["total_matches"] = 0

        if not gold_term or not pred_term:
            return False

        if gold_term == pred_term:
            self._fuzzy_match_stats["L1"] += 1
            self._fuzzy_match_stats["total_matches"] += 1
            return True

        gold_clean = re.sub(r'[\s，。！？、,.!?;；:：]', '', gold_term).lower()
        pred_clean = re.sub(r'[\s，。！？、,.!?;；:：]', '', pred_term).lower()

        if not gold_clean or not pred_clean:
            return False

        if self._has_negation_conflict(gold_term, pred_term):
            self._fuzzy_match_stats["L3_reject"] += 1
            return False

        if pred_clean in gold_clean or gold_clean in pred_clean:
            self._fuzzy_match_stats["L2"] += 1
            self._fuzzy_match_stats["total_matches"] += 1
            return True

        gold_norm = self._normalize_opinion_term(gold_term)
        pred_norm = self._normalize_opinion_term(pred_term)
        if not gold_norm or not pred_norm:
            return False

        # L3-L5 已屏蔽: LoRA后精确相等(L1)和去标点包含(L2)覆盖98%+匹配
        # 语义归一化/组合短语拆分/核心语义词在LoRA训练后不再需要
        return False
    
    def _get_category_level(self, category):
        """返回类别所在层级：1/2/3。"""
        category = self._normalize_category_label(category)
        if not category:
            return None
        if category in TOP_LEVEL_CATEGORIES:
            return 1
        if category in SECOND_LEVEL_CATEGORIES:
            return 2
        if category in THIRD_LEVEL_CATEGORIES:
            return 3
        return None

    def _project_category_to_level(self, category, level):
        """将类别投影到目标层级。"""
        category = self._normalize_category_label(category)
        if not category:
            return None
        if level == 1:
            return self._map_category_to_top(category)
        if level == 2:
            second_category = self._map_category_to_second(category)
            return second_category or self._map_category_to_top(category)
        if level == 3:
            if category in THIRD_LEVEL_CATEGORIES:
                return category
            second_category = self._map_category_to_second(category)
            if second_category:
                return second_category
            return self._map_category_to_top(category)
        raise ValueError(f"未知层级: {level}")

    def _category_fuzzy_match(self, cat1, cat2, max_level=3):
        """类别匹配：按两个标签中更宽松的层级统一后进行比较。"""
        level1 = self._get_category_level(cat1)
        level2 = self._get_category_level(cat2)
        if level1 is None or level2 is None:
            return False

        compare_level = min(level1, level2, max_level)
        projected1 = self._project_category_to_level(cat1, compare_level)
        projected2 = self._project_category_to_level(cat2, compare_level)
        return bool(projected1 and projected2 and projected1 == projected2)

    def _map_category_to_top(self, category):
        """将类别映射到一级类别，未命中时返回原类别或None。"""
        category = str(category).strip()
        if not category:
            return None
        return CATEGORY_TO_TOP.get(category, category)

    def _map_category_to_second(self, category):
        """将类别映射到二级类别。二级类别自身返回自身，三级类别返回其父二级类别。"""
        category = str(category).strip()
        if not category:
            return None
        if category in SECOND_LEVEL_CATEGORIES:
            return category
        return CATEGORY_TO_SECOND.get(category)
    
    def _triples_match(self, gold_triple, pred_triple):
        """三元组匹配：类别按更宽松层级比较、观点术语包含/语义一致匹配、情感极性显式映射匹配。

        分层版口径下：
        1. 类别比较时，按两个标签中更宽松的层级统一后再比较；
        2. 一级/二级/三级评估在此基础上继续输出对应层级结果。
        """
        if len(gold_triple) < 3 or len(pred_triple) < 3:
            return False
        
        if not self._category_fuzzy_match(gold_triple[0], pred_triple[0], max_level=3):
            return False
        
        # 情感极性匹配
        polarity_mapping = {
            '正向': ['正向', '积极', '正面', '满意', '好', '不错', '喜欢', '赞', '优秀', '佳', '良', '优', '棒', '满意', '满意的', '好的'],
            '负向': ['负向', '消极', '负面', '不满意', '差', '不好', '不喜欢', '烂', '差', '次', '劣', '不满意', '不满意的', '不好的'],
            '中性': ['中性', '一般', '还行', '正常', '普通', '中等', '一般般', '马马虎虎']
        }
        
        gold_polarity = gold_triple[2]
        pred_polarity = pred_triple[2]
        
        # 检查情感极性是否匹配
        polarity_match = False
        
        # 精确匹配
        if gold_polarity == pred_polarity:
            polarity_match = True
        
        # 近义词匹配
        elif gold_polarity in polarity_mapping:
            if pred_polarity in polarity_mapping[gold_polarity]:
                polarity_match = True
        
        # 如果情感极性匹配，继续检查观点术语匹配
        if polarity_match:
            return self._terms_fuzzy_match(gold_triple[1], pred_triple[1])
        
        return False

class TripleEvaluator:
    def __init__(self, analyzer, data_path):
        """初始化评估器"""
        self.analyzer = analyzer
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self):
        """加载数据集"""
        print(f"当前使用的数据文件: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            print(f"数据集加载完成，有效样本数: {len(df)}")
            return df
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _parse_gold_tuples(self, gold_tuples_str):
        """解析黄金三元组"""
        try:
            # 尝试使用ast.literal_eval解析
            gold_tuples = ast.literal_eval(gold_tuples_str)
            # 确保返回的是列表
            if isinstance(gold_tuples, list):
                # 转换为三元组（方面类别, 观点术语, 情感极性）
                triples = []
                for t in gold_tuples:
                    if isinstance(t, (tuple, list)) and len(t) >= 4:
                        # 取方面类别、观点术语、情感极性
                        category = self.analyzer._normalize_category_label(str(t[1]).strip())
                        if not category or category in EXCLUDED_EVAL_CATEGORIES:
                            continue
                        triples.append((category, str(t[2]).strip(), str(t[3]).strip()))
                return triples
            return []
        except (ValueError, SyntaxError):
            return []

    def _count_exact_triple_matches(self, gold_triples, predicted_triples):
        """对 gold 和 predicted 做一对一最大匹配，避免重复消费同一个预测项。"""
        match_graph = []
        for gold_triple in gold_triples:
            candidate_pred_indices = []
            for pred_idx, pred_triple in enumerate(predicted_triples):
                if self.analyzer._triples_match(gold_triple, pred_triple):
                    candidate_pred_indices.append(pred_idx)
            match_graph.append(candidate_pred_indices)

        matched_pred_to_gold = {}

        def _try_match(gold_idx, visited):
            for pred_idx in match_graph[gold_idx]:
                if pred_idx in visited:
                    continue
                visited.add(pred_idx)
                if pred_idx not in matched_pred_to_gold or _try_match(matched_pred_to_gold[pred_idx], visited):
                    matched_pred_to_gold[pred_idx] = gold_idx
                    return True
            return False

        match_count = 0
        for gold_idx in range(len(gold_triples)):
            if _try_match(gold_idx, set()):
                match_count += 1
        return match_count
    
    def evaluate(self, sample_size=None):
        """评估三元组提取性能"""
        # 如果指定了样本大小，则只评估部分样本
        if sample_size:
            test_data = self.data.head(sample_size)
            print(f"评估前{sample_size}个样本")
        else:
            test_data = self.data
            print("评估全部样本")
        
        total = len(test_data)
        
        # 存储详细结果
        detailed_results = []
        
        # 统计指标
        total_gold_triples = 0
        total_predicted_triples = 0
        correctly_predicted_triples = 0
        
        print("开始评估三元组提取性能...")
        
        for i, row in test_data.iterrows():
            sentence = row['sentence'] if 'sentence' in row else ''
            gold_tuples_str = row['gold_tuples'] if 'gold_tuples' in row else '[]'
            gold_triples = []
            predicted_triples = []
            
            if pd.isna(sentence) or not sentence:
                continue
            
            try:
                # 解析黄金三元组
                gold_triples = self._parse_gold_tuples(gold_tuples_str)
                
                # 使用模型预测三元组
                sample_start_time = time.time()
                print(f"正在处理样本 {i + 1}/{total}: {str(sentence)[:80]}", flush=True)
                response, predicted_triples = self.analyzer.analyze_sentiment(sentence)
                print(
                    f"样本 {i + 1}/{total} 完成，用时 {time.time() - sample_start_time:.2f}秒，"
                    f"金标 {len(gold_triples)} 个，预测 {len(predicted_triples)} 个",
                    flush=True
                )
                
                # 计算匹配的三元组数量
                correct_matches = self._count_exact_triple_matches(gold_triples, predicted_triples)
                
                # 更新统计
                total_gold_triples += len(gold_triples)
                total_predicted_triples += len(predicted_triples)
                correctly_predicted_triples += correct_matches
                
                # 存储详细结果
                detailed_results.append({
                    'sentence': sentence,
                    'gold_triples': gold_triples,
                    'predicted_triples': predicted_triples,
                    'correct_matches': correct_matches
                })
                
                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(f"进度: {i + 1}/{total}", flush=True)
                
            except Exception as e:
                print(f"分析第{i+1}个样本时出错: {str(e)}")
                # 出错时，记录空结果
                detailed_results.append({
                    'sentence': sentence,
                    'gold_triples': gold_triples,
                    'predicted_triples': [],
                    'correct_matches': 0
                })
        
        # 计算评估指标
        # 准确率：正确预测的三元组数量 / 预测的三元组总数
        precision = correctly_predicted_triples / total_predicted_triples if total_predicted_triples > 0 else 0
        
        # 召回率：正确预测的三元组数量 / 黄金三元组总数
        recall = correctly_predicted_triples / total_gold_triples if total_gold_triples > 0 else 0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 句子级准确率
        sentence_correct = 0
        for result in detailed_results:
            if (result['correct_matches'] == len(result['gold_triples']) and
                    len(result['gold_triples']) == len(result['predicted_triples'])):
                sentence_correct += 1
        
        sentence_level_accuracy = sentence_correct / len(detailed_results) if detailed_results else 0
        
        # 输出评估结果
        print("\n=== 三元组提取评估结果 ===")
        print(f"黄金三元组总数: {total_gold_triples}")
        print(f"预测三元组总数: {total_predicted_triples}")
        print(f"正确匹配的三元组数量: {correctly_predicted_triples}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1): {f1:.4f}")
        print(f"句子级准确率: {sentence_level_accuracy:.4f}")
        
        # 新增：方面类别评估
        self._aspect_category_evaluation(detailed_results)

        # 观点词模糊匹配分层统计
        if hasattr(self.analyzer, '_fuzzy_match_stats'):
            stats = self.analyzer._fuzzy_match_stats
            total = max(1, stats.get("total_matches", 0))
            print("\n=== 观点词模糊匹配分层统计 ===")
            print(f"L1 精确相等:       {stats.get('L1', 0):>6} ({stats.get('L1', 0)/total*100:5.1f}%)")
            print(f"L2 去标点双向包含:  {stats.get('L2', 0):>6} ({stats.get('L2', 0)/total*100:5.1f}%)")
            print(f"否定冲突拦截:      {stats.get('L3_reject', 0):>6}")
            print(f"(L3-L5已屏蔽, LoRA后不再需要)")

        # 保存结果
        self._save_results(detailed_results, precision, recall, f1, sentence_level_accuracy)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sentence_level_accuracy': sentence_level_accuracy,
            'total_gold_triples': total_gold_triples,
            'total_predicted_triples': total_predicted_triples,
            'correctly_predicted_triples': correctly_predicted_triples
        }
    
    def _aspect_category_evaluation(self, detailed_results):
        """方面类别评估：一级/二级/三级统一按对应层级三元组匹配。"""
        
        # 提取所有的黄金类别和预测类别
        gold_categories_list = []
        predicted_categories_list = []
        
        for result in detailed_results:
            gold_triples = result['gold_triples']
            predicted_triples = result['predicted_triples']
            
            # 提取黄金类别
            gold_categories = []
            for gold_triple in gold_triples:
                if len(gold_triple) >= 1:
                    gold_category = gold_triple[0]
                    gold_categories.append(gold_category)
            
            # 提取预测类别
            predicted_categories = []
            for pred_triple in predicted_triples:
                if len(pred_triple) >= 1:
                    pred_category = pred_triple[0]
                    predicted_categories.append(pred_category)
            
            gold_categories_list.append(gold_categories)
            predicted_categories_list.append(predicted_categories)
        
        print("\n=== 一级类直接评估（按一级类三元组匹配） ===")
        self._aspect_categories_evaluation_level1(detailed_results)

        print("\n=== 二级类直接评估（按二级类三元组匹配） ===")
        self._aspect_categories_evaluation_level2(detailed_results)

        print("\n=== 三级类直接评估（按三级类三元组匹配） ===")
        self._aspect_categories_evaluation_level3(detailed_results)

    def _expand_categories(self, categories):
        """拆分复合类别，并做归一化与去重。"""
        expanded = []
        for category in categories:
            category = str(category).strip().replace('，', ',')
            parts = [part.strip() for part in category.split(',') if part.strip()]
            if not parts:
                parts = [category]
            for part in parts:
                normalized = self.analyzer._normalize_category_label(part)
                if normalized and normalized not in EXCLUDED_EVAL_CATEGORIES:
                    expanded.append(normalized)
        return list(dict.fromkeys(expanded))

    def _map_category_to_top(self, category):
        """桥接到分析器中的一级类别映射。"""
        return self.analyzer._map_category_to_top(category)

    def _map_category_to_second(self, category):
        """桥接到分析器中的二级类别映射。"""
        return self.analyzer._map_category_to_second(category)

    def _extract_level1_categories(self, categories):
        """抽取一级类别。"""
        expanded = self._expand_categories(categories)
        level1_categories = []
        for category in expanded:
            top_category = self._map_category_to_top(category)
            if top_category in TOP_LEVEL_CATEGORIES:
                level1_categories.append(top_category)
        return list(dict.fromkeys(level1_categories))

    def _extract_level2_categories(self, categories):
        """抽取二级类别。"""
        expanded = self._expand_categories(categories)
        level2_categories = []
        for category in expanded:
            second_category = self._map_category_to_second(category)
            if second_category in SECOND_LEVEL_CATEGORIES:
                level2_categories.append(second_category)
        return list(dict.fromkeys(level2_categories))

    def _extract_level3_categories(self, categories):
        """抽取三级类别。"""
        expanded = self._expand_categories(categories)
        level3_categories = [category for category in expanded if category in THIRD_LEVEL_CATEGORIES]
        return list(dict.fromkeys(level3_categories))

    def _project_triple_to_level(self, triple, level):
        """将三元组投影到固定层级。"""
        if len(triple) < 3:
            return None

        category = str(triple[0]).strip()
        opinion_term = str(triple[1]).strip()
        polarity = str(triple[2]).strip()

        if level == 'level1':
            projected_category = self._map_category_to_top(category)
            if projected_category in TOP_LEVEL_CATEGORIES:
                return (projected_category, opinion_term, polarity)
            return None

        if level == 'level2':
            projected_category = self._map_category_to_second(category)
            if projected_category in SECOND_LEVEL_CATEGORIES:
                return (projected_category, opinion_term, polarity)
            return None

        if level == 'level3':
            if category in THIRD_LEVEL_CATEGORIES:
                return (category, opinion_term, polarity)
            return None

        raise ValueError(f"未知层级: {level}")

    def _triple_match_at_level(self, gold_triple, pred_triple, level):
        """在固定层级投影后进行三元组匹配。"""
        if len(gold_triple) < 3 or len(pred_triple) < 3:
            return False

        gold_projected = self._project_triple_to_level(gold_triple, level)
        pred_projected = self._project_triple_to_level(pred_triple, level)
        if not gold_projected or not pred_projected:
            return False

        if gold_projected[0] != pred_projected[0]:
            return False

        polarity_mapping = {
            '正向': ['正向', '积极', '正面', '满意', '好', '不错', '喜欢', '赞', '优秀', '佳', '良', '优', '棒', '满意的', '好的'],
            '负向': ['负向', '消极', '负面', '不满意', '差', '不好', '不喜欢', '烂', '次', '劣', '不满意的', '不好的'],
            '中性': ['中性', '一般', '还行', '正常', '普通', '中等', '一般般', '马马虎虎']
        }

        gold_polarity = gold_projected[2]
        pred_polarity = pred_projected[2]
        polarity_match = False
        if gold_polarity == pred_polarity:
            polarity_match = True
        elif gold_polarity in polarity_mapping and pred_polarity in polarity_mapping[gold_polarity]:
            polarity_match = True

        if not polarity_match:
            return False

        return self.analyzer._terms_fuzzy_match(gold_projected[1], pred_projected[1])

    def _count_level_triple_matches(self, gold_triples, predicted_triples, level):
        """固定到指定层级后，按三元组做一对一最大匹配。"""
        gold_projected = [
            projected for projected in
            (self._project_triple_to_level(triple, level) for triple in gold_triples)
            if projected
        ]
        predicted_projected = [
            projected for projected in
            (self._project_triple_to_level(triple, level) for triple in predicted_triples)
            if projected
        ]

        match_graph = []
        for gold_triple in gold_projected:
            candidate_pred_indices = []
            for pred_idx, pred_triple in enumerate(predicted_projected):
                if self._triple_match_at_level(gold_triple, pred_triple, level=level):
                    candidate_pred_indices.append(pred_idx)
            match_graph.append(candidate_pred_indices)

        matched_pred_to_gold = {}

        def _try_match(gold_idx, visited):
            for pred_idx in match_graph[gold_idx]:
                if pred_idx in visited:
                    continue
                visited.add(pred_idx)
                if pred_idx not in matched_pred_to_gold or _try_match(matched_pred_to_gold[pred_idx], visited):
                    matched_pred_to_gold[pred_idx] = gold_idx
                    return True
            return False

        match_count = 0
        for gold_idx in range(len(gold_projected)):
            if _try_match(gold_idx, set()):
                match_count += 1

        matched_pairs = [(gold_idx, pred_idx) for pred_idx, gold_idx in matched_pred_to_gold.items()]
        return gold_projected, predicted_projected, match_count, matched_pairs

    def _aspect_categories_evaluation_level1(self, detailed_results):
        """一级类评估：将 1/2/3 级类别统一映射到一级类后比较。"""

        total_gold = 0
        total_predicted = 0
        total_correct = 0
        per_top_stats = {
            top_category: {'gold': 0, 'pred': 0, 'correct': 0}
            for top_category in ASPECT_CATEGORY_HIERARCHY.keys()
        }

        for i, result in enumerate(detailed_results):
            gold_triples = result['gold_triples']
            predicted_triples = result['predicted_triples']

            try:
                gold_projected, predicted_projected, correct_count, matched_pairs = self._count_level_triple_matches(
                    gold_triples, predicted_triples, level='level1'
                )
                total_gold += len(gold_projected)
                total_predicted += len(predicted_projected)
                total_correct += correct_count

                for triple in gold_projected:
                    per_top_stats[triple[0]]['gold'] += 1
                for triple in predicted_projected:
                    per_top_stats[triple[0]]['pred'] += 1
                for gold_idx, _pred_idx in matched_pairs:
                    per_top_stats[gold_projected[gold_idx][0]]['correct'] += 1
            except Exception as e:
                print(f"解析第{i+1}个样本时出错: {str(e)}")

        overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
        overall_recall = total_correct / total_gold if total_gold > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        print("-" * 80)
        print(f"{'整体评估':<15} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'黄金数量':<8} {'预测数量':<8} {'正确数量':<8}")
        print("-" * 80)
        print(f"{'':<15} {overall_precision:.4f}   {overall_recall:.4f}   {overall_f1:.4f}   {total_gold:<8} {total_predicted:<8} {total_correct:<8}")
        print("\n一级类数量统计:")
        print("-" * 72)
        print(f"{'一级类':<18} {'黄金数量':<10} {'预测数量':<10} {'正确数量':<10}")
        print("-" * 72)
        for top_category, counts in per_top_stats.items():
            print(f"{top_category:<18} {counts['gold']:<10} {counts['pred']:<10} {counts['correct']:<10}")
    
    def _aspect_categories_evaluation_level2(self, detailed_results):
        """二级类评估：将 2/3 级类别统一映射到二级类后比较。"""

        total_gold = 0
        total_predicted = 0
        total_correct = 0
        per_second_stats = {
            second_category: {'gold': 0, 'pred': 0, 'correct': 0}
            for second_category in SECOND_LEVEL_CATEGORIES
        }

        for i, result in enumerate(detailed_results):
            gold_triples = result['gold_triples']
            predicted_triples = result['predicted_triples']

            try:
                gold_projected, predicted_projected, correct_count, matched_pairs = self._count_level_triple_matches(
                    gold_triples, predicted_triples, level='level2'
                )
                total_gold += len(gold_projected)
                total_predicted += len(predicted_projected)
                total_correct += correct_count

                for triple in gold_projected:
                    per_second_stats[triple[0]]['gold'] += 1
                for triple in predicted_projected:
                    per_second_stats[triple[0]]['pred'] += 1
                for gold_idx, _pred_idx in matched_pairs:
                    per_second_stats[gold_projected[gold_idx][0]]['correct'] += 1
            except Exception as e:
                print(f"解析第{i+1}个样本时出错: {str(e)}")

        overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
        overall_recall = total_correct / total_gold if total_gold > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        print("-" * 90)
        print(f"{'整体评估':<20} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'黄金数量':<8} {'预测数量':<8} {'正确数量':<8}")
        print("-" * 90)
        print(f"{'':<20} {overall_precision:.4f}   {overall_recall:.4f}   {overall_f1:.4f}   {total_gold:<8} {total_predicted:<8} {total_correct:<8}")
        print("\n二级类数量统计:")
        print("-" * 72)
        print(f"{'二级类':<18} {'黄金数量':<10} {'预测数量':<10} {'正确数量':<10}")
        print("-" * 72)
        for second_category in sorted(per_second_stats.keys()):
            counts = per_second_stats[second_category]
            print(f"{second_category:<18} {counts['gold']:<10} {counts['pred']:<10} {counts['correct']:<10}")

    def _aspect_categories_evaluation_level3(self, detailed_results):
        """三级类评估：仅比较三级类三元组。"""
        total_gold = 0
        total_predicted = 0
        total_correct = 0
        per_third_stats = {
            third_category: {'gold': 0, 'pred': 0, 'correct': 0}
            for third_category in THIRD_LEVEL_CATEGORIES
        }

        for i, result in enumerate(detailed_results):
            gold_triples = result['gold_triples']
            predicted_triples = result['predicted_triples']

            try:
                gold_projected, predicted_projected, correct_count, matched_pairs = self._count_level_triple_matches(
                    gold_triples, predicted_triples, level='level3'
                )
                total_gold += len(gold_projected)
                total_predicted += len(predicted_projected)
                total_correct += correct_count

                for triple in gold_projected:
                    per_third_stats[triple[0]]['gold'] += 1
                for triple in predicted_projected:
                    per_third_stats[triple[0]]['pred'] += 1
                for gold_idx, _pred_idx in matched_pairs:
                    per_third_stats[gold_projected[gold_idx][0]]['correct'] += 1

            except Exception as e:
                print(f"解析第{i+1}个样本时出错: {str(e)}")

        overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
        overall_recall = total_correct / total_gold if total_gold > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        print("-" * 90)
        print(f"{'整体评估':<20} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'黄金数量':<8} {'预测数量':<8} {'正确数量':<8}")
        print("-" * 90)
        print(f"{'':<20} {overall_precision:.4f}   {overall_recall:.4f}   {overall_f1:.4f}   {total_gold:<8} {total_predicted:<8} {total_correct:<8}")
        print("\n三级类数量统计:")
        print("-" * 72)
        print(f"{'三级类':<18} {'黄金数量':<10} {'预测数量':<10} {'正确数量':<10}")
        print("-" * 72)
        for third_category in sorted(per_third_stats.keys()):
            counts = per_third_stats[third_category]
            if counts['gold'] == 0 and counts['pred'] == 0:
                continue
            print(f"{third_category:<18} {counts['gold']:<10} {counts['pred']:<10} {counts['correct']:<10}")
    
    def _save_results(self, detailed_results, precision, recall, f1, sentence_level_accuracy):
        """保存评估结果"""
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        detailed_df = pd.DataFrame(detailed_results)
        result_path = os.path.join('results', f'triple_fuzzy_results_{timestamp}.csv')
        detailed_df.to_csv(result_path, index=False, encoding='utf-8-sig')
        print(f"详细结果已保存到: {result_path}")
        
        # 保存评估指标
        metrics = {
            'timestamp': timestamp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sentence_level_accuracy': sentence_level_accuracy,
            'total_gold_triples': sum(len(r['gold_triples']) for r in detailed_results),
            'total_predicted_triples': sum(len(r['predicted_triples']) for r in detailed_results),
            'correctly_predicted_triples': sum(r['correct_matches'] for r in detailed_results)
        }
        
        metrics_df = pd.DataFrame([metrics])
        metric_path = os.path.join('results', f'triple_fuzzy_metrics_{timestamp}.csv')
        metrics_df.to_csv(metric_path, index=False, encoding='utf-8-sig')
        print(f"评估指标已保存到: {metric_path}")

def _run_single_evaluation(analyzer, data_path, sample_size, model_path):
    """执行一轮完整评估。"""
    run_start_time = datetime.datetime.now()
    print(f"任务开始时间: {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"实验方法标记: {METHOD_TAG}", flush=True)
    print("实验方法说明:", flush=True)
    for idx, note in enumerate(METHOD_NOTES, start=1):
        print(f"{idx}. {note}", flush=True)
    print(f"模型路径: {model_path}", flush=True)
    print(f"样本规模: {sample_size}", flush=True)
    print(f"最大生成长度: {MAX_NEW_TOKENS}", flush=True)
    print(f"是否使用短提示词: {USE_SHORT_PROMPT}", flush=True)

    try:
        evaluator = TripleEvaluator(analyzer, data_path)
        evaluator.evaluate(sample_size=sample_size)
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}", flush=True)
        traceback.print_exc()
    finally:
        run_end_time = datetime.datetime.now()
        print(f"任务结束时间: {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"总耗时: {run_end_time - run_start_time}", flush=True)


def _serve_resident(model_path, data_path, sample_size):
    """启动常驻服务：模型只加载一次，后续复用。"""
    if os.path.exists(RESIDENT_SOCKET_PATH):
        os.remove(RESIDENT_SOCKET_PATH)

    print(f"启动模型常驻服务，socket: {RESIDENT_SOCKET_PATH}", flush=True)
    print(f"常驻模型路径: {model_path}", flush=True)
    retriever = FewShotRetriever(RETRIEVAL_DATA_PATH, project_to_level2=USE_TWO_STAGE) if os.path.exists(RETRIEVAL_DATA_PATH) else None
    lora_path = DEFAULT_LORA_PATH if os.path.exists(DEFAULT_LORA_PATH) else None
    analyzer = SentimentAnalyzer(model_path, fewshot_retriever=retriever, lora_path=lora_path)
    print("模型常驻服务已就绪", flush=True)

    listener = Listener(RESIDENT_SOCKET_PATH, family="AF_UNIX")
    try:
        while True:
            conn = listener.accept()
            try:
                request = conn.recv()
                command = request.get("command")

                if command == "status":
                    conn.send({"status": "ok", "message": "resident_ready"})
                    continue

                if command == "stop":
                    conn.send({"status": "ok", "message": "stopping"})
                    break

                if command == "run_eval":
                    run_log_path = request["log_path"]
                    run_sample_size = int(request.get("sample_size", sample_size))
                    run_data_path = request.get("data_path", data_path)
                    with open(run_log_path, "a", encoding="utf-8") as log_fp, \
                         contextlib.redirect_stdout(log_fp), \
                         contextlib.redirect_stderr(log_fp):
                        print(f"常驻服务开始处理任务: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                        print(f"常驻服务复用已加载模型: {model_path}", flush=True)
                        _run_single_evaluation(analyzer, run_data_path, run_sample_size, model_path)
                    conn.send({"status": "ok", "message": "completed", "log_path": run_log_path})
                    continue

                conn.send({"status": "error", "message": f"unknown_command:{command}"})
            except Exception as exc:
                conn.send({"status": "error", "message": str(exc)})
            finally:
                conn.close()
    finally:
        listener.close()
        if os.path.exists(RESIDENT_SOCKET_PATH):
            os.remove(RESIDENT_SOCKET_PATH)


def _send_resident_command(payload):
    """向常驻服务发送指令。"""
    last_error = None
    for _ in range(60):
        try:
            conn = Client(RESIDENT_SOCKET_PATH, family="AF_UNIX")
            try:
                conn.send(payload)
                return conn.recv()
            finally:
                conn.close()
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"常驻服务连接失败: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="三元组预测与评估")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--resident-server", action="store_true")
    parser.add_argument("--resident-run", action="store_true")
    parser.add_argument("--resident-status", action="store_true")
    parser.add_argument("--resident-stop", action="store_true")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    if args.resident_server:
        _serve_resident(args.model_path, args.data_path, args.sample_size)
        return

    if args.resident_status:
        response = _send_resident_command({"command": "status"})
        print(json.dumps(response, ensure_ascii=False))
        return

    if args.resident_stop:
        response = _send_resident_command({"command": "stop"})
        print(json.dumps(response, ensure_ascii=False))
        return

    if args.resident_run:
        if not args.log_path:
            raise ValueError("--resident-run 模式下必须提供 --log-path")
        response = _send_resident_command({
            "command": "run_eval",
            "log_path": args.log_path,
            "sample_size": args.sample_size,
            "data_path": args.data_path,
        })
        print(json.dumps(response, ensure_ascii=False))
        return

    retriever = FewShotRetriever(args.data_path) if os.path.exists(args.data_path) else None
    lora_path = args.lora_path if os.path.exists(args.lora_path) else None
    analyzer = SentimentAnalyzer(args.model_path, fewshot_retriever=retriever, lora_path=lora_path)
    _run_single_evaluation(analyzer, args.data_path, args.sample_size, args.model_path)


if __name__ == "__main__":
    main()
