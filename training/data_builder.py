"""
训练数据构建：从原始流水记录构建三元组（anchor, positive, negative）

anchor   - [收入]/[支出] + [币种] + [金额档位] + 流水摘要（去噪）+ 交易类型
positive - 正确科目名（account_chart.name）
negative - 硬负例：从与 positive 语义最近的科目中随机选取（top 3~10）

数据均衡策略：
  1. 文本归一化 + 噪音过滤（公司名→[对方单位]、数字/日期/单号删除）
  2. 软去重（同科目相同 anchor 最多保留 MAX_DUPLICATES_PER_ANCHOR 份，每份搭配不同负例）
  3. 超过 MAX_SAMPLES_PER_SUBJECT 则随机截断
  4. 验证集保留原始频率分布，不去重不截断
"""

from __future__ import annotations

import logging
import random
import re
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from db.reader import fetch_raw_pairs

logger = logging.getLogger(__name__)

# ── 噪音过滤正则 ────────────────────────────────────────────────────────────────
# 中文公司名：匹配到公司类后缀为止，向前贪婪抓取汉字/字母/数字/括号/空格
_RE_CN_COMPANY = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()·\s]{1,30}"
    r"(?:集团股份有限公司|股份有限公司|集团有限公司|有限责任公司|有限合伙企业|"
    r"合伙企业|有限公司|集团公司|事务所|研究院|集团)"
)
# 英文公司名：匹配各类英文后缀（全词，不区分大小写）
_RE_EN_COMPANY = re.compile(
    r"\b[\w\s\-&\.]{1,40}?"
    r"(?:Co\.,?\s*Ltd\.?|Limited|Ltd\.?|Corp\.?|Corporation|Inc\.?|LLC|L\.L\.C\.?|"
    r"Holdings?|Group|Pte\.?\s*Ltd\.?)\b",
    re.IGNORECASE,
)
# 纯数字串 ≥ 6 位（账号/流水号）
_RE_LONG_NUM = re.compile(r"\b\d{6,}\b")
# 日期（2024-01-15 / 2024/01/15 / 20240115）
_RE_DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{8}\b")
# 字母数字混合单号（含大写字母且长度≥6，如 TF2024001234、SF1234567890）
_RE_ORDER_NO = re.compile(r"\b(?=[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b")


def _denoise_text(text: str) -> str:
    """
    去除流水摘要中的噪音实体，保留语义关键词。

    去除顺序：日期 → 纯数字串 → 字母数字单号 → 英文公司名 → 中文公司名
    公司名替换为 [对方单位]，其余直接删除。
    """
    text = _RE_DATE.sub("", text)
    text = _RE_LONG_NUM.sub("", text)
    text = _RE_ORDER_NO.sub("", text)
    text = _RE_EN_COMPANY.sub("[对方单位]", text)
    text = _RE_CN_COMPANY.sub("[对方单位]", text)
    # 合并多个连续 [对方单位]
    text = re.sub(r"(\[对方单位\]\s*){2,}", "[对方单位] ", text)
    return re.sub(r"\s+", " ", text).strip()


class TrainingTriplet(NamedTuple):
    anchor: str    # 流水描述（含前缀标签）
    positive: str  # 正确科目名
    negative: str  # 错误科目名（负例）


def _normalize_anchor(
    summary: str,
    trade_type: str,
    type_: int = 0,
    currency: str = "",
    money: float = 0.0,
) -> str:
    """
    归一化 anchor：加收支方向 / 币种 / 金额档位前缀，合并多余空白。

    type_=1 (DEBIT 借方)  → [收入]
    type_=2 (CREDIT 贷方) → [支出]
    非人民币              → [USD] / [HKD] 等
    money >= 500000       → [大额]
    money >= 10000        → [中额]
    money > 0             → [小额]
    """
    direction = {1: "[收入] ", 2: "[支出] "}.get(type_, "")
    _cny = {"cny", "rmb", "人民币", ""}
    currency_tag = "" if currency.lower() in _cny else f"[{currency.upper()}] "
    if money >= 500_000:
        money_tag = "[大额] "
    elif money >= 10_000:
        money_tag = "[中额] "
    elif money > 0:
        money_tag = "[小额] "
    else:
        money_tag = ""
    text = f"{summary} {trade_type}" if trade_type else summary
    text = _denoise_text(text)
    return direction + currency_tag + money_tag + re.sub(r"\s+", " ", text).strip()


def _build_hard_negative_index(
    subjects: list[str],
    model_path: str,
    top_k: int = 10,
) -> dict[str, list[str]]:
    """
    用 BGE 模型预计算所有科目 embedding，
    为每个科目返回语义最近的 top_k 个错误科目列表（排除自身）。
    负例从 top 3~top_k 中随机采样，避开 top-1（防止真正的近义科目被当负例）。
    """
    if len(subjects) < 2:
        return {}
    model = SentenceTransformer(model_path)
    embeddings = model.encode(subjects, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings)
    sim_matrix = embeddings @ embeddings.T  # (N, N) 余弦相似度

    index: dict[str, list[str]] = {}
    for i, subj in enumerate(subjects):
        # 按相似度降序排列，排除自身（i）
        ranked = np.argsort(sim_matrix[i])[::-1]
        candidates = [subjects[j] for j in ranked if j != i]
        # 保留 top_k 个，实际采样时跳过 top-1（index 0 = 最近的）
        index[subj] = candidates[:top_k]
    return index


def _sample_negative(
    positive: str,
    anchor: str,
    same_dir_subjects: list[str],
    all_subjects: list[str],
    hard_index: dict[str, list[str]] | None = None,
) -> str | None:
    """
    硬负例采样（优先）：从与 positive 语义最近的 top 3~10 中随机选。
    回退策略：同方向随机 → 全科目随机。
    """
    if hard_index and positive in hard_index:
        candidates = hard_index[positive]
        # 跳过 top-1（最相似，可能是真近义），从 top-2 开始最多取到 top-10
        pool = candidates[1:] if len(candidates) > 1 else candidates
        # 进一步限制在同方向科目内（若可用候选足够）
        same_dir_set = set(same_dir_subjects)
        dir_pool = [s for s in pool if s in same_dir_set and s != positive]
        if dir_pool:
            return random.choice(dir_pool)
        if pool:
            return random.choice(pool)

    # 回退：同方向随机
    pool = [s for s in same_dir_subjects if s != positive]
    if not pool:
        pool = [s for s in all_subjects if s != positive]
    return random.choice(pool) if pool else None


def build_triplets(
    raw_pairs: list[dict],
    hard_index: dict[str, list[str]] | None = None,
) -> list[TrainingTriplet]:
    """
    将原始流水记录转换为训练三元组（去重均衡后）。

    Steps:
      1. 归一化 anchor（含去噪）
      2. 软去重（同科目相同 anchor 最多保留 MAX_DUPLICATES_PER_ANCHOR 份）
      3. 超 cap 随机截断
      4. 硬负例采样（优先），回退到同方向随机
    """
    cap = settings.MAX_SAMPLES_PER_SUBJECT
    dup_cap = settings.MAX_DUPLICATES_PER_ANCHOR
    subject_to_anchors: dict[str, dict[str, int]] = {}
    for row in raw_pairs:
        summary = (row.get("summary") or "").strip()
        trade_type = (row.get("trade_type") or "").strip()
        type_ = int(row.get("type") or 0)
        currency = (row.get("currency") or "").strip()
        money = float(row.get("money") or 0.0)
        anchor = _normalize_anchor(summary, trade_type, type_, currency, money)
        if not anchor:
            continue
        subject: str = row["correct_subject"]
        anchor_counts = subject_to_anchors.setdefault(subject, {})
        if anchor_counts.get(anchor, 0) < dup_cap:
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

    all_subjects = list(subject_to_anchors.keys())
    if len(all_subjects) < 2:
        logger.warning("科目种类不足（%d 种），无法构建负例", len(all_subjects))
        return []

    # 建立方向 → 科目列表的索引（一个科目可同时出现在两个方向）
    income_subjects: list[str] = []
    expense_subjects: list[str] = []
    for subject, anchor_dict in subject_to_anchors.items():
        if any(a.startswith("[收入]") for a in anchor_dict):
            income_subjects.append(subject)
        if any(a.startswith("[支出]") for a in anchor_dict):
            expense_subjects.append(subject)

    balanced_pairs: list[tuple[str, str]] = []
    for subject, anchor_counts in subject_to_anchors.items():
        # 展开：每个 anchor 按其 count 重复，再整体截断到 cap
        anchors: list[str] = []
        for anchor, cnt in anchor_counts.items():
            anchors.extend([anchor] * cnt)
        if cap > 0 and len(anchors) > cap:
            anchors = random.sample(anchors, cap)
        for anchor in anchors:
            balanced_pairs.append((anchor, subject))

    triplets: list[TrainingTriplet] = []
    for anchor, positive in balanced_pairs:
        if anchor.startswith("[收入]"):
            same_dir = income_subjects
        elif anchor.startswith("[支出]"):
            same_dir = expense_subjects
        else:
            same_dir = all_subjects
        neg = _sample_negative(positive, anchor, same_dir, all_subjects, hard_index)
        if neg is None:
            continue
        triplets.append(TrainingTriplet(anchor=anchor, positive=positive, negative=neg))

    logger.info(
        "构建训练三元组 %d 条（科目种类 %d 种，收入侧 %d 种，支出侧 %d 种）",
        len(triplets), len(all_subjects), len(income_subjects), len(expense_subjects),
    )
    return triplets


def build_train_val(
    raw_pairs: list[dict],
    val_ratio: float = 0.2,
) -> tuple[list[TrainingTriplet], list[TrainingTriplet]]:
    """
    将原始流水记录切分为训练集和验证集。

    - 训练集：精确去重 + 均衡截断（消除高频科目垄断）
    - 验证集：仅归一化，保留原始频率分布（评估更贴近生产）
    """
    random.shuffle(raw_pairs)
    val_size = max(1, int(len(raw_pairs) * val_ratio))
    val_raw = raw_pairs[:val_size]
    train_raw = raw_pairs[val_size:]

    # 预计算硬负例索引（用 base 模型，科目列表来自全量数据）
    all_subjects_full = list({row["correct_subject"] for row in raw_pairs})
    logger.info("预计算硬负例索引（共 %d 个科目）...", len(all_subjects_full))
    hard_index = _build_hard_negative_index(all_subjects_full, settings.BASE_MODEL_PATH)

    train_triplets = build_triplets(train_raw, hard_index)

    # 验证集用全量科目建负例池（同方向优先），不去重
    all_subjects = all_subjects_full
    # 构建方向索引（基于 type 字段，1=收入 2=支出）
    income_subjects_val = list({
        row["correct_subject"] for row in raw_pairs if int(row.get("type") or 0) == 1
    })
    expense_subjects_val = list({
        row["correct_subject"] for row in raw_pairs if int(row.get("type") or 0) == 2
    })
    val_triplets: list[TrainingTriplet] = []
    for row in val_raw:
        summary = (row.get("summary") or "").strip()
        trade_type = (row.get("trade_type") or "").strip()
        type_ = int(row.get("type") or 0)
        currency = (row.get("currency") or "").strip()
        money = float(row.get("money") or 0.0)
        anchor = _normalize_anchor(summary, trade_type, type_, currency, money)
        if not anchor:
            continue
        positive: str = row["correct_subject"]
        if type_ == 1:
            same_dir = income_subjects_val
        elif type_ == 2:
            same_dir = expense_subjects_val
        else:
            same_dir = all_subjects
        neg = _sample_negative(positive, anchor, same_dir, all_subjects, hard_index)
        if neg is None:
            continue
        val_triplets.append(TrainingTriplet(anchor=anchor, positive=positive, negative=neg))

    return train_triplets, val_triplets


def build_train_val_from_db(
    min_samples: int | None = None,
    limit: int = 50000,
    last_id: int = 0,
    val_ratio: float = 0.2,
) -> tuple[list[TrainingTriplet], list[TrainingTriplet], list[int], int]:
    """
    从只读库拉取数据并构建训练集 / 验证集。

    Returns:
        (train_triplets, val_triplets, ids, new_last_id)
        new_last_id — 本批次最大流水 ID，写入 checkpoint 防重复训练
    """
    if min_samples is None:
        min_samples = settings.MIN_TRAIN_SAMPLES

    raw_pairs = fetch_raw_pairs(limit, last_id)

    if len(raw_pairs) < min_samples:
        logger.info(
            "标注流水 %d 条，未达到训练阈值 %d 条，跳过本次训练",
            len(raw_pairs),
            min_samples,
        )
        return [], [], [], last_id

    ids: list[int] = [row["id"] for row in raw_pairs]
    new_last_id = max(ids) if ids else last_id

    train_triplets, val_triplets = build_train_val(raw_pairs, val_ratio)
    return train_triplets, val_triplets, ids, new_last_id
