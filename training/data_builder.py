"""
训练数据构建：从原始流水记录构建三元组（anchor, positive, negative）

anchor   - [收入]/[支出] + [币种] + [金额档位] + 流水摘要 + 交易类型
positive - 正确科目名（account_chart.name）
negative - 随机采样的错误科目名（in-batch negative）

数据均衡策略：
  1. 文本归一化（合并多余空白）
  2. 按科目精确去重（同科目相同 anchor 只保留一条）
  3. 超过 MAX_SAMPLES_PER_SUBJECT 则随机截断
  4. 验证集保留原始频率分布，不去重不截断
"""

from __future__ import annotations

import logging
import random
import re
from typing import NamedTuple

from config.settings import settings
from db.reader import fetch_raw_pairs

logger = logging.getLogger(__name__)


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
    return direction + currency_tag + money_tag + re.sub(r"\s+", " ", text).strip()


def build_triplets(raw_pairs: list[dict]) -> list[TrainingTriplet]:
    """
    将原始流水记录转换为训练三元组（去重均衡后）。

    Steps:
      1. 归一化 anchor
      2. 按科目精确去重
      3. 超 cap 随机截断
      4. 随机采样负例
    """
    cap = settings.MAX_SAMPLES_PER_SUBJECT
    subject_to_anchors: dict[str, dict[str, None]] = {}
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
        subject_to_anchors.setdefault(subject, {})[anchor] = None

    all_subjects = list(subject_to_anchors.keys())
    if len(all_subjects) < 2:
        logger.warning("科目种类不足（%d 种），无法构建负例", len(all_subjects))
        return []

    balanced_pairs: list[tuple[str, str]] = []
    for subject, anchor_dict in subject_to_anchors.items():
        anchors = list(anchor_dict.keys())
        if cap > 0 and len(anchors) > cap:
            anchors = random.sample(anchors, cap)
        for anchor in anchors:
            balanced_pairs.append((anchor, subject))

    triplets: list[TrainingTriplet] = []
    for anchor, positive in balanced_pairs:
        negatives = [s for s in all_subjects if s != positive]
        if not negatives:
            continue
        triplets.append(
            TrainingTriplet(
                anchor=anchor,
                positive=positive,
                negative=random.choice(negatives),
            )
        )

    logger.info(
        "构建训练三元组 %d 条（科目种类 %d 种）", len(triplets), len(all_subjects)
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

    train_triplets = build_triplets(train_raw)

    # 验证集用全量科目建负例池，不去重
    all_subjects = list({row["correct_subject"] for row in raw_pairs})
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
        negatives = [s for s in all_subjects if s != positive]
        if not negatives:
            continue
        val_triplets.append(
            TrainingTriplet(
                anchor=anchor,
                positive=positive,
                negative=random.choice(negatives),
            )
        )

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
