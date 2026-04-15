"""
训练数据构建：从原始流水记录构建三元组（anchor, positive, negative）

anchor   - [收入]/[支出] + [币种] + [金额档位] + 流水摘要（去噪）+ 交易类型
positive - 正确科目名（account_chart.name）
negative - 硬负例：从与 positive 语义最近的科目中随机选取（top-k）

数据均衡策略：
  1. 文本归一化 + 噪音过滤（公司名→[对方单位]、数字/日期/单号删除）
  2. 冲突标注过滤（同一归一化 anchor 指向多个科目时多数投票），在 train/val 切割前执行
  3. 软去重（同科目相同 anchor 最多保留 MAX_DUPLICATES_PER_ANCHOR 份，每份无放回采样不同负例）
  4. 超过 MAX_SAMPLES_PER_SUBJECT 则随机截断
  5. 验证集按 anchor 去重（每个唯一 anchor 保留一条，防止高频科目虚高准确率）
"""

from __future__ import annotations

import logging
import os
import random
import re
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from db.reader import fetch_raw_pairs

logger = logging.getLogger(__name__)


def get_latest_finetuned_path() -> str | None:
    """返回最新 finetuned 版本的路径，若不存在则返回 None。"""
    base_dir = settings.FINETUNED_MODEL_DIR
    if not os.path.isdir(base_dir):
        return None
    dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    if not dirs:
        return None
    latest = f"v{max(int(d[1:]) for d in dirs)}"
    return os.path.join(base_dir, latest)

# ── 噪音过滤正则 ────────────────────────────────────────────────────────────────
# ⚠️  同步要求：以下去噪逻辑必须与 subject-matcher 的 app/model.py 完全一致。
#    修改任一侧后，另一侧也必须同步，并更新：
#    hcsw-audit-subject-matcher/tests/test_normalize_anchor_sync.py
# 中文公司名：匹配到公司类后缀为止，向前贪婪抓取汉字/字母/数字/括号/空格
_RE_CN_COMPANY = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()·\s]{1,30}"
    r"(?:集团股份有限公司|股份有限公司|集团有限公司|有限责任公司|有限合伙企业|"
    r"合伙企业|有限公司|集团公司|事务所|研究院|集团)"
)
# 英文公司名：匹配各类英文后缀（全词，不区分大小写）
# re.ASCII：确保 \w / \b 仅对 ASCII 字符生效，
# 否则 Python 默认 Unicode 模式下汉字也是 \w，导致汉字与英文之间无 \b，
# 紧跟在中文公司名后的英文地名前缀（如 "Weifang"）会被漏过。
_RE_EN_COMPANY = re.compile(
    r"\b[\w\s\-&\.]{1,40}?"
    r"(?:Co\.,?\s*Ltd\.?|Limited|Ltd\.?|Corp\.?|Corporation|Inc\.?|LLC|L\.L\.C\.?|"
    r"Holdings?|Group|Pte\.?\s*Ltd\.?)\b",
    re.IGNORECASE | re.ASCII,
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
    top_k: int = 5,
) -> dict[str, list[str]]:
    """
    用指定模型预计算所有科目 embedding，
    为每个科目返回语义最近的 top_k 个错误科目列表（排除自身）。
    负例从 top-1 ~ top_k 中随机采样（包含 top-1，迫使模型区分最相近的科目）。
    建议第一批用 base 模型计算，后续批次传入最新 finetuned 模型路径以保持索引新鲜度。
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
        # 保留 top_k 个（含 top-1），迫使模型学习区分最相近的科目
        index[subj] = candidates[:top_k]
    return index


def _get_negative_pool(
    positive: str,
    same_dir_subjects: list[str],
    all_subjects: list[str],
    hard_index: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    返回可用的负例候选池（已排除 positive 本身）。
    优先：hard_index 同方向 → hard_index 全量 → 同方向随机 → 全科目随机。
    """
    if hard_index and positive in hard_index:
        candidates = hard_index[positive]
        # 直接从 top-k 中选，不跳过 top-1：
        # top-1 往往就是最需要对比的混淆科目（如"应收董事款" vs "其他应付款-董事往来款"），
        # 跳过它会导致最难混淆对从不进入训练。
        pool = [s for s in candidates if s != positive]
        same_dir_set = set(same_dir_subjects)
        dir_pool = [s for s in pool if s in same_dir_set]
        if dir_pool:
            return dir_pool
        if pool:
            return pool
    # 回退：同方向随机
    pool = [s for s in same_dir_subjects if s != positive]
    if not pool:
        pool = [s for s in all_subjects if s != positive]
    return pool


def _sample_negative(
    positive: str,
    same_dir_subjects: list[str],
    all_subjects: list[str],
    hard_index: dict[str, list[str]] | None = None,
) -> str | None:
    """单次采样（验证集专用）。训练集请用 _get_negative_pool + random.sample。"""
    pool = _get_negative_pool(positive, same_dir_subjects, all_subjects, hard_index)
    return random.choice(pool) if pool else None


def _filter_conflicting_anchors(raw_pairs: list[dict]) -> list[dict]:
    """
    全量数据上检测冲突标注（同一归一化 anchor 指向多个科目），
    多数投票保留多数派行，丢弃少数派行。

    必须在 train/val 切割之前调用，确保两侧均不受矛盾标注污染。
    平局时取字典序最小的科目名（保证确定性），并记录 WARNING 提示人工审核。
    """
    # Step 1: 统计每个 (anchor, subject) 的出现次数
    anchor_subj_cnt: dict[str, dict[str, int]] = {}
    for row in raw_pairs:
        summary = (row.get("summary") or "").strip()
        trade_type = (row.get("trade_type") or "").strip()
        type_ = int(row.get("type") or 0)
        currency = (row.get("currency") or "").strip()
        money = float(row.get("money") or 0.0)
        anc = _normalize_anchor(summary, trade_type, type_, currency, money)
        if not anc:
            continue
        subj: str = row["correct_subject"]
        anchor_subj_cnt.setdefault(anc, {})
        anchor_subj_cnt[anc][subj] = anchor_subj_cnt[anc].get(subj, 0) + 1

    # Step 2: 找出所有冲突 anchor 的少数派 (anchor, subject)
    losing: set[tuple[str, str]] = set()
    for anc, subj_cnts in anchor_subj_cnt.items():
        if len(subj_cnts) <= 1:
            continue
        max_cnt = max(subj_cnts.values())
        winners = [s for s, c in subj_cnts.items() if c == max_cnt]
        if len(winners) > 1:
            # 平局：取字典序最小（确定性），提示人工审核
            winner = min(winners)
            logger.warning(
                "冲突标注平局 anchor=%r 涉及科目 %s（各 %d 条）→ 自动选 %r，建议人工审核",
                anc, winners, max_cnt, winner,
            )
        else:
            winner = winners[0]
            logger.warning(
                "冲突标注 anchor=%r 共 %d 种科目（%s）→ 多数投票保留 %r(%d条)，丢弃其余",
                anc, len(subj_cnts),
                ", ".join(f"{s}:{c}" for s, c in subj_cnts.items()),
                winner, max_cnt,
            )
        for subj in subj_cnts:
            if subj != winner:
                losing.add((anc, subj))

    if not losing:
        return raw_pairs

    # Step 3: 过滤原始行
    filtered: list[dict] = []
    removed = 0
    for row in raw_pairs:
        summary = (row.get("summary") or "").strip()
        trade_type = (row.get("trade_type") or "").strip()
        type_ = int(row.get("type") or 0)
        currency = (row.get("currency") or "").strip()
        money = float(row.get("money") or 0.0)
        anc = _normalize_anchor(summary, trade_type, type_, currency, money)
        subj = row.get("correct_subject", "")
        if anc and (anc, subj) in losing:
            removed += 1
        else:
            filtered.append(row)

    logger.info(
        "冲突标注过滤：丢弃 %d 条少数派记录（%d → %d 条）",
        removed, len(raw_pairs), len(filtered),
    )
    return filtered


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

    # ── Fix 1: 展开后按 (anchor, subject) 分组，无放回采样负例 ─────────────────
    pair_counts: dict[tuple[str, str], int] = {}
    for subject, anchor_counts in subject_to_anchors.items():
        # 展开：每个 anchor 按其 count 重复，再整体截断到 cap
        anchors_flat: list[str] = []
        for anchor, cnt in anchor_counts.items():
            anchors_flat.extend([anchor] * cnt)
        if cap > 0 and len(anchors_flat) > cap:
            anchors_flat = random.sample(anchors_flat, cap)
        for anchor in anchors_flat:
            key = (anchor, subject)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    triplets: list[TrainingTriplet] = []
    for (anchor, positive), cnt in pair_counts.items():
        if anchor.startswith("[收入]"):
            same_dir = income_subjects
        elif anchor.startswith("[支出]"):
            same_dir = expense_subjects
        else:
            same_dir = all_subjects
        neg_pool = _get_negative_pool(positive, same_dir, all_subjects, hard_index)
        if not neg_pool:
            continue
        # 无放回采样：保证同一 (anchor, positive) 的 cnt 份使用不同负例；
        # 当候选池不足时，补充有放回采样
        if len(neg_pool) >= cnt:
            negs = random.sample(neg_pool, cnt)
        else:
            negs = list(neg_pool) + [
                random.choice(neg_pool) for _ in range(cnt - len(neg_pool))
            ]
        for neg in negs:
            triplets.append(TrainingTriplet(anchor=anchor, positive=positive, negative=neg))

    logger.info(
        "构建训练三元组 %d 条（科目种类 %d 种，收入侧 %d 种，支出侧 %d 种）",
        len(triplets), len(all_subjects), len(income_subjects), len(expense_subjects),
    )
    return triplets


def build_train_val(
    raw_pairs: list[dict],
    val_ratio: float = 0.2,
    index_model_path: str | None = None,
) -> tuple[list[TrainingTriplet], list[TrainingTriplet]]:
    """
    将原始流水记录切分为训练集和验证集。

    - 训练集：精确去重 + 均衡截断（消除高频科目垄断）
    - 验证集：仅归一化，保留原始频率分布（评估更贴近生产）
    - index_model_path：用于计算硬负例索引的模型路径，
      None 时使用 BASE_MODEL_PATH；传入最新 finetuned 路径可保持索引新鲜度。
    """
    raw_pairs = _filter_conflicting_anchors(raw_pairs)
    random.shuffle(raw_pairs)
    val_size = max(1, int(len(raw_pairs) * val_ratio))
    val_raw = raw_pairs[:val_size]
    train_raw = raw_pairs[val_size:]

    # 预计算硬负例索引：优先用显式传入路径，其次自动探测最新 finetuned，最后回退 base
    _latest_ft = get_latest_finetuned_path()
    model_for_index = index_model_path or _latest_ft or settings.BASE_MODEL_PATH
    all_subjects_full = list({row["correct_subject"] for row in raw_pairs})
    logger.info("预计算硬负例索引（共 %d 个科目，模型: %s）...", len(all_subjects_full), model_for_index)
    hard_index = _build_hard_negative_index(all_subjects_full, model_for_index)

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
        neg = _sample_negative(positive, same_dir, all_subjects, hard_index)
        if neg is None:
            continue
        val_triplets.append(TrainingTriplet(anchor=anchor, positive=positive, negative=neg))

    # ── Fix 2: 验证集按 anchor 去重 ──────────────────────────────────────────────
    # 同一 anchor 只保留第一条，防止高频科目的重复 anchor 虚高准确率。
    # 目标：验证集衡量「不同类型描述的识别能力」，而不是「高频描述被反复答对」。
    seen_val_anchors: set[str] = set()
    deduped_val: list[TrainingTriplet] = []
    for t in val_triplets:
        if t.anchor not in seen_val_anchors:
            deduped_val.append(t)
            seen_val_anchors.add(t.anchor)
    logger.info("验证集去重：%d → %d 条（唯一 anchor）", len(val_triplets), len(deduped_val))
    val_triplets = deduped_val

    return train_triplets, val_triplets


def build_train_val_from_db(
    min_samples: int | None = None,
    limit: int = 50000,
    last_id: int = 0,
    val_ratio: float = 0.2,
    index_model_path: str | None = None,
) -> tuple[list[TrainingTriplet], list[TrainingTriplet], list[int], int]:
    """
    从只读库拉取数据并构建训练集 / 验证集。

    Returns:
        (train_triplets, val_triplets, ids, new_last_id)
        new_last_id — 本批次最大流水 ID，写入 checkpoint 防重复训练
    index_model_path — 用于计算硬负例索引的模型路径，None 时用 BASE_MODEL_PATH
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

    train_triplets, val_triplets = build_train_val(raw_pairs, val_ratio, index_model_path)
    return train_triplets, val_triplets, ids, new_last_id
