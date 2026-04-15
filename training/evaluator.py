"""
验证集准确率评估

策略：对每条 anchor，在所有唯一 positive 科目中找向量最近邻，
命中正确科目则计为正确（Top-1 余弦相似度）。
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from training.data_builder import TrainingTriplet

logger = logging.getLogger(__name__)


def evaluate(
    model: SentenceTransformer,
    triplets: list[TrainingTriplet],
) -> float:
    """
    在给定三元组集合上评估 Top-1 准确率。

    Returns:
        accuracy: 0.0 ~ 1.0
    """
    if not triplets:
        return 0.0

    subjects = list({t.positive for t in triplets})
    if len(subjects) < 2:
        logger.warning("验证集科目种类不足（%d 种），跳过评估", len(subjects))
        return 0.0

    subject_embs: np.ndarray = model.encode(
        subjects, normalize_embeddings=True, show_progress_bar=False
    )
    anchors = [t.anchor for t in triplets]
    anchor_embs: np.ndarray = model.encode(
        anchors, normalize_embeddings=True, show_progress_bar=False
    )

    scores = anchor_embs @ subject_embs.T  # (n_anchors, n_subjects)
    pred_indices = np.argmax(scores, axis=1)

    correct = sum(
        1
        for pred_idx, triplet in zip(pred_indices, triplets)
        if subjects[pred_idx] == triplet.positive
    )
    accuracy = correct / len(triplets)

    # Macro 准确率（按科目平均）：不受高频科目数量影响，能暴露真正的难点科目
    subj_correct: dict[str, int] = {}
    subj_total: dict[str, int] = {}
    for pred_idx, triplet in zip(pred_indices, triplets):
        s = triplet.positive
        subj_total[s] = subj_total.get(s, 0) + 1
        if subjects[pred_idx] == s:
            subj_correct[s] = subj_correct.get(s, 0) + 1
    per_subj_acc = {s: subj_correct.get(s, 0) / subj_total[s] for s in subj_total}
    macro_acc = sum(per_subj_acc.values()) / len(per_subj_acc) if per_subj_acc else 0.0
    worst_5 = sorted(per_subj_acc.items(), key=lambda x: x[1])[:5]
    logger.info(
        "验证集准确率: micro=%.4f  macro=%.4f  (%d/%d) | 最难科目: %s",
        accuracy, macro_acc, correct, len(triplets),
        "  ".join(f"{s}={v:.2f}" for s, v in worst_5),
    )
    return accuracy
